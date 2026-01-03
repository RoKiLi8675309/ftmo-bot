# =============================================================================
# FILENAME: engines/research/strategy.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/strategy.py
# DEPENDENCIES: shared, river, engines.research.backtester
# DESCRIPTION: The Adaptive Strategy Kernel (Backtesting Version).
# 
# PHOENIX STRATEGY V12.2 (ROBUST REGIME LOGIC):
# 1. LOGIC: Implemented Priority Regime Detection (Trend > Reversion > Neutral).
#    Ensures USDCAD/USDCHF are correctly identified as Mean Reversion despite 'USD'.
# 2. RISK: Profit Buffer Scaling active.
# =============================================================================
import logging
import sys
import numpy as np
import random
import math
import pytz
import json
from collections import deque, defaultdict, Counter
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timedelta

# Third-Party Imports
from river import drift

# Shared Imports
from shared import (
    CONFIG,
    OnlineFeatureEngineer,
    AdaptiveTripleBarrier,
    ProbabilityCalibrator,
    RiskManager,
    TradeContext,
    LogSymbols,
    Trade
)
# New Feature Import
from shared.financial.features import MetaLabeler

# Local Imports
from engines.research.backtester import MarketSnapshot, BacktestBroker, BacktestOrder

logger = logging.getLogger("ResearchStrategy")

class ResearchStrategy:
    """
    Represents an independent trading agent for a single symbol.
    Manages its own Feature Engineering, Adaptive Labeler, and River Model.
    Strictly implements the Phoenix V12.0 Alpha Seeker Protocol.
    """
    def __init__(self, model: Any, symbol: str, params: dict[str, Any]):
        self.model = model
        self.symbol = symbol
        self.params = params
        self.debug_mode = False
        
        # 1. Feature Engineer (The Eyes)
        self.fe = OnlineFeatureEngineer(
            window_size=params.get('window_size', 50)
        )
        
        # --- RISK CONFIGURATION ---
        self.risk_conf = params.get('risk_management', CONFIG.get('risk_management', {}))
        # V11.1: Tighter stops (1.5 ATR) by default
        self.sl_atr_mult = float(self.risk_conf.get('stop_loss_atr_mult', 1.5))
        
        # 2. Adaptive Triple Barrier Labeler (The Teacher)
        tbm_conf = params.get('tbm', {})
        risk_mult_conf = self.sl_atr_mult
        self.optimized_reward_mult = float(tbm_conf.get('barrier_width', 3.0))
        
        self.labeler = AdaptiveTripleBarrier(
            horizon_ticks=int(tbm_conf.get('horizon_minutes', 120)), 
            risk_mult=risk_mult_conf, 
            reward_mult=self.optimized_reward_mult,
            drift_threshold=float(tbm_conf.get('drift_threshold', 1.5))
        )
        
        # 3. Meta Labeler (The Gatekeeper)
        self.meta_labeler = MetaLabeler()
        self.meta_label_events = 0 
        
        # 4. Probability Calibrators
        self.calibrator_buy = ProbabilityCalibrator(window=2000)
        self.calibrator_sell = ProbabilityCalibrator(window=2000)
        
        # 5. Warm-up State
        self.burn_in_limit = params.get('burn_in_periods', 100) # Reduced for V11
        self.burn_in_counter = 0
        self.burn_in_complete = False
        
        # State
        self.last_features = None
        self.last_price = 0.0
        self.last_price_map = {}
        self.bars_processed = 0
        self.consecutive_losses = 0 
        
        # --- V11.0 GOLDEN TRIO BUFFERS ---
        self.window_size_trio = 100
        self.closes_buffer = deque(maxlen=self.window_size_trio)
        self.volume_buffer = deque(maxlen=self.window_size_trio)
        
        # MTF Simulation State
        self.h4_buffer = deque(maxlen=200) 
        self.d1_buffer = deque(maxlen=200) 
        self.last_h4_idx = -1
        self.last_d1_idx = -1
        self.current_d1_ema = 0.0 
        
        # --- SNIPER PROTOCOL INDICATORS ---
        self.sniper_closes = deque(maxlen=200)      
        self.sniper_rsi = deque(maxlen=15)   
        
        # --- V11.1 MOMENTUM INDICATORS ---
        self.bb_window = 20
        self.bb_std = 1.5 # V11.1 FIX: Lowered from 2.0 to 1.5 to catch starts of moves
        self.bb_buffer = deque(maxlen=self.bb_window)
        
        # --- FORENSIC RECORDER ---
        self.decision_log = deque(maxlen=1000)
        self.trade_events = []
        self.rejection_stats = defaultdict(int) 
        self.feature_importance_counter = Counter() 
        
        # --- PHOENIX V12.0 CONFIGURATION ---
        phx_conf = CONFIG.get('phoenix_strategy', {})
        
        # V12.0 Logic Thresholds
        self.ker_floor = float(phx_conf.get('ker_trend_threshold', 0.02)) 
        self.hurst_breakout = float(phx_conf.get('hurst_breakout_threshold', 0.55))
        self.hurst_mean_rev = float(phx_conf.get('hurst_mean_reversion_threshold', 0.45))
        self.rvol_trigger = float(phx_conf.get('rvol_volatility_trigger', 3.0))
        self.require_d1_trend = phx_conf.get('require_d1_trend', False)
        
        # V12.0 ASSET MAP
        self.asset_regime_map = phx_conf.get('asset_regime_map', {})
        
        self.vol_gate_ratio = float(phx_conf.get('volume_gate_ratio', 1.1)) 
        self.max_rvol_thresh = float(phx_conf.get('max_relative_volume', 8.0))
        self.chop_threshold = float(phx_conf.get('choppiness_threshold', 60.0)) # Relaxed
        
        adx_cfg = CONFIG.get('features', {}).get('adx', {})
        # Relaxed ADX to 20.0
        self.adx_threshold = float(params.get('adx_threshold', adx_cfg.get('threshold', 20.0))) 
        
        self.limit_order_offset_pips = CONFIG.get('trading', {}).get('limit_order_offset_pips', 0.2)
        
        # Friday Liquidation (Aggressor: 18:00 cutoff)
        self.friday_entry_cutoff = self.risk_conf.get('friday_entry_cutoff_hour', 18)
        self.friday_close_hour = self.risk_conf.get('friday_liquidation_hour_server', 21)
        
        # Timezone Handling
        tz_str = self.risk_conf.get('risk_timezone', 'Europe/Prague')
        try:
            self.server_tz = pytz.timezone(tz_str)
        except Exception:
            self.server_tz = pytz.timezone('Europe/Prague')
        
        # --- REC 1: Dynamic Gate Scaling State ---
        self.ker_drift_detector = drift.ADWIN(delta=0.01)
        self.dynamic_ker_offset = 0.0

        # --- REC 3: Daily Circuit Breaker State ---
        self.daily_max_losses = 2
        self.daily_max_loss_pct = 0.01 

    def _calculate_golden_trio(self) -> Tuple[float, float, float]:
        """
        Calculates the "Golden Trio" of features locally using accurate buffers.
        Replicates Live Predictor logic exactly.
        """
        closes = self.closes_buffer
        vols = self.volume_buffer
        
        hurst = 0.5
        ker = 0.5
        rvol = 1.0
        
        if len(closes) < 30:
            return hurst, ker, rvol

        prices = np.array(closes)
        
        # 1. Simple Hurst
        try:
            lags = range(2, 20)
            tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] * 2.0 
            hurst = max(0.0, min(1.0, hurst))
        except:
            hurst = 0.5

        # 2. KER
        try:
            diffs = np.diff(prices)
            net_change = abs(prices[-1] - prices[0])
            sum_changes = np.sum(np.abs(diffs))
            if sum_changes > 0:
                ker = net_change / sum_changes
            else:
                ker = 0.0
        except:
            ker = 0.0

        # 3. RVOL
        if len(vols) > 10:
            curr_vol = vols[-1]
            avg_vol = np.mean(list(vols)[:-1]) 
            if avg_vol > 0:
                rvol = curr_vol / avg_vol
            else:
                rvol = 1.0
        
        return hurst, ker, rvol

    def _calibrate_confidence(self, raw_conf: float) -> float:
        """
        Calibrates raw model probability to a more reliable confidence score.
        """
        x = (raw_conf - 0.5) * 10.0 
        calibrated = 1 / (1 + np.exp(-x))
        return calibrated

    def _get_preferred_regime(self, symbol: str) -> str:
        """
        V12.2: ROBUST ASSET PERSONALITY DETECTION.
        Prevents 'USD' (Neutral) from overriding 'CAD' (MeanRev) in USDCAD.
        Logic: Trend > MeanReversion > Neutral.
        """
        # 1. Check for Exact Match (e.g. if we map "USDCAD" explicitly)
        if symbol in self.asset_regime_map:
            return self.asset_regime_map[symbol]
        
        # 2. Scan Components (Priority Logic)
        regimes_found = []
        for key, regime in self.asset_regime_map.items():
            if key in symbol:
                regimes_found.append(regime)
        
        # Priority 1: Trend Breakout (Aggressor)
        # If ANY component is Trend (e.g., JPY in GBPJPY), we treat it as Trend.
        if "TREND_BREAKOUT" in regimes_found:
            return "TREND_BREAKOUT"
            
        # Priority 2: Mean Reversion
        # If NO Trend, but ANY component is Mean Rev (e.g., CAD in USDCAD), treat as Mean Rev.
        if "MEAN_REVERSION" in regimes_found:
            return "MEAN_REVERSION"
            
        # Priority 3: Neutral (Default for EURUSD if defined as such)
        return "NEUTRAL"

    def on_data(self, snapshot: MarketSnapshot, broker: BacktestBroker):
        """
        Main Event Loop for the Strategy (V12.0 Alpha Seeker).
        """
        price = snapshot.get_price(self.symbol, 'close')
        high = snapshot.get_high(self.symbol)
        low = snapshot.get_low(self.symbol)
        volume = snapshot.get_price(self.symbol, 'volume')
        
        if price <= 0: return

        # Update Price Map
        self.last_price_map = snapshot.to_price_dict()
        if self.symbol not in self.last_price_map:
            self.last_price_map[self.symbol] = price

        # Update Buffers (Golden Trio + Sniper)
        self.closes_buffer.append(price)
        self.volume_buffer.append(volume)
        
        self.sniper_closes.append(price)
        if len(self.sniper_closes) > 1:
            delta = self.sniper_closes[-1] - self.sniper_closes[-2]
            self.sniper_rsi.append(delta)
        self.bb_buffer.append(price)

        # Inject Aux Data
        self._inject_auxiliary_data()

        # Timestamp Handling
        dt_ts = datetime.now()
        try:
            if hasattr(snapshot.timestamp, 'timestamp'):
                timestamp = snapshot.timestamp.timestamp()
                dt_ts = snapshot.timestamp
            else:
                timestamp = float(snapshot.timestamp)
                dt_ts = datetime.fromtimestamp(timestamp)
        except Exception:
            timestamp = 0.0

        if dt_ts.tzinfo is None:
            dt_ts = dt_ts.replace(tzinfo=pytz.utc)
        server_time = dt_ts.astimezone(self.server_tz)

        # --- UPDATE MTF CONTEXT ---
        context_data = self._simulate_mtf_context(price, server_time)
        self.current_d1_ema = context_data.get('d1', {}).get('ema200', 0.0)

        # --- GLOBAL ACCOUNT GUARD ---
        if self._check_daily_loss_limit(broker):
            self.rejection_stats["Account Daily Limit Hit"] += 1
            return

        # --- SYMBOL CIRCUIT BREAKER ---
        if self._check_symbol_circuit_breaker(broker, server_time):
            self.rejection_stats["Symbol Circuit Breaker (Loss Limit)"] += 1
            return

        self._update_streak_status(broker)
        self._manage_trailing_stops(broker, price, dt_ts)
        self._manage_time_stops(broker, dt_ts) # V11.1 NEW: Force close stale trades

        # --- FRIDAY LOGIC ---
        if server_time.weekday() == 4 and server_time.hour >= self.friday_close_hour:
            if self.symbol in broker.positions:
                pos = broker.positions[self.symbol]
                broker._close_partial_position(pos, pos.quantity, price, dt_ts, "Friday Liquidation")
            return 

        if server_time.weekday() == 4 and server_time.hour >= self.friday_entry_cutoff:
            return 

        # Flow Volumes
        buy_vol = snapshot.get_price(self.symbol, 'buy_vol')
        sell_vol = snapshot.get_price(self.symbol, 'sell_vol')
        
        # Retail Fallback
        if buy_vol == 0 and sell_vol == 0:
            if self.last_price > 0:
                if price > self.last_price:
                    buy_vol = volume; sell_vol = 0.0
                elif price < self.last_price:
                    buy_vol = 0.0; sell_vol = volume
                else:
                    buy_vol = volume / 2.0; sell_vol = volume / 2.0
        
        self.last_price = price

        # A. Feature Engineering
        features = self.fe.update(
            price=price,
            timestamp=timestamp,
            volume=volume,
            high=high,
            low=low,
            buy_vol=buy_vol,
            sell_vol=sell_vol,
            time_feats={},
            context_data=context_data
        )
        
        if features is None: return

        # --- V12.0: GOLDEN TRIO INJECTION ---
        hurst, ker_val, rvol_val = self._calculate_golden_trio()
        
        features['hurst'] = hurst
        features['ker'] = ker_val
        features['rvol'] = rvol_val
        
        self.last_features = features
        
        if self.burn_in_counter < self.burn_in_limit:
            self.burn_in_counter += 1
            if self.burn_in_counter == self.burn_in_limit:
                self.burn_in_complete = True
            return

        self.bars_processed += 1

        # Drift Detection
        self.ker_drift_detector.update(ker_val)
        if self.ker_drift_detector.drift_detected:
            self.dynamic_ker_offset = max(-0.10, self.dynamic_ker_offset - 0.05)
        else:
            self.dynamic_ker_offset = min(0.0, self.dynamic_ker_offset + 0.001)

        # B. Delayed Training
        resolved_labels = self.labeler.resolve_labels(high, low, current_close=price)
        if resolved_labels:
            for (stored_feats, outcome_label, realized_ret) in resolved_labels:
                w_pos = float(self.params.get('positive_class_weight', 1.5))
                w_neg = float(self.params.get('negative_class_weight', 1.0))
                base_weight = w_pos if outcome_label != 0 else w_neg
                
                ret_scalar = math.log1p(abs(realized_ret) * 100.0)
                ret_scalar = max(0.5, min(ret_scalar, 5.0))
                
                hist_ker = stored_feats.get('ker', 0.5)
                ker_weight = hist_ker * 2.0 
                
                final_weight = base_weight * ret_scalar * ker_weight
                
                self.model.learn_one(stored_feats, outcome_label, sample_weight=final_weight)
                
                if outcome_label != 0:
                      self.model.learn_one(stored_feats, outcome_label, sample_weight=final_weight * 1.5)
                    
                if outcome_label != 0:
                    self.meta_labeler.update(stored_feats, primary_action=outcome_label, outcome_pnl=realized_ret)

        # C. Add Trade Opportunity
        parkinson = features.get('parkinson_vol', 0.0)
        current_atr = features.get('atr', 0.001)
        self.labeler.add_trade_opportunity(features, price, current_atr, timestamp, parkinson_vol=parkinson)

        # D. ACTIVE TRADE MANAGEMENT
        if self.symbol in broker.positions:
            return 

        # ============================================================
        # E. AGGRESSOR GATES & ASSET PERSONALITY (V12.0)
        # ============================================================
        
        # G1: EFFICIENCY (KER) - AGGRESSIVE (V11.0)
        base_thresh = self.ker_floor 
        effective_ker_thresh = max(0.005, base_thresh + self.dynamic_ker_offset)
            
        if ker_val < effective_ker_thresh:
            self.rejection_stats[f"Low Efficiency (KER {ker_val:.3f} < {effective_ker_thresh:.3f})"] += 1
            return

        # G2: REGIME IDENTIFICATION (Hurst)
        # Determine Preferred Regime for this Asset
        preferred_regime = self._get_preferred_regime(self.symbol)
        
        regime_label = "Neutral"
        proposed_action = 0 # 0=Hold, 1=Buy, -1=Sell
        
        # Check Bollinger Bands state
        if len(self.bb_buffer) < self.bb_window:
            self.rejection_stats["Warming Up BB"] += 1
            return
            
        bb_mu = np.mean(self.bb_buffer)
        bb_std = np.std(self.bb_buffer)
        bb_mult = 1.5
        upper_bb = bb_mu + (bb_mult * bb_std)
        lower_bb = bb_mu - (bb_mult * bb_std)
        
        # --- V12.0 LOGIC MAPPING ---
        # 1. DETECT PHYSICS
        is_trending = hurst > self.hurst_breakout
        is_reverting = hurst < self.hurst_mean_rev
        
        # 2. FILTER BY PERSONALITY (Robust Match)
        if is_trending:
            if preferred_regime == "MEAN_REVERSION":
                self.rejection_stats["Personality Clash (Trend on MeanRev Asset)"] += 1
                return 
            
            # Valid Trend Signal
            regime_label = "TREND_BREAKOUT"
            if price > upper_bb:
                proposed_action = 1 # Breakout Buy
            elif price < lower_bb:
                proposed_action = -1 # Breakout Sell
                
        elif is_reverting:
            if preferred_regime == "TREND_BREAKOUT":
                self.rejection_stats["Personality Clash (MeanRev on Trend Asset)"] += 1
                return
                
            # Valid Reversion Signal
            regime_label = "MEAN_REVERSION"
            if price > upper_bb:
                proposed_action = -1 # Fade Buy (Short the top)
            elif price < lower_bb:
                proposed_action = 1 # Fade Sell (Long the bottom)
                
        else:
            # NEUTRAL ZONE
            self.rejection_stats[f"Random Walk Regime (H={hurst:.2f})"] += 1
            return

        if proposed_action == 0:
            self.rejection_stats["No Trigger"] += 1
            return

        # G3: EXHAUSTION
        if rvol_val > self.max_rvol_thresh:
            self.rejection_stats[f"Volume Climax"] += 1
            return 
        
        # G4: TREND STRENGTH (ADX) - Only for Trend Regime
        if regime_label == "TREND_BREAKOUT":
            adx_val = features.get('adx', 0.0)
            if adx_val < self.adx_threshold: # Relaxed to 20.0
                self.rejection_stats[f"Weak Trend"] += 1
                return

        # ============================================================
        # F. ML CONFIRMATION & EXECUTION
        # ============================================================
        
        try:
            pred_proba = self.model.predict_proba_one(features)
            prob_buy = pred_proba.get(1, 0.0)
            prob_sell = pred_proba.get(-1, 0.0)
            raw_confidence = prob_buy if proposed_action == 1 else prob_sell
            
            # Calibrate
            confidence = self._calibrate_confidence(raw_confidence)
            
            is_profitable = self.meta_labeler.predict(
                features, 
                proposed_action, 
                threshold=float(self.params.get('meta_labeling_threshold', 0.52))
            )
            
            if proposed_action != 0:
                self.meta_label_events += 1

            min_prob = float(self.params.get('min_calibrated_probability', 0.52))
            if confidence < min_prob:
                self.rejection_stats[f"Low Confidence ({confidence:.2f} < {min_prob:.2f})"] += 1
                return

            # --- SNIPER PROTOCOL ---
            if not self._check_sniper_filters(proposed_action, price):
                self.rejection_stats["Sniper Reject"] += 1
                return

            if is_profitable:
                # V11.0: Tighten Stops Logic (RVOL Trigger)
                tighten_stops = (rvol_val > self.rvol_trigger)
                self._execute_entry(confidence, price, features, broker, dt_ts, proposed_action, regime_label, tighten_stops)
            else:
                self.rejection_stats['Meta-Labeler Reject'] += 1

        except Exception as e:
            if self.debug_mode: logger.error(f"Strategy Error: {e}")
            pass

    def _execute_entry(self, confidence, price, features, broker, dt_timestamp, action_int, regime, tighten_stops):
        """
        Executes the trade entry logic with Fixed Risk.
        V12.0: Includes Profit Buffer Scaling Logic.
        """
        action = "BUY" if action_int == 1 else "SELL"
        
        exposure_count = 0
        quote_currency = self.symbol[-3:] 
        for pos in broker.open_positions:
            if quote_currency in pos.symbol: 
                exposure_count += 1
        
        ctx = TradeContext(
            symbol=self.symbol,
            price=price,
            stop_loss_price=0.0,
            account_equity=broker.equity,
            account_currency="USD",
            win_rate=0.45, 
            risk_reward_ratio=self.optimized_reward_mult 
        )

        current_atr = features.get('atr', 0.001)
        current_ker = features.get('ker', 1.0)
        volatility = features.get('volatility', 0.001)
        risk_override = self.params.get('risk_per_trade_percent')
        
        sqn_score = self._calculate_symbol_sqn(broker)
        
        if -3.0 < sqn_score < -1.0:
            effective_sqn = -0.99
        else:
            effective_sqn = sqn_score

        # V12.0: CALCULATE DAILY PNL PCT FOR BUFFER SCALING
        # (Start Equity - Current Equity) / Start Equity = Drawdown
        # Current Equity - Start Equity = Profit
        daily_pnl_val = broker.equity - broker.daily_start_equity
        daily_pnl_pct = daily_pnl_val / broker.daily_start_equity if broker.daily_start_equity > 0 else 0.0

        trade_intent, risk_usd = RiskManager.calculate_rck_size(
            context=ctx,
            conf=confidence,
            volatility=volatility,
            active_correlations=exposure_count,
            market_prices=self.last_price_map,
            atr=current_atr, 
            ker=current_ker,
            account_size=broker.equity,
            risk_percent_override=risk_override,
            performance_score=effective_sqn,
            daily_pnl_pct=daily_pnl_pct # V12.0 UPDATE
        )

        if trade_intent.volume <= 0:
            self.rejection_stats[f"Risk Zero ({trade_intent.comment})"] += 1
            return

        # V11.0: If Volatility Trigger is active (tighten_stops), reduce the ATR multiplier
        atr_mult = self.sl_atr_mult
        if tighten_stops:
            atr_mult = max(1.0, atr_mult * 0.75) 

        stop_dist = current_atr * atr_mult
        trade_intent.stop_loss = stop_dist
        dynamic_tp_dist = current_atr * self.optimized_reward_mult
        trade_intent.take_profit = dynamic_tp_dist
        
        trade_intent.action = action
        qty = trade_intent.volume
        tp_dist = trade_intent.take_profit

        if qty < 0.01:
            self.rejection_stats['Zero Size (< 0.01)'] += 1
            return

        if action == "BUY":
            sl_price = price - stop_dist
            tp_price = price + tp_dist
            side = 1
        else:
            sl_price = price + stop_dist
            tp_price = price - tp_dist
            side = -1

        order = BacktestOrder(
            symbol=self.symbol,
            side=side,
            quantity=qty,
            timestamp_created=dt_timestamp, 
            stop_loss=sl_price,
            take_profit=tp_price,
            comment=f"{trade_intent.comment}|{regime}",
            metadata={
                'regime': regime,
                'confidence': float(confidence),
                'rvol': features.get('rvol', 0),
                'parkinson': features.get('parkinson_vol', 0),
                'ker': current_ker,
                'atr': current_atr,
                'optimized_rr': self.optimized_reward_mult,
                'initial_risk_dist': stop_dist, 
                'entry_price_snap': price,
                'tighten_stops': tighten_stops
            }
        )
        
        broker.submit_order(order)
        
        imp_feats = []
        imp_feats.append(regime)
        if features.get('rvol', 0) > 2.0: imp_feats.append('High_Fuel')
        
        for f in imp_feats:
            self.feature_importance_counter[f] += 1

        self.trade_events.append({
            'time': dt_timestamp.timestamp(), 
            'action': action,
            'price': price,
            'conf': confidence,
            'rvol': features.get('rvol', 0),
            'ker': current_ker,
            'atr': current_atr,
            'regime': regime,
            'top_feats': imp_feats
        })

    def _calculate_symbol_sqn(self, broker: BacktestBroker) -> float:
        trades = [t.net_pnl for t in broker.closed_positions if t.symbol == self.symbol]
        if len(trades) < 5: 
            return 0.0
        window = trades[-50:]
        avg_pnl = np.mean(window)
        std_pnl = np.std(window)
        if std_pnl < 1e-9:
            return 0.0
        sqn = math.sqrt(len(window)) * (avg_pnl / std_pnl)
        return sqn

    def _manage_time_stops(self, broker: BacktestBroker, current_time: datetime):
        """
        V11.1 FEATURE: Time-Based Force Exit.
        """
        cutoff_seconds = 86400 
        
        to_close = []
        for pos in broker.open_positions:
            if pos.symbol != self.symbol: continue
            
            if pos.timestamp_created.tzinfo is None:
                pos_time = pos.timestamp_created.replace(tzinfo=pytz.utc)
            else:
                pos_time = pos.timestamp_created
                
            if current_time.tzinfo is None:
                curr_time_aware = current_time.replace(tzinfo=pytz.utc)
            else:
                curr_time_aware = current_time
            
            duration = (curr_time_aware - pos_time).total_seconds()
            
            if duration > cutoff_seconds:
                to_close.append(pos)
        
        for pos in to_close:
            broker._close_partial_position(
                pos, 
                pos.quantity, 
                self.last_price, 
                current_time, 
                "Time Stop (24h)"
            )

    def _manage_trailing_stops(self, broker: BacktestBroker, current_price: float, timestamp: datetime):
        for pos in broker.open_positions:
            if pos.symbol != self.symbol: continue
            
            risk_dist = pos.metadata.get('initial_risk_dist', 0.0)
            if risk_dist <= 0: continue
            
            if pos.side == 1: 
                dist_pnl = current_price - pos.entry_price
            else: 
                dist_pnl = pos.entry_price - current_price
                
            r_multiple = dist_pnl / risk_dist
            new_sl = None
            reason = ""
            
            # V11.1: Aggressive Scalp Trailing (0.5R Activation)
            if r_multiple >= 0.5 and r_multiple < 1.0:
                target_sl = pos.entry_price - (risk_dist * 0.5) if pos.side == 1 else pos.entry_price + (risk_dist * 0.5)
                
                if pos.side == 1 and target_sl > pos.stop_loss:
                    new_sl = target_sl
                    reason = "Tighten (0.5R)"
                elif pos.side == -1 and target_sl < pos.stop_loss:
                    new_sl = target_sl
                    reason = "Tighten (0.5R)"

            elif r_multiple >= 1.0:
                trail_dist = risk_dist * 0.5
                if pos.side == 1: 
                    target_sl = current_price - trail_dist
                    if target_sl > pos.stop_loss:
                        new_sl = target_sl
                        reason = f"Trail ({r_multiple:.1f}R)"
                else: 
                    target_sl = current_price + trail_dist
                    if target_sl < pos.stop_loss:
                        new_sl = target_sl
                        reason = f"Trail ({r_multiple:.1f}R)"
            
            if new_sl is not None:
                pos.stop_loss = new_sl
                if "Trail" in reason or "Tighten" in reason:
                      if reason not in pos.comment:
                          pos.comment += f"|{reason}"
                      if self.debug_mode:
                          logger.info(f"🛡️ {self.symbol} SL Moved to {new_sl:.5f} ({reason})")

    def _check_daily_loss_limit(self, broker: BacktestBroker) -> bool:
        try:
            current_dd_pct = (broker.initial_balance - broker.equity) / broker.initial_balance
            if current_dd_pct > 0.049: 
                return True
            return False
        except:
            return False

    def _check_symbol_circuit_breaker(self, broker: BacktestBroker, server_time: datetime) -> bool:
        today_losses = 0
        today_pnl = 0.0
        
        current_day_start = server_time.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        
        for trade in broker.closed_positions:
            if trade.symbol != self.symbol:
                continue
            
            exit_ts = trade.exit_time if hasattr(trade, 'exit_time') else 0.0
            
            if exit_ts >= current_day_start:
                today_pnl += trade.net_pnl
                if trade.net_pnl < 0:
                    today_losses += 1
        
        if today_losses >= self.daily_max_losses:
            return True
            
        loss_limit_usd = broker.initial_balance * self.daily_max_loss_pct
        if today_pnl < -loss_limit_usd:
            return True
            
        return False

    def _update_streak_status(self, broker: BacktestBroker):
        if not broker.closed_positions:
            self.consecutive_losses = 0
            return
            
        streak = 0
        for trade in reversed(broker.closed_positions):
            if trade.net_pnl < 0:
                streak += 1
            else:
                break
        self.consecutive_losses = streak

    def _simulate_mtf_context(self, price: float, dt: datetime) -> Dict[str, Any]:
        h4_idx = (dt.day * 6) + (dt.hour // 4)
        if h4_idx != self.last_h4_idx:
            self.h4_buffer.append(price)
            self.last_h4_idx = h4_idx
            
        d1_idx = dt.toordinal()
        if d1_idx != self.last_d1_idx:
            self.d1_buffer.append(price)
            self.last_d1_idx = d1_idx
            
        ctx = {'d1': {}, 'h4': {}}
        
        # D1 EMA 200 (Calculated on Daily Closes)
        if len(self.d1_buffer) > 0:
            arr = np.array(self.d1_buffer)
            if len(arr) < 200:
                ema = np.mean(arr)
            else:
                span = 200
                alpha = 2 / (span + 1)
                ema = arr[0]
                for x in arr[1:]:
                    ema = (alpha * x) + ((1 - alpha) * ema)
            ctx['d1']['ema200'] = ema
        else:
            ctx['d1']['ema200'] = 0.0 # Fallback
            
        if len(self.h4_buffer) > 14:
            arr = np.array(self.h4_buffer)
            changes = np.diff(arr)
            gains = changes[changes > 0]
            losses = -changes[changes < 0]
            avg_gain = np.mean(gains[-14:]) if len(gains) > 0 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) > 0 else 0
            
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            ctx['h4']['rsi'] = rsi
        else:
            ctx['h4']['rsi'] = 50.0
            
        return ctx

    def _check_sniper_filters(self, signal: int, price: float) -> bool:
        """
        V11.0 SNIPER PROTOCOL:
        1. Trend Filter: Trade ONLY if price aligns with D1 EMA 200.
           (DISABLED in V11 Aggressor Mode if require_d1_trend is False)
        2. RSI Guard: Block Buy if RSI > 70, Block Sell if RSI < 30.
        """
        # 1. RSI CALCULATION (M5 Extension)
        if len(self.sniper_rsi) < 14:
            rsi = 50.0 
        else:
            gains = [x for x in self.sniper_rsi if x > 0]
            losses = [abs(x) for x in self.sniper_rsi if x < 0]
            avg_gain = sum(gains) / 14 if gains else 0
            avg_loss = sum(losses) / 14 if losses else 1e-9
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # 2. TREND FILTER (D1 Bias)
        trend_aligned = False
        d1_ema = self.current_d1_ema
        
        if self.require_d1_trend and d1_ema > 0:
            if signal == 1: # BUY Signal
                if price > d1_ema: trend_aligned = True
                else:
                    self.rejection_stats['Counter_Trend_D1_EMA'] += 1
                    return False
            elif signal == -1: # SELL Signal
                if price < d1_ema: trend_aligned = True
                else:
                    self.rejection_stats['Counter_Trend_D1_EMA'] += 1
                    return False
        else:
            trend_aligned = True 

        # 3. RSI EXTREME GUARD
        if signal == 1: # BUY
            if rsi > 70:
                self.rejection_stats['Overbought_RSI_>70'] += 1
                return False
        elif signal == -1: # SELL
            if rsi < 30:
                self.rejection_stats['Oversold_RSI_<30'] += 1
                return False
        
        return True

    def _inject_auxiliary_data(self):
        defaults = {
            "USDJPY": 150.0, "GBPUSD": 1.25, "EURUSD": 1.08,
            "USDCAD": 1.35, "USDCHF": 0.90, "AUDUSD": 0.65, "NZDUSD": 0.60,
            "GBPJPY": 190.0, "EURJPY": 160.0, "AUDJPY": 95.0
        }
        for sym, price in defaults.items():
            if sym not in self.last_price_map:
                self.last_price_map[sym] = price

    def generate_autopsy(self) -> str:
        if not self.trade_events:
            sorted_rejects = sorted(self.rejection_stats.items(), key=lambda item: item[1], reverse=True)
            reject_str = ", ".join([f"{k}: {v}" for k, v in sorted_rejects[:5]])
            
            status = "Waiting for Warm-Up" if not self.burn_in_complete else "No Trigger Conditions Met"
            return f"AUTOPSY: No trades. Status: {status}. Top Rejections: {{{reject_str}}}. Bars processed: {self.bars_processed}"
        
        avg_conf = np.mean([t['conf'] for t in self.trade_events])
        avg_rvol = np.mean([t['rvol'] for t in self.trade_events]) 
        
        sorted_rejects = sorted(self.rejection_stats.items(), key=lambda item: item[1], reverse=True)
        reject_str = ", ".join([f"{k}: {v}" for k, v in sorted_rejects[:5]])
        
        top_features = self.feature_importance_counter.most_common(5)
        feat_str = str(top_features)
        
        report = (
            f"\n --- 💀 PHOENIX AUTOPSY ({self.symbol}) ---\n"
            f" Trades: {len(self.trade_events)}\n"
            f" Avg Conf: {avg_conf:.2f}\n"
            f" Avg RVol: {avg_rvol:.2f}\n"
            f" Top Drivers: {feat_str}\n"
            f" Rejections: {{{reject_str}}}\n"
            f" ----------------------------------------\n"
        )
        return report