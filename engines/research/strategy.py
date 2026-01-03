# =============================================================================
# FILENAME: engines/research/strategy.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/strategy.py
# DEPENDENCIES: shared, river, engines.research.backtester
# DESCRIPTION: The Adaptive Strategy Kernel (Research/Backtest Version).
# 
# PHOENIX STRATEGY V10.0 (RESEARCH PARITY):
# 1. PARITY: Implements _calculate_golden_trio to match Live Predictor.
# 2. LOGIC: Enforces strict V10.0 Anti-Chop (Hurst) and Efficiency (KER) gates.
# 3. MODEL: Aligned with Adaptive Random Forest (ARF) signal generation.
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
from datetime import datetime

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
    Strictly implements the Phoenix V10.0 Defensive System for Research Parity.
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
        self.sl_atr_mult = float(self.risk_conf.get('stop_loss_atr_mult', 2.0))
        
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
        self.burn_in_limit = params.get('burn_in_periods', 200) 
        self.burn_in_counter = 0
        self.burn_in_complete = False
        
        # State
        self.last_features = None
        self.last_price = 0.0
        self.last_price_map = {}
        self.bars_processed = 0
        self.consecutive_losses = 0 
        
        # --- V10.0 GOLDEN TRIO BUFFERS (Research Parity) ---
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
        
        # --- V10.0 MOMENTUM INDICATORS ---
        self.bb_window = 20
        self.bb_std = 2.0
        self.bb_buffer = deque(maxlen=self.bb_window)
        
        # --- FORENSIC RECORDER ---
        self.decision_log = deque(maxlen=1000)
        self.trade_events = []
        self.rejection_stats = defaultdict(int) 
        self.feature_importance_counter = Counter() 
        
        # --- PHOENIX V10.0 CONFIGURATION ---
        phx_conf = CONFIG.get('phoenix_strategy', {})
        
        # 1. Trend Filter: ENABLED (V10 Requirement)
        self.require_d1_trend = True 
        
        # 2. Efficiency Filter: STRICT (V10 Requirement)
        config_ker = float(phx_conf.get('ker_trend_threshold', 0.10))
        self.ker_thresh = max(0.25, config_ker) # Hard floor at 0.25
        
        self.vol_gate_ratio = float(phx_conf.get('volume_gate_ratio', 1.1)) 
        self.max_rvol_thresh = float(phx_conf.get('max_relative_volume', 8.0))
        self.chop_threshold = float(phx_conf.get('choppiness_threshold', 50.0)) 
        
        adx_cfg = CONFIG.get('features', {}).get('adx', {})
        self.adx_threshold = float(params.get('adx_threshold', adx_cfg.get('threshold', 20.0)))
        
        self.limit_order_offset_pips = CONFIG.get('trading', {}).get('limit_order_offset_pips', 0.2)
        
        # Friday Liquidation
        self.friday_entry_cutoff = self.risk_conf.get('friday_entry_cutoff_hour', 16)
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

    def on_data(self, snapshot: MarketSnapshot, broker: BacktestBroker):
        """
        Main Event Loop for the Strategy (V10.0 Research).
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

        # --- V10.0: GOLDEN TRIO INJECTION ---
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
        # E. ENTRY LOGIC GATES (V10.0 DEFENSIVE)
        # ============================================================
        
        adx_val = features.get('adx', 0.0)
        choppiness = features.get('choppiness', 50.0)
        
        # G1: ANTI-CHOP (Hurst Filter)
        if hurst < 0.45:
            self.rejection_stats[f"Chop Regime (Hurst {hurst:.2f})"] += 1
            return

        # G2: FUEL GAUGE
        if rvol_val < self.vol_gate_ratio:
            self.rejection_stats[f"Low Fuel (RVol {rvol_val:.2f})"] += 1
            return

        # G3: EXHAUSTION
        if rvol_val > self.max_rvol_thresh:
            self.rejection_stats[f"Volume Climax"] += 1
            return 
        
        # G4: STRICT EFFICIENCY (KER)
        base_thresh = max(0.25, self.ker_thresh)
        effective_ker_thresh = max(0.25, base_thresh + self.dynamic_ker_offset)
            
        if ker_val < effective_ker_thresh:
            self.rejection_stats[f"Low Efficiency (KER {ker_val:.2f})"] += 1
            return

        # G5: TREND STRENGTH
        if adx_val < self.adx_threshold:
            self.rejection_stats[f"Weak Trend"] += 1
            return

        proposed_action = 0 
        regime_label = "Trend"

        # --- MOMENTUM BREAKOUT TRIGGER ---
        if len(self.bb_buffer) < self.bb_window:
            self.rejection_stats["Warming Up BB"] += 1
            return
            
        bb_mu = np.mean(self.bb_buffer)
        bb_std = np.std(self.bb_buffer)
        upper_bb = bb_mu + (self.bb_std * bb_std)
        lower_bb = bb_mu - (self.bb_std * bb_std)
        
        rsi_val = features.get('rsi_norm', 0.5) * 100.0
        
        trigger_bull = (price > upper_bb) and (rsi_val > 50)
        trigger_bear = (price < lower_bb) and (rsi_val < 50)

        if trigger_bull:
            proposed_action = 1
            regime_label = "Breakout-Long"
        elif trigger_bear:
            proposed_action = -1
            regime_label = "Breakout-Short"
        else:
            self.rejection_stats["No Breakout"] += 1
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
                threshold=float(self.params.get('meta_labeling_threshold', 0.50))
            )
            
            if proposed_action != 0:
                self.meta_label_events += 1

            min_prob = float(self.params.get('min_calibrated_probability', 0.55))
            if confidence < min_prob:
                self.rejection_stats[f"Low Confidence ({confidence:.2f} < {min_prob:.2f})"] += 1
                return

            # --- SNIPER PROTOCOL ---
            if not self._check_sniper_filters(proposed_action, price):
                self.rejection_stats["Sniper Reject"] += 1
                return

            if is_profitable:
                self._execute_entry(confidence, price, features, broker, dt_ts, proposed_action, regime_label)
            else:
                self.rejection_stats['Meta-Labeler Reject'] += 1

        except Exception as e:
            if self.debug_mode: logger.error(f"Strategy Error: {e}")
            pass

    def _execute_entry(self, confidence, price, features, broker, dt_timestamp, action_int, regime):
        """
        Executes the trade entry logic with Fixed Risk.
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
            performance_score=effective_sqn 
        )

        if trade_intent.volume <= 0:
            self.rejection_stats[f"Risk Zero ({trade_intent.comment})"] += 1
            return

        stop_dist = current_atr * self.sl_atr_mult
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
                'entry_price_snap': price
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
            
            if r_multiple >= 1.0 and r_multiple < 1.5:
                target_sl = pos.entry_price
                if pos.side == 1 and target_sl > pos.stop_loss:
                    new_sl = target_sl
                    reason = "BE (1R)"
                elif pos.side == -1 and target_sl < pos.stop_loss:
                    new_sl = target_sl
                    reason = "BE (1R)"

            elif r_multiple >= 1.5:
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
                if "Trail" in reason or "BE" in reason:
                     if reason not in pos.comment:
                         pos.comment += f"|{reason}"
                     if self.debug_mode:
                         logger.info(f"🛡️ {self.symbol} SL Moved to {new_sl:.5f} ({reason})")

    def _check_daily_loss_limit(self, broker: BacktestBroker) -> bool:
        """Global Account Circuit Breaker (Max 4.9% DD)"""
        try:
            current_dd_pct = (broker.initial_balance - broker.equity) / broker.initial_balance
            if current_dd_pct > 0.049: 
                return True
            return False
        except:
            return False

    def _check_symbol_circuit_breaker(self, broker: BacktestBroker, server_time: datetime) -> bool:
        """
        V10.0 FEATURE: Per-Symbol Daily Circuit Breaker.
        """
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
        """
        Approximates D1 and H4 context from the M5 stream.
        """
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
        V10.0 SNIPER PROTOCOL:
        1. Trend Filter: Trade ONLY if price aligns with D1 EMA 200.
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
        d1_ema = self.current_d1_ema
        trend_aligned = False
        
        if d1_ema == 0:
            self.rejection_stats['Trend_Filter_Warmup'] += 1
            return False

        if signal == 1: # BUY Signal
            if price > d1_ema:
                trend_aligned = True
            else:
                self.rejection_stats['Counter_Trend_D1_EMA'] += 1
                return False
                
            if rsi > 70:
                self.rejection_stats['Overbought_RSI_>70'] += 1
                return False
                
        elif signal == -1: # SELL Signal
            if price < d1_ema:
                trend_aligned = True
            else:
                self.rejection_stats['Counter_Trend_D1_EMA'] += 1
                return False
                
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