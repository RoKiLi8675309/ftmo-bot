from __future__ import annotations
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
    Strictly implements the Sniper Survival Protocol (V17.0).
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
        # Standardize risk multiplier (Increased to 4.0 for V17.0 to survive noise)
        self.sl_atr_mult = float(self.risk_conf.get('stop_loss_atr_mult', 4.0))
        self.max_currency_exposure = int(self.risk_conf.get('max_currency_exposure', 8)) 
        self.cooldown_minutes = int(self.risk_conf.get('loss_cooldown_minutes', 120))
        
        # 2. Adaptive Triple Barrier Labeler (The Teacher)
        tbm_conf = params.get('tbm', {})
        risk_mult_conf = self.sl_atr_mult
        self.optimized_reward_mult = float(tbm_conf.get('barrier_width', 2.5)) 
        
        # Default Horizon Reduced to 240m (4h) for Scalping
        self.labeler = AdaptiveTripleBarrier(
            horizon_ticks=int(tbm_conf.get('horizon_minutes', 240)), 
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
        # Reduced burn-in limit to match Live (assuming data sufficiency)
        self.burn_in_limit = params.get('burn_in_periods', 30) 
        self.burn_in_counter = 0
        self.burn_in_complete = False
        
        # State
        self.last_features = None
        self.last_price = 0.0
        self.last_price_map = {}
        self.bars_processed = 0
        self.consecutive_losses = 0 
        
        # --- GOLDEN TRIO BUFFERS ---
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
        
        # --- MOMENTUM INDICATORS ---
        self.bb_window = 20
        self.bb_std = CONFIG['phoenix_strategy'].get('bb_std_dev', 1.5) # STRICTER (1.5)
        self.bb_buffer = deque(maxlen=self.bb_window)
        
        # --- NEW FILTERS ---
        # SMA 200 Trend Filter Buffer
        self.sma_window = deque(maxlen=200)
        # Volatility Gate Buffer (Returns)
        self.returns_window = deque(maxlen=20)
        
        # --- FORENSIC RECORDER ---
        self.decision_log = deque(maxlen=1000)
        self.trade_events = []
        self.rejection_stats = defaultdict(int) 
        self.feature_importance_counter = Counter() 
        
        # --- STRATEGY CONFIGURATION ---
        phx_conf = CONFIG.get('phoenix_strategy', {})
        
        # Logic Thresholds (Strict Tuning V17.0)
        self.ker_floor = float(phx_conf.get('ker_trend_threshold', 0.003)) 
        self.hurst_breakout = float(phx_conf.get('hurst_breakout_threshold', 0.48)) 
        self.rvol_trigger = float(phx_conf.get('rvol_volatility_trigger', 3.0))
        self.require_d1_trend = phx_conf.get('require_d1_trend', False)
        
        # Regime Enforcement
        self.regime_enforcement = phx_conf.get('regime_enforcement', 'DISABLED').upper()
        self.asset_regime_map = phx_conf.get('asset_regime_map', {})
        
        self.vol_gate_ratio = float(phx_conf.get('volume_gate_ratio', 0.8)) 
        self.max_rvol_thresh = float(phx_conf.get('max_relative_volume', 25.0))
        self.chop_threshold = float(phx_conf.get('choppiness_threshold', 60.0))
        
        adx_cfg = CONFIG.get('features', {}).get('adx', {})
        # STRICT ADX for V17.0
        self.adx_threshold = float(params.get('adx_threshold', adx_cfg.get('threshold', 20.0))) 
        
        self.limit_order_offset_pips = CONFIG.get('trading', {}).get('limit_order_offset_pips', 0.1)
        
        # --- SESSION CONTROL ---
        session_conf = self.risk_conf.get('session_control', {})
        self.session_enabled = session_conf.get('enabled', False)
        self.start_hour = session_conf.get('start_hour_server', 10)
        self.liq_hour = session_conf.get('liquidate_hour_server', 21)
        
        # Legacy Friday Liquidation (Reinforced)
        self.friday_entry_cutoff = self.risk_conf.get('friday_entry_cutoff_hour', 16)
        self.friday_close_hour = self.risk_conf.get('friday_liquidation_hour_server', 21)
        
        # Timezone Handling
        tz_str = self.risk_conf.get('risk_timezone', 'Europe/Prague')
        try:
            self.server_tz = pytz.timezone(tz_str)
        except Exception:
            self.server_tz = pytz.timezone('Europe/Prague')
        
        # --- Dynamic Gate Scaling State ---
        self.ker_drift_detector = drift.ADWIN(delta=0.01)
        self.dynamic_ker_offset = 0.0

        # --- Daily Circuit Breaker State ---
        self.daily_max_losses = 5 
        self.daily_max_loss_pct = 0.045

    def _calculate_golden_trio(self) -> Tuple[float, float, float]:
        """
        Calculates the "Golden Trio" of features locally using accurate buffers.
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
            # Math Guard: Variance check
            if np.var(prices) < 1e-9:
                hurst = 0.5
            else:
                lags = range(2, 20)
                tau = []
                for lag in lags:
                    diff = np.subtract(prices[lag:], prices[:-lag])
                    std = np.std(diff)
                    tau.append(std if std > 1e-9 else 1e-9)
                
                # Polyfit on log-log
                # Slope = H.
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst = poly[0] 
                hurst = max(0.0, min(1.0, hurst))
        except:
            hurst = 0.5

        # 2. KER
        try:
            diffs = np.diff(prices)
            net_change = abs(prices[-1] - prices[0])
            sum_changes = np.sum(np.abs(diffs))
            if sum_changes > 1e-9:
                ker = net_change / sum_changes
            else:
                ker = 0.0
        except:
            ker = 0.0

        # 3. RVOL
        if len(vols) > 10:
            curr_vol = vols[-1]
            avg_vol = np.mean(list(vols)[:-1]) 
            if avg_vol > 1e-9:
                rvol = curr_vol / avg_vol
            else:
                rvol = 1.0
        
        return hurst, ker, rvol

    def _calculate_trend_bias(self, current_price: float) -> int:
        """
        Returns 1 (Bullish), -1 (Bearish), or 0 (Neutral).
        Based on Price vs SMA(200).
        """
        if len(self.sma_window) < 200:
            return 0 
            
        sma_200 = sum(self.sma_window) / 200
        
        # 0.05% Filter Buffer
        threshold = sma_200 * 0.0005
        
        if current_price > (sma_200 + threshold):
            return 1
        elif current_price < (sma_200 - threshold):
            return -1
        else:
            return 0

    def _check_volatility_condition(self, current_price: float) -> bool:
        """
        Ensures distinct market movement exists before entering.
        """
        if len(self.sma_window) < 2:
            return True 
            
        # Calculate Log Return
        prev_price = self.sma_window[-2]
        if prev_price > 0:
            ret = math.log(current_price / prev_price)
            self.returns_window.append(ret)
            
        if len(self.returns_window) < 10:
            return True
            
        # Calculate Volatility (Std Dev of Returns)
        vol = np.std(list(self.returns_window))
        
        # Minimum Volatility Threshold
        MIN_VOLATILITY = 0.00015 
        
        if vol < MIN_VOLATILITY:
            return False
            
        return True

    def _calibrate_confidence(self, raw_conf: float) -> float:
        """
        Calibrates raw model probability to a more reliable confidence score.
        """
        x = (raw_conf - 0.5) * 10.0 
        calibrated = 1 / (1 + np.exp(-x))
        return calibrated

    def _get_preferred_regime(self, symbol: str) -> str:
        """
        ROBUST ASSET PERSONALITY DETECTION.
        """
        # 1. Check for Exact Match
        if symbol in self.asset_regime_map:
            return self.asset_regime_map[symbol]
        
        # 2. Scan Components (Priority Logic)
        regimes_found = []
        for key, regime in self.asset_regime_map.items():
            if key in symbol:
                regimes_found.append(regime)
        
        # Priority 1: Trend Breakout (Aggressor)
        if "TREND_BREAKOUT" in regimes_found:
            return "TREND_BREAKOUT"
            
        # Priority 2: Mean Reversion
        if "MEAN_REVERSION" in regimes_found:
            return "MEAN_REVERSION"
            
        # Priority 3: Neutral
        return "NEUTRAL"

    def _check_currency_exposure(self, broker: BacktestBroker, symbol: str) -> bool:
        """
        Enforces Portfolio Heat Limits.
        """
        base_ccy = symbol[:3]
        quote_ccy = symbol[3:]
        
        base_count = 0
        quote_count = 0
        
        for pos in broker.open_positions:
            pos_base = pos.symbol[:3]
            pos_quote = pos.symbol[3:]
            
            if base_ccy == pos_base or base_ccy == pos_quote:
                base_count += 1
            if quote_ccy == pos_base or quote_ccy == pos_quote:
                quote_count += 1
        
        if base_count >= self.max_currency_exposure:
            self.rejection_stats[f"Max Exposure ({base_ccy})"] += 1
            return False
            
        if quote_count >= self.max_currency_exposure:
            self.rejection_stats[f"Max Exposure ({quote_ccy})"] += 1
            return False
            
        return True

    def _check_revenge_guard(self, broker: BacktestBroker, current_time: datetime) -> bool:
        """
        MANDATORY COOLDOWN PROTOCOL.
        """
        # Get trades for this symbol, sorted by exit time
        my_closed_trades = [t for t in broker.closed_positions if t.symbol == self.symbol and t.close_time is not None]
        
        if not my_closed_trades:
            return False
            
        last_trade = my_closed_trades[-1]
        
        # Ensure timezone awareness match
        close_time = last_trade.close_time
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=current_time.tzinfo)
        else:
            close_time = close_time.astimezone(current_time.tzinfo)
            
        cooldown_expiry = close_time + timedelta(minutes=self.cooldown_minutes)
        
        if current_time < cooldown_expiry:
            return True 
            
        return False

    def on_data(self, snapshot: MarketSnapshot, broker: BacktestBroker):
        """
        Main Event Loop for the Strategy (Scalper Mode).
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

        # Update Buffers
        self.closes_buffer.append(price)
        self.volume_buffer.append(volume)
        self.sma_window.append(price)
        
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
            
        # --- REVENGE TRADING GUARD (MANDATORY COOLDOWN) ---
        if self._check_revenge_guard(broker, server_time):
            self.rejection_stats["Mandatory Cooldown (Revenge Guard)"] += 1
            return

        self._update_streak_status(broker)
        self._manage_trailing_stops(broker, price, dt_ts)
        self._manage_time_stops(broker, dt_ts)

        # --- GAP-PROOF LIQUIDATION (SESSION & WEEKEND) ---
        is_weekend_hold = (server_time.weekday() > 4) 
        is_friday_close = (server_time.weekday() == 4 and server_time.hour >= self.friday_close_hour)
        
        # Daily Session Liquidation (Hard Close 3h before NY Close)
        is_daily_session_close = False
        if self.session_enabled:
            if server_time.hour >= self.liq_hour:
                is_daily_session_close = True

        if is_weekend_hold or is_friday_close or is_daily_session_close:
            # Force close if we have an open position
            for pos in broker.open_positions:
                if pos.symbol == self.symbol:
                      broker._close_partial_position(
                        pos, 
                        pos.quantity, 
                        price, 
                        dt_ts, 
                        "Session/Friday Liquidation"
                    )
            return # Stop processing
        
        # --- SESSION START GUARD ---
        if self.session_enabled and server_time.hour < self.start_hour:
            return 

        # Friday Entry Guard
        if server_time.weekday() == 4 and server_time.hour >= self.friday_entry_cutoff:
            return 

        # Flow Volumes (Tick Rule Fallback)
        buy_vol = snapshot.get_price(self.symbol, 'buy_vol')
        sell_vol = snapshot.get_price(self.symbol, 'sell_vol')
        
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

        # --- GOLDEN TRIO INJECTION ---
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

        # D. ACTIVE TRADE MANAGEMENT (PYRAMIDING)
        existing_positions = [p for p in broker.open_positions if p.symbol == self.symbol]
        
        # ============================================================
        # E. SURVIVAL GATES & ASSET PERSONALITY
        # ============================================================
        
        # G1: EFFICIENCY (KER)
        base_thresh = self.ker_floor 
        effective_ker_thresh = max(0.001, base_thresh + self.dynamic_ker_offset)
            
        if ker_val < effective_ker_thresh:
            self.rejection_stats[f"Low Efficiency (KER {ker_val:.3f} < {effective_ker_thresh:.3f})"] += 1
            return

        # G2: REGIME IDENTIFICATION
        preferred_regime = self._get_preferred_regime(self.symbol)
        regime_label = "Neutral"
        proposed_action = 0 # 0=Hold, 1=Buy, -1=Sell
        is_regime_clash = False
        
        # Check Bollinger Bands state
        if len(self.bb_buffer) < self.bb_window:
            self.rejection_stats["Warming Up BB"] += 1
            return
            
        bb_mu = np.mean(self.bb_buffer)
        bb_std = np.std(self.bb_buffer)
        bb_mult = self.bb_std 
        
        upper_bb = bb_mu + (bb_mult * bb_std)
        lower_bb = bb_mu - (bb_mult * bb_std)
        
        # --- LOGIC MAPPING: TREND ONLY ---
        is_trending = hurst > self.hurst_breakout
        
        if is_trending:
            if preferred_regime == "MEAN_REVERSION":
                is_regime_clash = True
            
            # Trend Logic
            regime_label = "TREND_BREAKOUT"
            if price > upper_bb:
                proposed_action = 1 # Breakout Buy
            elif price < lower_bb:
                proposed_action = -1 # Breakout Sell
                
        else:
            # NEUTRAL ZONE 
            if self.regime_enforcement == "DISABLED":
                 # Low Hurst = Reversion. If hitting bands, fade it.
                 regime_label = "MEAN_REVERSION"
                 if price > upper_bb:
                     proposed_action = -1 # Reversion Sell
                 elif price < lower_bb:
                     proposed_action = 1  # Reversion Buy
            else:
                self.rejection_stats[f"Random Walk Regime (H={hurst:.2f})"] += 1
                return

        if proposed_action == 0:
            self.rejection_stats["No Trigger"] += 1
            return
            
        # --- GATE: VOLATILITY EXPANSION ---
        if not self._check_volatility_condition(price):
            self.rejection_stats["Low Volatility (Dead Zone)"] += 1
            return

        # --- REGIME ENFORCEMENT ---
        if is_regime_clash:
            if self.regime_enforcement == "HARD":
                self.rejection_stats[f"Personality Clash ({regime_label} on {preferred_regime})"] += 1
                return
            elif self.regime_enforcement == "DISABLED":
                 is_regime_clash = False 
            else:
                pass 

        # G3: EXHAUSTION
        if rvol_val > self.max_rvol_thresh:
            self.rejection_stats[f"Volume Climax"] += 1
            return 
        
        # G4: TREND STRENGTH (ADX) - HARD BLOCK
        # V17.0 FIX: If we want to trade trend, we MUST have ADX > 20. No compromises.
        if regime_label == "TREND_BREAKOUT":
            adx_val = features.get('adx', 0.0)
            if adx_val < self.adx_threshold:
                self.rejection_stats[f"Weak Trend (ADX {adx_val:.1f} < {self.adx_threshold})"] += 1
                return

        # ============================================================
        # F. ML CONFIRMATION & EXECUTION
        # ============================================================
        
        try:
            pred_proba = self.model.predict_proba_one(features)
            
            # SURVIVAL: Bypass Calibration
            confidence = 1.0 
            
            is_profitable = self.meta_labeler.predict(
                features, 
                proposed_action, 
                threshold=float(self.params.get('meta_labeling_threshold', 0.60)) # STRICTER
            )
            
            if proposed_action != 0:
                self.meta_label_events += 1

            if is_regime_clash:
               pass 

            # --- SNIPER PROTOCOL ---
            # Pass current_hurst to check for Ignition (but restricted)
            if not self._check_sniper_filters(proposed_action, price, hurst):
                self.rejection_stats["Sniper Reject (RSI Extreme)"] += 1
                return
            
            # --- PORTFOLIO HEAT ---
            if not self._check_currency_exposure(broker, self.symbol):
                return

            if is_profitable:
                # PYRAMIDING CHECK
                is_pyramid = False
                
                if existing_positions:
                    pyramid_config = self.risk_conf.get('pyramiding', {})
                    if not pyramid_config.get('enabled', False):
                        return # Standard Block
                    
                    # 1. Direction Check
                    last_pos = existing_positions[-1]
                    if (last_pos.side == 1 and proposed_action != 1) or \
                       (last_pos.side == -1 and proposed_action != -1):
                        return 

                    # 2. Max Adds Check
                    if len(existing_positions) > pyramid_config.get('max_adds', 1): # Reduced
                        return
                        
                    # 3. Profit Threshold Check
                    risk_dist = last_pos.metadata.get('initial_risk_dist', 1.0)
                    dist = (price - last_pos.entry_price) if last_pos.side == 1 else (last_pos.entry_price - price)
                    current_r = dist / risk_dist if risk_dist > 0 else 0
                    
                    if current_r < pyramid_config.get('add_on_profit_r', 1.0):
                        return 
                    
                    is_pyramid = True

                # Tighten Stops Logic (RVOL Trigger)
                tighten_stops = (rvol_val > self.rvol_trigger)
                self._execute_entry(confidence, price, features, broker, dt_ts, proposed_action, regime_label, tighten_stops, is_pyramid)
            else:
                self.rejection_stats['Meta-Labeler Reject'] += 1

        except Exception as e:
            if self.debug_mode: logger.error(f"Strategy Error: {e}")
            pass

    def _execute_entry(self, confidence, price, features, broker, dt_timestamp, action_int, regime, tighten_stops, is_pyramid=False):
        """
        Executes the trade entry logic.
        V17.0: Passes 'fixed_lots' method implicitly via RiskManager logic.
        """
        action = "BUY" if action_int == 1 else "SELL"
        
        exposure_count = 0
        quote_currency = self.symbol[-3:] 
        for pos in broker.open_positions:
            if quote_currency in pos.symbol: 
                exposure_count += 1
        
        # Check High Fuel (RVOL) for Reward Boost
        rvol = features.get('rvol', 0.0)
        current_reward = self.optimized_reward_mult
        
        if rvol > 3.0:
            current_reward = max(current_reward, 4.0)

        ctx = TradeContext(
            symbol=self.symbol,
            price=price,
            stop_loss_price=0.0,
            account_equity=broker.equity,
            account_currency="USD",
            win_rate=0.45, 
            risk_reward_ratio=current_reward 
        )

        current_atr = features.get('atr', 0.001)
        current_ker = features.get('ker', 1.0)
        volatility = features.get('volatility', 0.001)
        
        # Risk Override 
        if is_pyramid:
             risk_override = self.risk_conf.get('pyramiding', {}).get('risk_per_add_percent', 0.00)
        else:
             risk_override = self.params.get('risk_per_trade_percent')
        
        sqn_score = self._calculate_symbol_sqn(broker)
        
        if -3.0 < sqn_score < -1.0:
            effective_sqn = -0.99
        else:
            effective_sqn = sqn_score

        # CALCULATE DAILY PNL PCT
        daily_pnl_val = broker.equity - broker.daily_start_equity
        daily_pnl_pct = daily_pnl_val / broker.daily_start_equity if broker.daily_start_equity > 0 else 0.0

        # --- CALCULATE TOTAL OPEN RISK % ---
        current_open_risk_usd = 0.0
        used_margin = 0.0
        contract_size = 100000 
        
        for pos in broker.open_positions:
            price_dist = abs(pos.entry_price - pos.stop_loss)
            rate = RiskManager.get_conversion_rate(pos.symbol, pos.entry_price, self.last_price_map)
            risk_val = price_dist * pos.quantity * contract_size * rate
            current_open_risk_usd += risk_val
            
            trade_margin = RiskManager.calculate_required_margin(
                pos.symbol, pos.quantity, pos.entry_price, contract_size, rate
            )
            used_margin += trade_margin
            
        current_open_risk_pct = (current_open_risk_usd / broker.equity) * 100.0
        
        estimated_free_margin = max(0.0, broker.equity - used_margin)

        # CALL RISK MANAGER (V17.0: This will use fixed_lots if config says so)
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
            daily_pnl_pct=daily_pnl_pct,
            current_open_risk_pct=current_open_risk_pct,
            free_margin=estimated_free_margin 
        )

        if trade_intent.volume <= 0:
            self.rejection_stats[f"Risk Zero ({trade_intent.comment})"] += 1
            return

        # Explicitly set stops if not returned (fixed lots path might need this)
        if trade_intent.stop_loss == 0.0:
             # Recalculate geometry
             atr_mult_sl = self.sl_atr_mult
             stop_dist = current_atr * atr_mult_sl
             pip_val_raw, _ = RiskManager.get_pip_info(self.symbol)
             # Hard floor 25 pips
             min_stop = 25.0 * pip_val_raw
             stop_dist = max(stop_dist, min_stop)
             
             trade_intent.stop_loss = stop_dist
             trade_intent.take_profit = stop_dist * (current_reward)

        trade_intent.action = action
        qty = trade_intent.volume
        stop_dist = trade_intent.stop_loss
        tp_dist = trade_intent.take_profit

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
                'optimized_rr': current_reward,
                'initial_risk_dist': stop_dist, 
                'entry_price_snap': price,
                'tighten_stops': tighten_stops,
                'is_pyramid': is_pyramid
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
            'top_feats': imp_feats,
            'pyramid': is_pyramid
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
        Managed Time Exits.
        """
        # Fetch dynamic horizon from config (default 240m / 4h)
        tbm_conf = self.params.get('tbm', {})
        horizon_minutes = int(tbm_conf.get('horizon_minutes', 240))
        hard_stop_seconds = horizon_minutes * 60
        
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
            
            if duration > hard_stop_seconds:
                hours = int(horizon_minutes / 60)
                to_close.append((pos, f"Time Stop ({hours}h)"))
            
        for pos, reason in to_close:
            broker._close_partial_position(
                pos, 
                pos.quantity, 
                self.last_price, 
                current_time, 
                reason
            )

    def _manage_trailing_stops(self, broker: BacktestBroker, current_price: float, timestamp: datetime):
        """
        PATIENCE UPDATE: Wait for 1.5R before moving to Breakeven.
        """
        trail_conf = self.risk_conf.get('trailing_stop', {})
        activation_r = float(trail_conf.get('activation_r', 1.5))
        trail_dist_r = float(trail_conf.get('trail_dist_r', 1.0))

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
            
            if r_multiple >= activation_r:
                trail_pips = risk_dist * trail_dist_r
                
                if pos.side == 1: 
                    target_sl = current_price - trail_pips
                    min_lock = pos.entry_price + (risk_dist * 0.1)
                    target_sl = max(target_sl, min_lock)
                    
                    if target_sl > pos.stop_loss:
                        new_sl = target_sl
                        reason = f"Trail ({r_multiple:.1f}R)"
                        
                else: 
                    target_sl = current_price + trail_pips
                    min_lock = pos.entry_price - (risk_dist * 0.1)
                    target_sl = min(target_sl, min_lock)
                    
                    if target_sl < pos.stop_loss:
                        new_sl = target_sl
                        reason = f"Trail ({r_multiple:.1f}R)"
            
            if new_sl is not None:
                pos.stop_loss = new_sl
                if "Trail" in reason:
                      if reason not in pos.comment:
                          pos.comment += f"|{reason}"
                      if self.debug_mode:
                          logger.info(f"🛡️ {self.symbol} SL Moved to {new_sl:.5f} ({reason})")

    def _check_daily_loss_limit(self, broker: BacktestBroker) -> bool:
        try:
            current_dd_pct = (broker.initial_balance - broker.equity) / broker.initial_balance
            if current_dd_pct > 0.045: 
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
            ctx['d1']['ema200'] = 0.0
            
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

    def _check_sniper_filters(self, signal: int, price: float, current_hurst: float) -> bool:
        """
        V17.0 UPDATE:
        - REMOVED 'Ignition' Bypass.
        - Strict RSI logic: Buy Low, Sell High.
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

        # 2. RSI EXTREME GUARD (STRICT)
        # Prevent buying the top or selling the bottom.
        if "JPY" in self.symbol:
             # JPY pairs can trend harder, allow wider RSI
             upper = 80
             lower = 20
        else:
             upper = 75
             lower = 25

        if signal == 1: # BUY
            if rsi > upper: 
                self.rejection_stats['Overbought_RSI'] += 1
                return False # Exhaustion
                
        elif signal == -1: # SELL
            if rsi < lower: 
                self.rejection_stats['Oversold_RSI'] += 1
                return False # Exhaustion
        
        return True

    def _inject_auxiliary_data(self):
        defaults = {
            "USDJPY": 150.0, "GBPUSD": 1.25, "EURUSD": 1.08,
            "USDCAD": 1.35, "USDCHF": 0.90, "AUDUSD": 0.65, "NZDUSD": 0.60,
            "GBPJPY": 190.0, "EURJPY": 160.0, "AUDJPY": 95.0,
            "GBPAUD": 1.95, "EURAUD": 1.65, "GBPNZD": 2.05
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