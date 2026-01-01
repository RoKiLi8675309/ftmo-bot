# =============================================================================
# FILENAME: engines/research/strategy.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/strategy.py
# DEPENDENCIES: shared, river, engines.research.backtester
# DESCRIPTION: The Adaptive Strategy Kernel (Backtesting Version).
# 
# PHOENIX STRATEGY V7.5 (SNIPER PROTOCOL):
# 1. MEAN REVERSION PURGED: Removed all Regime C logic.
# 2. AGGRESSOR LOGIC:
#    - REGIME FILTER: KER > 0.3 (Trend Efficiency).
#    - FUEL GAUGE: RVOL > 2.0 (Momentum injection).
#    - TRIGGER: Order Flow Imbalance (Buy > 1.2x Sell).
# 3. SNIPER FILTERS (NEW):
#    - TREND: SMA 200 Filter (No counter-trend).
#    - EXTENSION: RSI Filter (No buying tops/selling bottoms).
# =============================================================================
import logging
import sys
import numpy as np
import random
import math
import pytz
from collections import deque, defaultdict, Counter
from typing import Any, Dict, Optional, List
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
    Strictly implements the Aggressor Breakout System with Sniper Protocols.
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
        
        # AUDIT FIX: Load SL Multiplier from Config (Was Hardcoded 1.5)
        self.sl_atr_mult = float(self.risk_conf.get('stop_loss_atr_mult', 1.5))
        
        # 2. Adaptive Triple Barrier Labeler (The Teacher)
        tbm_conf = params.get('tbm', {})
        
        # Pass the configurable Risk Multiplier to the Labeler
        risk_mult_conf = self.sl_atr_mult
        
        # Target Reward should align with Strategy (Use Optimization Param)
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
        self.consecutive_losses = 0 # Track Streak
        
        # MTF Simulation State (for backtesting consistency)
        self.h4_buffer = deque(maxlen=200) # Store H4 closes for RSI
        self.d1_buffer = deque(maxlen=200) # Store D1 closes for EMA
        self.last_h4_idx = -1
        self.last_d1_idx = -1
        
        # --- SNIPER PROTOCOL INDICATORS ---
        self.closes = deque(maxlen=200)     # For Trend (SMA 200)
        self.rsi_buffer = deque(maxlen=15)  # For Extension (RSI 14)
        
        # --- FORENSIC RECORDER ---
        self.decision_log = deque(maxlen=1000)
        self.trade_events = []
        self.rejection_stats = defaultdict(int) 
        self.feature_importance_counter = Counter() 
        
        # --- PHOENIX V7.0 PARAMETERS (AGGRESSOR) ---
        phx_conf = CONFIG.get('phoenix_strategy', {})
        
        # Gates
        self.require_d1_trend = phx_conf.get('require_d1_trend', True)
        self.require_h4_alignment = phx_conf.get('require_h4_alignment', True)
        
        # Safety Cap
        self.max_rvol_thresh = float(phx_conf.get('max_relative_volume', 6.0)) # Cap for huge spikes
        
        # --- STRICT THRESHOLDS (FROM DOCUMENT) ---
        self.ker_thresh = float(phx_conf.get('ker_trend_threshold', 0.30))
        self.vol_gate_ratio = float(phx_conf.get('volume_gate_ratio', 2.0))
        self.aggressor_ratio_min = 1.2 # "Buy Vol > Sell Vol * 1.2"
        self.adx_threshold = float(phx_conf.get('adx', {}).get('threshold', 25.0))
        
        # Aggressor Ratio Feature Threshold (Price Action proxy)
        self.aggressor_thresh_price = float(phx_conf.get('aggressor_threshold', 0.55))
        
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

    def on_data(self, snapshot: MarketSnapshot, broker: BacktestBroker):
        """
        Main Event Loop for the Strategy.
        """
        # Data Extraction
        price = snapshot.get_price(self.symbol, 'close')
        high = snapshot.get_high(self.symbol)
        low = snapshot.get_low(self.symbol)
        volume = snapshot.get_price(self.symbol, 'volume')
        
        if price <= 0: return

        # Update Price Map for Risk Calculation
        self.last_price_map = snapshot.to_price_dict()
        if self.symbol not in self.last_price_map:
            self.last_price_map[self.symbol] = price

        # Update Sniper Buffers (SMA 200 / RSI)
        self.closes.append(price)
        if len(self.closes) > 1:
            delta = self.closes[-1] - self.closes[-2]
            self.rsi_buffer.append(delta)

        # Inject Aux Data for single-symbol tests
        self._inject_auxiliary_data()

        # Robust Timestamp Extraction (Handle UTC properly)
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

        # --- TIMEZONE CONVERSION (Fixes Friday Guard Spam) ---
        # Ensure dt_ts is timezone-aware (UTC) then convert to Server Time (Prague)
        if dt_ts.tzinfo is None:
            dt_ts = dt_ts.replace(tzinfo=pytz.utc)
        
        server_time = dt_ts.astimezone(self.server_tz)

        # --- SIMULATED SESSION GUARD (Daily Loss Limit) ---
        if self._check_daily_loss_limit(broker):
            self.rejection_stats["Daily Loss Limit Hit"] += 1
            return

        # --- UPDATE STREAK STATUS (CRITICAL FOR FILTER) ---
        self._update_streak_status(broker)

        # --- FRIDAY LOGIC: LIQUIDATION VS ENTRY GUARD ---
        # 1. Liquidation (21:00 Server Time)
        if server_time.weekday() == 4 and server_time.hour >= self.friday_close_hour:
            if self.symbol in broker.positions:
                pos = broker.positions[self.symbol]
                broker._close_partial_position(pos, pos.quantity, price, dt_ts, "Friday Liquidation")
                if self.debug_mode: logger.info(f"🚫 {self.symbol} Liquidated for Weekend (Friday {server_time.hour}:00)")
            return 

        # 2. Entry Guard (16:00 Server Time) - Aggressive Filter
        # Stop new entries, but do not spam the logs
        if server_time.weekday() == 4 and server_time.hour >= self.friday_entry_cutoff:
            return 
        # -------------------------------------------------

        # Flow Volumes (L2 Proxies handled by Feature Engineer)
        buy_vol = snapshot.get_price(self.symbol, 'buy_vol')
        sell_vol = snapshot.get_price(self.symbol, 'sell_vol')
        
        # --- RETAIL FALLBACK LOGIC ---
        if buy_vol == 0 and sell_vol == 0:
            if self.last_price > 0:
                if price > self.last_price:
                    buy_vol = volume; sell_vol = 0.0
                elif price < self.last_price:
                    buy_vol = 0.0; sell_vol = volume
                else:
                    buy_vol = volume / 2.0; sell_vol = volume / 2.0
        
        self.last_price = price

        # --- MTF CONTEXT SIMULATION ---
        context_data = self._simulate_mtf_context(price, server_time)

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

        self.last_features = features
        
        # --- WARM-UP GATE ---
        if self.burn_in_counter < self.burn_in_limit:
            self.burn_in_counter += 1
            if self.burn_in_counter == self.burn_in_limit:
                self.burn_in_complete = True
            return

        self.bars_processed += 1

        # --- REC 1: DYNAMIC GATE SCALING (Drift Detection) ---
        # Update drift detector with current KER
        ker_val = features.get('ker', 0.5)
        self.ker_drift_detector.update(ker_val)
        
        # If distribution of KER changes significantly (Drift), adjust threshold
        if self.ker_drift_detector.drift_detected:
            # Drift implies regime shift. Relax gate temporarily.
            self.dynamic_ker_offset = max(-0.10, self.dynamic_ker_offset - 0.05)
        else:
            # Slowly decay offset back to 0 (Normalization)
            self.dynamic_ker_offset = min(0.0, self.dynamic_ker_offset + 0.001)

        # B. Delayed Training (Label Resolution via Adaptive Barrier)
        resolved_labels = self.labeler.resolve_labels(high, low, current_close=price)
        
        if resolved_labels:
            for (stored_feats, outcome_label, realized_ret) in resolved_labels:
                # --- PROFIT WEIGHTED LEARNING ---
                w_pos = float(self.params.get('positive_class_weight', 1.5))
                w_neg = float(self.params.get('negative_class_weight', 1.0))
                
                base_weight = w_pos if outcome_label != 0 else w_neg
                
                # Scale by Profit Magnitude (Log Scale)
                ret_scalar = math.log1p(abs(realized_ret) * 100.0)
                ret_scalar = max(0.5, ret_scalar)
                
                # --- EFFICIENCY-WEIGHTED LEARNING ---
                # Weight samples by KER. High efficiency bars are more valuable.
                hist_ker = stored_feats.get('ker', 0.5)
                ker_weight = hist_ker * 2.0 
                
                final_weight = base_weight * ret_scalar * ker_weight
                
                # Train the model
                self.model.learn_one(stored_feats, outcome_label, sample_weight=final_weight)
                
                # Double Learn for Positive outcomes (Reinforcement)
                if outcome_label != 0:
                      self.model.learn_one(stored_feats, outcome_label, sample_weight=final_weight * 1.5)

                # Train Meta Labeler
                if outcome_label != 0:
                    self.meta_labeler.update(stored_feats, primary_action=outcome_label, outcome_pnl=realized_ret)

        # C. Add CURRENT Bar as new Trade Opportunity
        parkinson = features.get('parkinson_vol', 0.0)
        features['parkinson_vol'] = parkinson 
        
        current_atr = features.get('atr', 0.001)
        self.labeler.add_trade_opportunity(features, price, current_atr, timestamp, parkinson_vol=parkinson)

        # ============================================================
        # D. PROJECT PHOENIX: ACTIVE TRADE MANAGEMENT
        # ============================================================
        
        # Enforce "1 Trade Per Symbol" rule (Set and Forget)
        if self.symbol in broker.positions:
            return 

        # ============================================================
        # E. ENTRY LOGIC GATES (V7.0 AGGRESSOR ONLY)
        # ============================================================
        
        # 1. Extract Phoenix Indicators
        rvol = features.get('rvol', 1.0)
        aggressor = features.get('aggressor', 0.5)
        adx_val = features.get('adx', 0.0)
        
        # --- CRITICAL FILTER 1: FUEL GAUGE (HARD GATE) ---
        # RVOL must be > Threshold to inject momentum
        if rvol < self.vol_gate_ratio:
            self.rejection_stats[f"Low Fuel (RVol {rvol:.2f} < {self.vol_gate_ratio})"] += 1
            return

        # --- CRITICAL FILTER 2: VOLUME EXHAUSTION ---
        if rvol > self.max_rvol_thresh:
            self.rejection_stats[f"Volume Climax (RVol {rvol:.2f} > {self.max_rvol_thresh})"] += 1
            return 
        
        # --- CRITICAL FILTER 3: EFFICIENCY GATE (HARD GATE) ---
        # Apply Dynamic Offset from ADWIN but keep hard floor
        effective_ker_thresh = max(0.15, self.ker_thresh + self.dynamic_ker_offset)
        
        if self.consecutive_losses > 0:
            # Soft Breaker: Require higher efficiency if losing
            effective_ker_thresh += min(0.1, self.consecutive_losses * 0.02)
            
        if ker_val < effective_ker_thresh:
            self.rejection_stats[f"Low Efficiency (KER {ker_val:.2f} < {effective_ker_thresh:.2f})"] += 1
            return

        # --- CRITICAL FILTER 4: TREND STRENGTH ---
        if adx_val < self.adx_threshold:
            self.rejection_stats[f"Weak Trend (ADX {adx_val:.1f} < {self.adx_threshold})"] += 1
            return

        # --- CRITICAL FILTER 5: MTF CONTEXT ---
        d1_ema = context_data.get('d1', {}).get('ema200', 0.0)
        d1_trend_up = (price > d1_ema) if d1_ema > 0 else True
        d1_trend_down = (price < d1_ema) if d1_ema > 0 else True
        
        h4_rsi = context_data.get('h4', {}).get('rsi', 50.0)
        h4_bull = h4_rsi > 50
        h4_bear = h4_rsi < 50
        
        proposed_action = 0 # 0=HOLD, 1=BUY, -1=SELL
        regime_label = "Trend"

        # --- AGGRESSOR TRIGGER (ORDER FLOW IMBALANCE) ---
        # Logic: Confirm Buy Vol > Sell Vol * 1.2
        
        # Prevent division by zero
        safe_sell_vol = sell_vol if sell_vol > 0 else 1.0
        safe_buy_vol = buy_vol if buy_vol > 0 else 1.0
        
        flow_ratio_bull = buy_vol / safe_sell_vol
        flow_ratio_bear = sell_vol / safe_buy_vol
        
        is_bullish_flow = flow_ratio_bull > self.aggressor_ratio_min
        is_bearish_flow = flow_ratio_bear > self.aggressor_ratio_min
        
        # Secondary Price Action Check
        is_bullish_pa = aggressor > self.aggressor_thresh_price
        is_bearish_pa = aggressor < (1.0 - self.aggressor_thresh_price)

        # >>> ENTRY LOGIC <<<
        
        # BUY SCENARIO
        if is_bullish_flow and is_bullish_pa and h4_bull:
            if self.require_d1_trend and not d1_trend_up:
                self.rejection_stats["Counter-D1 Trend"] += 1
                return
            proposed_action = 1
            regime_label = "Aggressor-Long"

        # SELL SCENARIO
        elif is_bearish_flow and is_bearish_pa and h4_bear:
            if self.require_d1_trend and not d1_trend_down:
                self.rejection_stats["Counter-D1 Trend"] += 1
                return
            proposed_action = -1
            regime_label = "Aggressor-Short"
            
        else:
            self.rejection_stats["No Trigger"] += 1
            return

        # ============================================================
        # F. ML CONFIRMATION & EXECUTION
        # ============================================================
        
        try:
            # Primary Prediction
            pred_proba = self.model.predict_proba_one(features)
            
            prob_buy = pred_proba.get(1, 0.0)
            prob_sell = pred_proba.get(-1, 0.0)
            
            confidence = prob_buy if proposed_action == 1 else prob_sell
            
            # --- META LABELING ---
            is_profitable = self.meta_labeler.predict(
                features, 
                proposed_action, 
                threshold=float(self.params.get('meta_labeling_threshold', 0.50))
            )
            
            if proposed_action != 0:
                self.meta_label_events += 1

            # --- EXECUTION WITH DYNAMIC CONFIDENCE ---
            min_prob = float(self.params.get('min_calibrated_probability', 0.55))
            
            if self.consecutive_losses > 0:
                # Streak Breaker: Increase required confidence if losing
                min_prob += min(0.1, self.consecutive_losses * 0.02)
                
            if confidence < min_prob:
                self.rejection_stats[f"Low Confidence ({confidence:.2f} < {min_prob:.2f})"] += 1
                return

            # --- SNIPER PROTOCOL: FINAL FILTER GATE ---
            # Validate Trend (SMA200) and Extension (RSI) before executing
            if not self._check_sniper_filters(proposed_action, price):
                return
            # ------------------------------------------

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
        Overrides Config defaults with Optuna parameters to ensure Dynamic R:R.
        """
        action = "BUY" if action_int == 1 else "SELL"
        
        # Dynamic Correlation Check
        exposure_count = 0
        quote_currency = self.symbol[-3:] 
        for pos in broker.open_positions:
            if quote_currency in pos.symbol: 
                exposure_count += 1
        
        # Construct Context
        ctx = TradeContext(
            symbol=self.symbol,
            price=price,
            stop_loss_price=0.0,
            account_equity=broker.equity,
            account_currency="USD",
            win_rate=0.45, 
            risk_reward_ratio=self.optimized_reward_mult 
        )

        # Calculate Size using RiskManager 
        current_atr = features.get('atr', 0.001)
        current_ker = features.get('ker', 1.0)
        volatility = features.get('volatility', 0.001)

        # Retrieve Optimized Risk Parameter
        risk_override = self.params.get('risk_per_trade_percent')

        trade_intent, risk_usd = RiskManager.calculate_rck_size(
            context=ctx,
            conf=confidence,
            volatility=volatility,
            active_correlations=exposure_count,
            market_prices=self.last_price_map,
            atr=current_atr, 
            ker=current_ker,
            account_size=broker.equity,
            risk_percent_override=risk_override
        )

        if trade_intent.volume <= 0:
            self.rejection_stats[f"Risk Zero: {trade_intent.comment}"] += 1
            return

        # --- STOP LOSS ENFORCEMENT (DYNAMIC CONFIG) ---
        # AUDIT FIX: Use the configured SL multiplier, not hardcoded 1.5
        stop_dist = current_atr * self.sl_atr_mult
        trade_intent.stop_loss = stop_dist
        
        # TP based on Optimized R:R (e.g., 3.0 * ATR)
        dynamic_tp_dist = current_atr * self.optimized_reward_mult
        trade_intent.take_profit = dynamic_tp_dist
        
        # 2. Assign Action
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
        else: # SELL
            sl_price = price + stop_dist
            tp_price = price - tp_dist
            side = -1

        # Submit Order
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
                'optimized_rr': self.optimized_reward_mult
            }
        )
        
        broker.submit_order(order)
        
        # Track Features for Explainability
        imp_feats = []
        imp_feats.append(regime)
        if features.get('rvol', 0) > 2.0: imp_feats.append('High_Fuel')
        if features.get('aggressor', 0) > 0.6: imp_feats.append('High_Aggressor')
        if features.get('mtf_alignment', 0) == 1.0: imp_feats.append('MTF_Aligned')
        
        for f in imp_feats:
            self.feature_importance_counter[f] += 1

        self.trade_events.append({
            'time': dt_timestamp.timestamp(), 
            'action': action,
            'price': price,
            'conf': confidence,
            'rvol': features.get('rvol', 0),
            'parkinson': features.get('parkinson_vol', 0),
            'aggressor': features.get('aggressor', 0.5),
            'ker': current_ker,
            'atr': current_atr,
            'regime': regime,
            'top_feats': imp_feats
        })

    def _check_daily_loss_limit(self, broker: BacktestBroker) -> bool:
        """
        Simulates SessionGuard in Backtest.
        If Daily Drawdown > 4.9%, returns True (BLOCK TRADES).
        """
        try:
            current_dd_pct = (broker.initial_balance - broker.equity) / broker.initial_balance
            if current_dd_pct > 0.049: 
                return True
            return False
        except:
            return False

    def _update_streak_status(self, broker: BacktestBroker):
        """
        Updates the consecutive loss counter based on the last closed trade.
        """
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
        Approximates D1 and H4 context from the M5 stream for Backtesting.
        """
        # H4 Approximation (Every 4 hours)
        h4_idx = (dt.day * 6) + (dt.hour // 4)
        if h4_idx != self.last_h4_idx:
            self.h4_buffer.append(price)
            self.last_h4_idx = h4_idx
            
        # D1 Approximation (Every day)
        d1_idx = dt.toordinal()
        if d1_idx != self.last_d1_idx:
            self.d1_buffer.append(price)
            self.last_d1_idx = d1_idx
            
        # Calculate Context
        ctx = {'d1': {}, 'h4': {}}
        
        # D1 EMA 200
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
            
        # H4 RSI 14
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
        SNIPER PROTOCOL:
        1. Trend Filter: Trade ONLY if price is above/below SMA 200.
        2. Extension Filter: Don't Buy if RSI > 70, Don't Sell if RSI < 30.
        """
        if len(self.closes) < 200:
            return True # Not enough data, default to allow (or False to be safe)

        # 1. Calculate SMA 200
        sma_200 = sum(self.closes) / len(self.closes)
        
        # 2. Calculate Simple RSI (Approximate)
        gains = [x for x in self.rsi_buffer if x > 0]
        losses = [abs(x) for x in self.rsi_buffer if x < 0]
        avg_gain = sum(gains) / 14 if gains else 0
        avg_loss = sum(losses) / 14 if losses else 1e-9
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # 3. Apply Filters
        if signal == 1: # BUY Signal
            # Trend Check: Price MUST be > SMA 200 (Uptrend)
            if price < sma_200:
                self.rejection_stats['Counter_Trend_SMA200'] += 1
                return False
            # Extension Check: Price MUST NOT be Overbought
            if rsi > 70:
                self.rejection_stats['Overbought_RSI'] += 1
                return False
                
        elif signal == -1: # SELL Signal
            # Trend Check: Price MUST be < SMA 200 (Downtrend)
            if price > sma_200:
                self.rejection_stats['Counter_Trend_SMA200'] += 1
                return False
            # Extension Check: Price MUST NOT be Oversold
            if rsi < 30:
                self.rejection_stats['Oversold_RSI'] += 1
                return False
                
        return True

    def _inject_auxiliary_data(self):
        """Injects static approximations ONLY if missing."""
        defaults = {
            "USDJPY": 150.0, "GBPUSD": 1.25, "EURUSD": 1.08,
            "USDCAD": 1.35, "USDCHF": 0.90, "AUDUSD": 0.65, "NZDUSD": 0.60,
            "GBPJPY": 190.0, "EURJPY": 160.0, "AUDJPY": 95.0
        }
        for sym, price in defaults.items():
            if sym not in self.last_price_map:
                self.last_price_map[sym] = price

    def generate_autopsy(self) -> str:
        """
        Generates a text report explaining WHY the strategy behaved this way.
        """
        if not self.trade_events:
            sorted_rejects = sorted(self.rejection_stats.items(), key=lambda item: item[1], reverse=True)
            reject_str = ", ".join([f"{k}: {v}" for k, v in sorted_rejects[:5]])
            
            status = "Waiting for Warm-Up" if not self.burn_in_complete else "No Trigger Conditions Met"
            return f"AUTOPSY: No trades. Status: {status}. Top Rejections: {{{reject_str}}}. Bars processed: {self.bars_processed}"
        
        avg_conf = np.mean([t['conf'] for t in self.trade_events])
        avg_rvol = np.mean([t['rvol'] for t in self.trade_events]) 
        avg_park = np.mean([t['parkinson'] for t in self.trade_events]) 
        
        sorted_rejects = sorted(self.rejection_stats.items(), key=lambda item: item[1], reverse=True)
        reject_str = ", ".join([f"{k}: {v}" for k, v in sorted_rejects[:5]])
        
        top_features = self.feature_importance_counter.most_common(5)
        feat_str = str(top_features)
        
        report = (
            f"\n --- 💀 PHOENIX AUTOPSY ({self.symbol}) ---\n"
            f" Trades: {len(self.trade_events)}\n"
            f" Avg Conf: {avg_conf:.2f}\n"
            f" Avg RVol: {avg_rvol:.2f} | Avg Parkinson: {avg_park:.5f}\n"
            f" Top Drivers: {feat_str}\n"
            f" Rejections: {{{reject_str}}}\n"
            f" ----------------------------------------\n"
        )
        return report