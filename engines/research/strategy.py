# =============================================================================
# FILENAME: engines/research/strategy.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/strategy.py
# DEPENDENCIES: shared, river, engines.research.backtester
# DESCRIPTION: The Adaptive Strategy Kernel (Backtesting Version).
# 
# PHOENIX STRATEGY V6.0 (AGGRESSIVE PROFIT RECOVERY):
# 1. REGIME C UNLOCKED: Implemented Mean Reversion for ranging markets.
#    - Condition: ADX < 25 + Bollinger Band Touch + RSI Extremes.
#    - Goal: Monetize the 70% of time the market is chopping.
# 2. FILTER RELAXATION:
#    - D1 Trend: Downgraded from Hard Lock to Directional Bias.
#    - Volume Gate: Lowered to 0.8x average to catch quiet moves.
# 3. PROFIT FOCUS: Prioritizes entry frequency over perfection.
# =============================================================================
import logging
import sys
import numpy as np
import random
import math
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
        
        # 2. Adaptive Triple Barrier Labeler (The Teacher)
        tbm_conf = params.get('tbm', {})
        
        # Sync Labeler Risk with Config (Default 1.5 ATR for labeling)
        risk_mult_conf = CONFIG['risk_management'].get('stop_loss_atr_mult', 1.5)
        
        # Target Reward should align with Strategy (Use Optimization Param)
        # We ensure the Labeler trains on the SAME target we plan to execute.
        self.optimized_reward_mult = tbm_conf.get('barrier_width', 3.0)
        
        self.labeler = AdaptiveTripleBarrier(
            horizon_ticks=tbm_conf.get('horizon_minutes', 120), 
            risk_mult=risk_mult_conf, 
            reward_mult=self.optimized_reward_mult,
            drift_threshold=tbm_conf.get('drift_threshold', 1.5)
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
        
        # --- FORENSIC RECORDER ---
        self.decision_log = deque(maxlen=1000)
        self.trade_events = []
        self.rejection_stats = defaultdict(int) 
        self.feature_importance_counter = Counter() 
        
        # --- PHOENIX STRATEGY PARAMETERS (V6.0 AGGRESSIVE) ---
        phx_conf = CONFIG.get('phoenix_strategy', {})
        
        # Gates
        self.enable_regime_a = phx_conf.get('enable_regime_a_entries', True)
        self.enable_regime_c = True # Enable Mean Reversion (Regime C)
        self.require_d1_trend = phx_conf.get('require_d1_trend', True)
        self.require_h4_alignment = phx_conf.get('require_h4_alignment', True)
        
        # Safety Cap - Relaxed to 5.0 to allow volatility
        self.max_rvol_thresh = phx_conf.get('max_relative_volume', 5.0)
        
        # Thresholds (AGGRESSIVELY LOWERED FOR PROFITABILITY)
        self.ker_thresh = 0.15 # Was 0.20
        self.adx_threshold = 15.0 # Was 18.0 - Catch trends earlier
        
        # Volume Gate - Relaxed
        self.vol_gate_ratio = 0.8 # Was 1.1 - Allow trades on standard volume
        
        # Momentum
        self.aggressor_thresh = 0.52 # Was 0.55 - Lower bar for entry
        
        self.limit_order_offset_pips = CONFIG.get('trading', {}).get('limit_order_offset_pips', 0.2)
        
        # Friday Liquidation
        self.friday_entry_cutoff = CONFIG.get('risk_management', {}).get('friday_entry_cutoff_hour', 16)
        self.friday_close_hour = CONFIG.get('risk_management', {}).get('friday_liquidation_hour_server', 21)
        
        # --- REC 1: Dynamic Gate Scaling State ---
        self.ker_drift_detector = drift.ADWIN(delta=0.01)
        self.dynamic_ker_offset = 0.0
        
        # --- TRAILING STOP & BREAK EVEN LOGIC (DISABLED IN V5.0) ---
        self.use_trailing_stop = False
        self.use_breakeven = False

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

        # Inject Aux Data for single-symbol tests
        self._inject_auxiliary_data()

        # Robust Timestamp Extraction
        try:
            if hasattr(snapshot.timestamp, 'timestamp'):
                timestamp = snapshot.timestamp.timestamp()
                dt_ts = snapshot.timestamp
            else:
                timestamp = float(snapshot.timestamp)
                dt_ts = datetime.fromtimestamp(timestamp)
        except Exception:
            timestamp = 0.0
            dt_ts = datetime.now()

        # --- SIMULATED SESSION GUARD (Daily Loss Limit) ---
        if self._check_daily_loss_limit(broker):
            self.rejection_stats["Daily Loss Limit Hit"] += 1
            return

        # --- UPDATE STREAK STATUS (CRITICAL FOR FILTER) ---
        self._update_streak_status(broker)

        # --- FRIDAY LOGIC: LIQUIDATION VS ENTRY GUARD ---
        # 1. Liquidation (21:00)
        if dt_ts.weekday() == 4 and dt_ts.hour >= self.friday_close_hour:
            if self.symbol in broker.positions:
                pos = broker.positions[self.symbol]
                broker._close_partial_position(pos, pos.quantity, price, dt_ts, "Friday Liquidation")
                if self.debug_mode: logger.info(f"🚫 {self.symbol} Liquidated for Weekend (Friday {dt_ts.hour}:00)")
            return 

        # 2. Entry Guard (16:00) - Aggressive Filter
        is_friday_afternoon = (dt_ts.weekday() == 4 and dt_ts.hour >= self.friday_entry_cutoff)
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
        context_data = self._simulate_mtf_context(price, dt_ts)

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
            self.dynamic_ker_offset = max(-0.15, self.dynamic_ker_offset - 0.05)
        else:
            # Slowly decay offset back to 0 (Normalization)
            self.dynamic_ker_offset = min(0.0, self.dynamic_ker_offset + 0.001)

        # B. Delayed Training (Label Resolution via Adaptive Barrier)
        resolved_labels = self.labeler.resolve_labels(high, low, current_close=price)
        
        if resolved_labels:
            for (stored_feats, outcome_label, realized_ret) in resolved_labels:
                # --- PROFIT WEIGHTED LEARNING ---
                w_pos = self.params.get('positive_class_weight', 1.5)
                w_neg = self.params.get('negative_class_weight', 1.0)
                
                base_weight = w_pos if outcome_label != 0 else w_neg
                
                # Scale by Profit Magnitude (Log Scale)
                ret_scalar = math.log1p(abs(realized_ret) * 100.0)
                ret_scalar = max(0.5, ret_scalar)
                
                # --- REC 3: EFFICIENCY-WEIGHTED LEARNING ---
                # Weight samples by KER. High efficiency bars are more valuable.
                hist_ker = stored_feats.get('ker', 0.5)
                # Boost multiplier from 1.0+ker to ker*2.0
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
        # --- REC 2: Volatility Passing ---
        # Inject Parkinson for Labeler boost (Rec 2)
        parkinson = features.get('parkinson_vol', 0.0)
        features['parkinson_vol'] = parkinson 
        
        current_atr = features.get('atr', 0.0)
        self.labeler.add_trade_opportunity(features, price, current_atr, timestamp, parkinson_vol=parkinson)

        # ============================================================
        # D. PROJECT PHOENIX: ACTIVE TRADE MANAGEMENT
        # ============================================================
        
        # Enforce "1 Trade Per Symbol" rule (Set and Forget)
        if self.symbol in broker.positions:
            return 

        # --- ENTRY BLOCK: FRIDAY AFTERNOON (AUDIT FIX) ---
        if is_friday_afternoon:
            self.rejection_stats["Friday Entry Guard"] += 1
            return 
        # -------------------------------------------------

        # ============================================================
        # E. ENTRY LOGIC GATES (V6.0 OVERHAUL)
        # ============================================================
        
        # 1. Extract Phoenix Indicators
        rvol = features.get('rvol', 1.0)
        aggressor = features.get('aggressor', 0.5)
        atr_val = features.get('atr', 0.0001)
        mtf_align = features.get('mtf_alignment', 0.0)
        adx_val = features.get('adx', 0.0)
        
        # Bollinger Bands for Regime C
        bb_upper = features.get('bb_upper', price * 1.01)
        bb_lower = features.get('bb_lower', price * 0.99)
        bb_width = features.get('bb_width', 0.0)
        
        # RSI for Regime C
        rsi_val = features.get('rsi_norm', 0.5) * 100.0
        
        # --- CRITICAL FILTER 1: VOLUME EXHAUSTION FILTER ---
        if rvol > self.max_rvol_thresh:
            self.rejection_stats[f"Volume Climax (RVol {rvol:.2f} > {self.max_rvol_thresh})"] += 1
            return # Block trade
        
        # --- CORRECTED STREAK BREAKER (V6.0) & DYNAMIC SCALING ---
        # Apply Dynamic Offset from ADWIN (Rec 1)
        effective_ker_thresh = self.ker_thresh + self.dynamic_ker_offset
        
        if self.consecutive_losses > 0:
            if self.consecutive_losses >= 3:
                # FIRM BREAKER: Softened to +0.05 to avoid dormancy
                effective_ker_thresh += 0.05 
            else:
                # SOFT BREAKER: Mild penalty for 1-2 losses
                effective_ker_thresh += min(0.05, self.consecutive_losses * 0.01)
        
        # Ensure gate doesn't go below absolute safety minimum
        effective_ker_thresh = max(0.05, effective_ker_thresh)
            
        # --- CRITICAL FILTER 2: MTF TREND CONTEXT ---
        d1_ema = context_data.get('d1', {}).get('ema200', 0.0)
        d1_trend_up = (price > d1_ema) if d1_ema > 0 else True
        d1_trend_down = (price < d1_ema) if d1_ema > 0 else True
        
        h4_rsi = context_data.get('h4', {}).get('rsi', 50.0)
        h4_bull = h4_rsi > 50
        h4_bear = h4_rsi < 50
        
        # Gate Definitions
        vol_gate = rvol > self.vol_gate_ratio
        
        # Momentum Direction (Aggressor Ratio)
        is_bullish_candle = aggressor > self.aggressor_thresh
        is_bearish_candle = aggressor < (1.0 - self.aggressor_thresh)
        
        proposed_action = 0 # 0=HOLD, 1=BUY, -1=SELL
        regime_label = "C (Noise)"
        
        # --- REGIME DETERMINATION ---
        is_trending = adx_val > self.adx_threshold
        
        # >>> LOGIC BRANCH 1: TREND FOLLOWING (REGIME A/B) <<<
        if is_trending and vol_gate:
            
            # REQUIRE H4 ALIGNMENT (Stronger than D1 for Intraday)
            if is_bullish_candle and h4_bull:
                proposed_action = 1
                regime_label = "Trend-Long"
                
                # Downgraded D1 Check: Only block if D1 is aggressively against us
                # Now handled as a soft bias rather than a hard lock
                if not d1_trend_up:
                    regime_label += " (Counter-D1)"
                    
            elif is_bearish_candle and h4_bear:
                proposed_action = -1
                regime_label = "Trend-Short"
                if not d1_trend_down:
                    regime_label += " (Counter-D1)"
            
            else:
                self.rejection_stats["Trend: H4 Mismatch"] += 1

        # >>> LOGIC BRANCH 2: MEAN REVERSION (REGIME C) <<<
        # "Unlock the Chop" - Monetize ranging markets
        elif self.enable_regime_c and not is_trending:
            # Conditions:
            # 1. Price is interacting with Bands
            # 2. RSI confirms overbought/oversold
            # 3. Not in a squeeze (BB Width decent)
            
            if bb_width > 0.001: # Ensure bands aren't pinched tight
                # Reversion to Mean (Long)
                if price <= bb_lower and rsi_val < 35:
                    proposed_action = 1
                    regime_label = "MeanRev-Long"
                
                # Reversion to Mean (Short)
                elif price >= bb_upper and rsi_val > 65:
                    proposed_action = -1
                    regime_label = "MeanRev-Short"
                else:
                    self.rejection_stats["MeanRev: No Trigger"] += 1
            else:
                self.rejection_stats["MeanRev: Squeeze"] += 1
        
        else:
            self.rejection_stats["No Regime"] += 1

        # --- FINAL GATE CHECK ---
        if proposed_action == 0:
            return 

        # Ker Check (Only applicable for Trend Trades, skipped for MeanRev)
        if "Trend" in regime_label and ker_val < effective_ker_thresh:
            self.rejection_stats[f"Low Efficiency (KER {ker_val:.2f} < {effective_ker_thresh:.2f})"] += 1
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
                threshold=self.params.get('meta_labeling_threshold', 0.50) # Relaxed to 0.50
            )
            
            if proposed_action != 0:
                self.meta_label_events += 1

            # --- EXECUTION WITH DYNAMIC CONFIDENCE (CORRECTED BREAKER) ---
            min_prob = self.params.get('min_calibrated_probability', 0.55) # Lowered base to 0.55
            
            # Penalize Counter-Trend / MeanRev slightly to ensure quality
            if "MeanRev" in regime_label or "Counter" in regime_label:
                min_prob += 0.05
            
            if self.consecutive_losses > 0:
                if self.consecutive_losses >= 3:
                    # FIRM BREAKER: +5% Confidence
                    min_prob += 0.05
                else:
                    # SOFT BREAKER: +1% per loss
                    min_prob += min(0.05, self.consecutive_losses * 0.01)
                
            if confidence < min_prob:
                self.rejection_stats[f"Low Confidence ({confidence:.2f} < {min_prob:.2f})"] += 1
                return

            if is_profitable:
                # CRITICAL FIX: Pass the datetime object (dt_ts) to _execute_entry
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
        
        # --- AUDIT FIX: DYNAMIC CORRELATION CHECK ---
        exposure_count = 0
        quote_currency = self.symbol[-3:] 
        for pos in broker.open_positions:
            if quote_currency in pos.symbol: 
                exposure_count += 1
        
        # Construct Context
        # Note: Risk Reward passed here is informational, sizing is handled by RiskManager
        ctx = TradeContext(
            symbol=self.symbol,
            price=price,
            stop_loss_price=0.0,
            account_equity=broker.equity,
            account_currency="USD",
            win_rate=0.45, 
            risk_reward_ratio=self.optimized_reward_mult # Use Optimized R:R
        )

        # Calculate Size using RiskManager (Calculates default TP/SL from Config)
        current_atr = features.get('atr', 0.001)
        current_ker = features.get('ker', 1.0)
        volatility = features.get('volatility', 0.001)

        # --- RETRIEVE OPTIMIZED RISK PARAMETER (AUDIT FIX) ---
        # Check self.params for 'risk_per_trade_percent' which Optuna injects
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
            risk_percent_override=risk_override # NEW: Pass Override to Risk Manager
        )

        if trade_intent.volume <= 0:
            self.rejection_stats[f"Risk Zero: {trade_intent.comment}"] += 1
            return

        # --- OPTIMIZATION OVERRIDE (CRITICAL FOR R:R SEARCH) ---
        # RiskManager returns Config-based TP. We must override it with
        # the 'barrier_width' (Reward Multiplier) found by Optuna.
        # This ensures the Strategy executes exactly what the ML model learned.
        
        # 1. Recalculate Dynamic TP
        dynamic_tp_dist = current_atr * self.optimized_reward_mult
        trade_intent.take_profit = dynamic_tp_dist
        
        # 2. Assign Action
        trade_intent.action = action

        qty = trade_intent.volume
        stop_dist = trade_intent.stop_loss # Keep RiskManager's SL (Fixed Risk)
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
        if features.get('rvol', 0) > 1.2: imp_feats.append('High_Volume')
        if features.get('parkinson_vol', 0) > 0.002: imp_feats.append('High_Parkinson')
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
        If Daily Drawdown > 4.9% (Relaxed from 4.8%), returns True (BLOCK TRADES).
        """
        try:
            current_dd_pct = (broker.initial_balance - broker.equity) / broker.initial_balance
            if current_dd_pct > 0.049: # V5.0: Relaxed to 4.9%
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
            
        # Re-calc streak from history tail (stateless safe)
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