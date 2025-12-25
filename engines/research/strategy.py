# =============================================================================
# FILENAME: engines/research/strategy.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/strategy.py
# DEPENDENCIES: shared, river, engines.research.backtester
# DESCRIPTION: The Adaptive Strategy Kernel (Backtesting Version).
#
# PHOENIX STRATEGY UPGRADE (2025-12-25 - UNBLOCKED & RELAXED):
# 1. LOGIC OVERHAUL: Relaxed entry triggers to fix 100% rejection rate.
# 2. TREND REGIME: Now enters on Momentum (>0.65 BB Position) not just Breakouts.
# 3. MEAN REV REGIME: RSI thresholds relaxed to 60/40 to unblock "Neutral" zones.
# 4. PARITY: Maintained structure for live engine synchronization.
# =============================================================================
import logging
import sys
import numpy as np
import random
import math
from collections import deque, defaultdict, Counter
from typing import Any, Dict, Optional, List
from datetime import datetime

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
        self.labeler = AdaptiveTripleBarrier(
            horizon_ticks=tbm_conf.get('horizon_minutes', 30),
            risk_mult=CONFIG['risk_management']['stop_loss_atr_mult'],
            reward_mult=tbm_conf.get('barrier_width', 2.0),
            drift_threshold=tbm_conf.get('drift_threshold', 1.0)
        )
        
        # 3. Meta Labeler (The Gatekeeper)
        self.meta_labeler = MetaLabeler()
        self.meta_label_events = 0 
        
        # 4. Probability Calibrators
        self.calibrator_buy = ProbabilityCalibrator(window=2000)
        self.calibrator_sell = ProbabilityCalibrator(window=2000)
        
        # 5. Warm-up State
        self.burn_in_limit = params.get('burn_in_periods', 1000)
        self.burn_in_counter = 0
        self.burn_in_complete = False
        
        # State
        self.last_features = None
        self.last_price = 0.0
        self.last_price_map = {}
        self.bars_processed = 0
        
        # MTF Simulation State (for backtesting consistency)
        self.h4_buffer = deque(maxlen=200) # Store H4 closes for RSI
        self.d1_buffer = deque(maxlen=200) # Store D1 closes for EMA
        self.last_h4_idx = -1
        self.last_d1_idx = -1
        
        # --- FORENSIC RECORDER ---
        self.decision_log = deque(maxlen=1000)
        self.trade_events = []
        self.rejection_stats = defaultdict(int) 
        self.feature_importance_counter = Counter() # Tracks top features
        
        # --- STRATEGY PARAMETERS (From Config) ---
        feat_conf = CONFIG.get('features', {})
        self.bb_dev = feat_conf.get('bollinger_bands', {}).get('std_dev', 2.0)
        
        # Thresholds (Regime Gates)
        self.ker_trend = CONFIG.get('trading', {}).get('ker_threshold_trend', 0.6)
        self.ker_mean_rev = CONFIG.get('trading', {}).get('ker_threshold_mean_rev', 0.3)
        self.limit_offset = CONFIG.get('trading', {}).get('limit_order_offset_pips', 0.5)

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

        # Flow Volumes (Critical for OFI)
        buy_vol = snapshot.get_price(self.symbol, 'buy_vol')
        sell_vol = snapshot.get_price(self.symbol, 'sell_vol')
        
        # --- RETAIL FALLBACK LOGIC ---
        if buy_vol == 0 and sell_vol == 0:
            if self.last_price > 0:
                if price > self.last_price:
                    buy_vol = volume
                    sell_vol = 0.0
                elif price < self.last_price:
                    buy_vol = 0.0
                    sell_vol = volume
                else:
                    buy_vol = volume / 2.0
                    sell_vol = volume / 2.0
        
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

        # B. Delayed Training (Label Resolution via Adaptive Barrier)
        resolved_labels = self.labeler.resolve_labels(high, low, current_close=price)
        
        if resolved_labels:
            for (stored_feats, outcome_label, realized_ret) in resolved_labels:
                # --- PROFIT WEIGHTED LEARNING ---
                w_pos = self.params.get('positive_class_weight', 2.0)
                w_neg = self.params.get('negative_class_weight', 2.0)
                
                base_weight = w_pos if outcome_label != 0 else w_neg
                
                # Scale by Profit Magnitude (Log Scale)
                ret_scalar = math.log1p(abs(realized_ret) * 100.0)
                ret_scalar = max(0.5, ret_scalar)
                
                final_weight = base_weight * ret_scalar
                
                # Train the model
                self.model.learn_one(stored_feats, outcome_label, sample_weight=final_weight)
                
                # Double Learn for Positive outcomes (Reinforcement)
                if outcome_label != 0:
                     self.model.learn_one(stored_feats, outcome_label, sample_weight=final_weight)

                # Train Meta Labeler
                if outcome_label != 0:
                    self.meta_labeler.update(stored_feats, primary_action=outcome_label, outcome_pnl=realized_ret)

        # C. Add CURRENT Bar as new Trade Opportunity
        current_atr = features.get('atr', 0.0)
        self.labeler.add_trade_opportunity(features, price, current_atr, timestamp)

        # ============================================================
        # D. STRATEGY LOGIC: REGIME SWITCHING (RELAXED & UNBLOCKED)
        # ============================================================
        
        # Extract Core Indicators
        ker_val = features.get('ker', 0.5)
        mtf_align = features.get('mtf_alignment', 0.0) # 1.0 if aligned
        bb_upper = features.get('bb_upper', 999999.0)
        bb_lower = features.get('bb_lower', 0.0)
        rsi_val = features.get('rsi', 50.0)
        
        # New: BB Position (0=Lower, 0.5=Mid, 1=Upper)
        bb_pos = features.get('bb_position', 0.5)
        
        proposed_action = 0 # 0=HOLD, 1=BUY, -1=SELL
        regime_label = "C (Noise)"
        
        # --- REGIME A: EFFICIENT TREND ---
        # UNBLOCK: Trigger on MOMENTUM (High in Band) rather than pure Breakout
        if (ker_val > self.ker_trend and mtf_align == 1.0) or (ker_val > 0.75):
            regime_label = "A (Trend)"
            
            # TRIGGER: Strong Position within Bands (e.g., > 65%)
            if bb_pos > 0.65:
                proposed_action = 1 # BUY
            elif bb_pos < 0.35:
                proposed_action = -1 # SELL
            else:
                self.rejection_stats["Regime A: Weak Momentum"] += 1
                
        # --- REGIME B: MEAN REVERSION (CHOPPY) ---
        elif ker_val < self.ker_mean_rev:
            regime_label = "B (MeanRev)"
            
            # Trigger 1: Band Reversal (Classic)
            if price > bb_upper:
                proposed_action = -1 # SELL
            elif price < bb_lower:
                proposed_action = 1 # BUY
                
            # Trigger 2: RELAXED RSI (Unblocker)
            # If inside bands, use RSI Extremes to force activity
            elif rsi_val > 60: # AUDIT FIX: Relaxed from 70
                proposed_action = -1 # SELL (Overbought)
                regime_label = "B (RSI-Ext)"
            elif rsi_val < 40: # AUDIT FIX: Relaxed from 30
                proposed_action = 1 # BUY (Oversold)
                regime_label = "B (RSI-Ext)"
            else:
                self.rejection_stats["Regime B: Neutral Zone"] += 1
                
        # --- REGIME C: NOISE ---
        else:
            # 0.3 <= KER <= 0.6 or Trend but misaligned
            # Sniper Mode: Only take extremes
            if rsi_val > 75:
                proposed_action = -1
                regime_label = "C (Sniper-Short)"
            elif rsi_val < 25:
                proposed_action = 1
                regime_label = "C (Sniper-Long)"
            else:
                regime_label = "C (Noise)"
                self.rejection_stats[f"Regime C: KER {ker_val:.2f}"] += 1
                return # Explicit HOLD

        if proposed_action == 0:
            return

        # ============================================================
        # E. ML CONFIRMATION & EXECUTION
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
                threshold=self.params.get('meta_labeling_threshold', 0.60)
            )
            
            if proposed_action != 0:
                self.meta_label_events += 1

            # --- EXECUTION ---
            # Safety Check: If ML thinks probability is terrible (< 0.4), skip even if Rule triggers.
            if confidence < 0.40:
                self.rejection_stats[f"ML Disagreement (Conf {confidence:.2f})"] += 1
                return

            if is_profitable:
                self._execute_logic(confidence, price, features, broker, dt_ts, proposed_action, regime_label)
            else:
                self.rejection_stats['Meta-Labeler Reject'] += 1

        except Exception as e:
            if self.debug_mode: logger.error(f"Strategy Error: {e}")
            pass

    def _simulate_mtf_context(self, price: float, dt: datetime) -> Dict[str, Any]:
        """
        Approximates D1 and H4 context from the M5 stream for Backtesting.
        This allows 'mtf_alignment' feature to function without external data feeds.
        """
        # H4 Approximation (Every 48 M5 bars)
        # We define H4 bucket by hour // 4
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
            ema = np.mean(arr) # Fallback to mean for simplicity in simulation
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
            "USDCAD": 1.35, "USDCHF": 0.90, "AUDUSD": 0.65, "NZDUSD": 0.60
        }
        for sym, price in defaults.items():
            if sym not in self.last_price_map:
                self.last_price_map[sym] = price

    def _execute_logic(self, confidence, price, features, broker, timestamp: datetime, action_int: int, regime: str):
        """Decides whether to enter a trade using Volatility Targeting."""
        
        # 1. Signal Threshold
        min_prob = self.params.get('min_calibrated_probability', 0.85)
        
        if confidence < min_prob:
            self.rejection_stats[f'Low Confidence ({confidence:.2f} < {min_prob})'] += 1
            return

        action = "BUY" if action_int == 1 else "SELL"

        # 2. Position Sizing
        if broker.get_position(self.symbol): return
        
        volatility = features.get('volatility', 0.001)
        current_atr = features.get('atr', 0.001)
        current_ker = features.get('ker', 1.0) 

        ctx = TradeContext(
            symbol=self.symbol,
            price=price,
            stop_loss_price=0.0,
            account_equity=broker.equity,
            account_currency="USD",
            win_rate=0.55,
            risk_reward_ratio=2.0
        )

        trade_intent, risk_usd = RiskManager.calculate_rck_size(
            context=ctx,
            conf=confidence,
            volatility=volatility,
            active_correlations=0,
            market_prices=self.last_price_map,
            atr=current_atr, 
            ker=current_ker  
        )

        trade_intent.action = action

        if trade_intent.volume <= 0:
            self.rejection_stats[f"Risk Zero: {trade_intent.comment}"] += 1
            return

        qty = trade_intent.volume
        stop_dist = trade_intent.stop_loss
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
            timestamp_created=timestamp,
            stop_loss=sl_price,
            take_profit=tp_price,
            comment=f"{trade_intent.comment}|Regime:{regime}|MTF:{features.get('mtf_alignment',0):.0f}|Limit:{self.limit_offset}p"
        )
        
        broker.submit_order(order)
        
        # Track Features for Explainability
        imp_feats = []
        imp_feats.append(regime)
        if features.get('mtf_alignment', 0) == 1.0: imp_feats.append('MTF_Aligned')
        
        for f in imp_feats:
            self.feature_importance_counter[f] += 1

        self.trade_events.append({
            'time': timestamp,
            'action': action,
            'price': price,
            'conf': confidence,
            'vpin': features.get('vpin', 0),
            'entropy': features.get('entropy', 0),
            'ker': current_ker,
            'fdi': features.get('fdi', 0),
            'atr': current_atr,
            'volatility': volatility,
            'frac_diff': features.get('frac_diff', 0.0),
            'micro_ofi': features.get('micro_ofi', 0.0),
            'regime': regime,
            'top_feats': imp_feats
        })

    def generate_autopsy(self) -> str:
        """
        Generates a text report explaining WHY the strategy behaved this way.
        """
        if not self.trade_events:
            # Sort rejections by count desc
            sorted_rejects = sorted(self.rejection_stats.items(), key=lambda item: item[1], reverse=True)
            reject_str = ", ".join([f"{k}: {v}" for k, v in sorted_rejects[:5]]) # Top 5 reasons
            
            status = "Waiting for Warm-Up" if not self.burn_in_complete else "No Trigger Conditions Met"
            return f"AUTOPSY: No trades. Status: {status}. Top Rejections: {{{reject_str}}}. Bars processed: {self.bars_processed}"
        
        avg_conf = np.mean([t['conf'] for t in self.trade_events])
        avg_ker = np.mean([t['ker'] for t in self.trade_events]) 
        avg_ofi = np.mean([t['micro_ofi'] for t in self.trade_events]) 
        
        sorted_rejects = sorted(self.rejection_stats.items(), key=lambda item: item[1], reverse=True)
        reject_str = ", ".join([f"{k}: {v}" for k, v in sorted_rejects[:5]])
        
        # Explainability Report
        top_features = self.feature_importance_counter.most_common(5)
        feat_str = str(top_features)
        
        report = (
            f"\n --- 💀 STRATEGY AUTOPSY ({self.symbol}) ---\n"
            f" Trades: {len(self.trade_events)}\n"
            f" Avg Conf: {avg_conf:.2f}\n"
            f" Avg KER: {avg_ker:.2f} | Avg OFI: {avg_ofi:.2f}\n"
            f" Top Regimes: {feat_str}\n"
            f" Rejections: {{{reject_str}}}\n"
            f" ----------------------------------------\n"
        )
        return report