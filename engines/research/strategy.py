# =============================================================================
# FILENAME: engines/research/strategy.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/strategy.py
# DEPENDENCIES: shared, river, engines.research.backtester
# DESCRIPTION: The Adaptive Strategy Kernel.
#
# AUDIT REMEDIATION (PROFIT WEIGHTED LEARNING):
# 1. ADDED: Realized Return capture from Labeler.
# 2. ADDED: Log-weighted sample learning (Big Wins = Big Weights).
# 3. FIXED: Epsilon-Greedy logic.
# =============================================================================
import logging
import sys
import numpy as np
import random
import math
from collections import deque, defaultdict
from typing import Any, Dict, Optional, List
from datetime import datetime

# Shared Imports
from shared import (
    CONFIG,
    OnlineFeatureEngineer,
    AdaptiveTripleBarrier, 
    ProbabilityCalibrator,
    RiskManager,
    enrich_with_d1_data,
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
            horizon_ticks=tbm_conf.get('horizon_minutes', 60),
            risk_mult=CONFIG['risk_management']['stop_loss_atr_mult'],
            reward_mult=tbm_conf.get('barrier_width', 2.0)
        )

        # 3. Meta Labeler (The Gatekeeper)
        self.meta_labeler = MetaLabeler()
        self.meta_label_events = 0  # Count events to bypass cold start

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
        
        # --- FORENSIC RECORDER ---
        self.decision_log = deque(maxlen=1000)
        self.trade_events = []
        self.rejection_stats = defaultdict(int)  # Track why we didn't trade

    def on_data(self, snapshot: MarketSnapshot, broker: BacktestBroker):
        """
        Main Event Loop for the Strategy.
        """
        # Data Extraction
        price = snapshot.get_price(self.symbol, 'close')
        high = snapshot.get_high(self.symbol)
        low = snapshot.get_low(self.symbol)
        volume = snapshot.get_price(self.symbol, 'volume')
        
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
            else:
                timestamp = float(snapshot.timestamp)
        except Exception:
            timestamp = 0.0

        # Flow Volumes
        buy_vol = snapshot.get_price(self.symbol, 'buy_vol')
        sell_vol = snapshot.get_price(self.symbol, 'sell_vol')
        
        # Fallback to Tick Rule if flow missing
        if buy_vol == 0 and sell_vol == 0:
            if self.last_price > 0:
                if price > self.last_price:
                    buy_vol = volume
                    sell_vol = 0
                elif price < self.last_price:
                    buy_vol = 0
                    sell_vol = volume
                else:
                    buy_vol = volume / 2
                    sell_vol = volume / 2
            else:
                buy_vol = volume / 2
                sell_vol = volume / 2
        
        self.last_price = price

        # A. Feature Engineering (Recursive Update)
        features = self.fe.update(
            price=price,
            timestamp=timestamp,
            volume=volume,
            high=high,
            low=low,
            buy_vol=buy_vol,
            sell_vol=sell_vol,
            time_feats={}
        )
        self.last_features = features

        # --- WARM-UP GATE ---
        if self.burn_in_counter < self.burn_in_limit:
            self.burn_in_counter += 1
            if self.burn_in_counter == self.burn_in_limit:
                self.burn_in_complete = True
            return

        self.bars_processed += 1

        # B. Delayed Training (Label Resolution via Adaptive Barrier)
        # UPDATED: Unpack realized_ret for Weighted Learning
        resolved_labels = self.labeler.resolve_labels(high, low, current_close=price)
        
        if resolved_labels:
            for (stored_feats, outcome_label, realized_ret) in resolved_labels:
                # --- PROFIT WEIGHTED LEARNING ---
                # Weight = Base Weight * Log(1 + |Return| * 100)
                # Helps model focus on the Trades that made MONEY, not just small wiggles.
                
                w_pos = self.params.get('positive_class_weight', 10.0)
                w_neg = self.params.get('negative_class_weight', 1.0)
                
                base_weight = w_pos if outcome_label != 0 else w_neg
                
                # Dynamic Scalar: Returns are typically 0.001 to 0.01. 
                # Multiply by 100 to get percentage like 0.1 to 1.0.
                # log(1 + 1.0) = 0.69. log(1 + 0.1) = 0.09.
                # Big wins get ~7x more weight than small wins.
                ret_scalar = math.log1p(abs(realized_ret) * 100.0)
                
                # Clamp scalar to avoid exploding gradients/weights (min 0.5 for stability)
                ret_scalar = max(0.5, ret_scalar)
                
                final_weight = base_weight * ret_scalar
                
                self.model.learn_one(stored_feats, outcome_label, sample_weight=final_weight)
                
                # Update Meta Labeler (Gatekeeper)
                if outcome_label != 0:
                    # Meta Labeler learns: "Given these features + action, did we profit?"
                    # We treat realized_ret > 0 as profit
                    self.meta_labeler.update(stored_feats, primary_action=outcome_label, outcome_pnl=realized_ret)

        # C. Add CURRENT Bar as new Trade Opportunity
        current_atr = features.get('atr', 0.0)
        self.labeler.add_trade_opportunity(features, price, current_atr, timestamp)

        # D. Inference
        try:
            # --- FILTER RELAXATION ---
            entropy_val = features.get('entropy', 0)
            entropy_thresh = self.params.get('entropy_threshold', 0.85)
            
            if entropy_val > entropy_thresh:
                self.rejection_stats['High Entropy'] += 1
                return

            vpin_val = features.get('vpin', 0)
            vpin_thresh = self.params.get('vpin_threshold', 0.85)
            
            if vpin_val > vpin_thresh:
                self.rejection_stats['High VPIN'] += 1
                return

            # Primary Prediction
            pred_class = self.model.predict_one(features)
            pred_proba = self.model.predict_proba_one(features)
            
            try:
                pred_action = int(pred_class)
            except:
                pred_action = 0
                
            prob_buy = pred_proba.get(1, 0.0)
            prob_sell = pred_proba.get(-1, 0.0)

            # E. Execution Logic & Discovery Override
            dt_timestamp = datetime.fromtimestamp(timestamp) if timestamp > 0 else datetime.now()
            
            # --- DISCOVERY MODE LOGIC (AGGRESSIVE FIX) ---
            # If we are in the early phase (first 2500 bars), and the model is
            # predicting 0 (Hold), we check if there is ANY bias in probability.
            # If so, we FORCE an action to generate PnL data.
            is_discovery = self.bars_processed < 2500
            effective_action = pred_action
            discovery_triggered = False

            if is_discovery and effective_action == 0:
                max_prob = max(prob_buy, prob_sell)
                
                # 1. Cold Start Epsilon-Greedy (20% Random Exploration)
                # Fixes paralysis when model probabilities are exactly 0.0 (fresh model)
                if max_prob == 0.0:
                    if random.random() < 0.2:  # 20% chance to force trade
                        effective_action = random.choice([1, -1])
                        discovery_triggered = True
                
                # 2. Weak Signal Amplification (Bias Check)
                # Only runs if model has some opinion (max_prob > 0)
                else:
                    # AGGRESSIVE: If model is even 5% confident in a direction, take it.
                    bias_buy = prob_buy > 0.05
                    bias_sell = prob_sell > 0.05
                    
                    if bias_buy and prob_buy > prob_sell:
                        effective_action = 1
                        discovery_triggered = True
                    elif bias_sell and prob_sell > prob_buy:
                        effective_action = -1
                        discovery_triggered = True
                
                # 3. Fallback: Momentum Injection (Volatility Breakout)
                # If neither epsilon nor bias triggered, but volatility is high, trade the trend.
                if not discovery_triggered and features.get('vol_breakout', 0) > 0:
                    ret = features.get('return_raw', 0)
                    if ret > 0:
                        effective_action = 1
                        discovery_triggered = True
                    elif ret < 0:
                        effective_action = -1
                        discovery_triggered = True

            # --- META LABELING ---
            if self.meta_label_events < 50 or discovery_triggered:
                is_profitable = True
            else:
                is_profitable = self.meta_labeler.predict(
                    features,
                    effective_action,
                    threshold=self.params.get('meta_labeling_threshold', 0.55)
                )
            
            if effective_action != 0:
                self.meta_label_events += 1

            # --- EXECUTION ---
            if effective_action == 1 or effective_action == -1:
                if is_profitable:
                        confidence = prob_buy if effective_action == 1 else prob_sell
                        # Ensure confidence isn't 0.0 passed to risk manager
                        if confidence < 0.01: confidence = 0.5
                        
                        self._execute_logic(confidence, price, features, broker, dt_timestamp, effective_action, discovery_triggered)
                else:
                    self.rejection_stats['Meta-Labeler'] += 1
            else:
                self.rejection_stats['Model Predicted 0'] += 1

        except Exception as e:
            if self.debug_mode: logger.error(f"Strategy Error: {e}")
            pass

    def _inject_auxiliary_data(self):
        """Injects static approximations ONLY if missing."""
        defaults = {
            "USDJPY": 150.0, "GBPUSD": 1.25, "EURUSD": 1.08,
            "USDCAD": 1.35, "USDCHF": 0.90, "AUDUSD": 0.65, "NZDUSD": 0.60
        }
        for sym, price in defaults.items():
            if sym not in self.last_price_map:
                self.last_price_map[sym] = price

    def _execute_logic(self, confidence, price, features, broker, timestamp: datetime, action_int: int, discovery_mode: bool):
        """Decides whether to enter a trade using Inverse-Volatility Sizing."""
        
        # 1. Signal Threshold
        min_prob = self.params.get('min_calibrated_probability', 0.50)
        
        # --- DISCOVERY MODE OVERRIDE ---
        if not discovery_mode:
            if confidence < min_prob:
                self.rejection_stats['Low Confidence'] += 1
                return

        action = "BUY" if action_int == 1 else "SELL"

        # 2. Position Sizing & Risk
        if broker.get_position(self.symbol): return
        
        volatility = features.get('volatility', 0.001)
        current_atr = features.get('atr', 0.001)

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
            # If discovery mode, pretend we have valid confidence to get a non-zero size
            conf=confidence if not discovery_mode else 0.5,
            volatility=volatility,
            active_correlations=0,
            market_prices=self.last_price_map,
            atr=current_atr
        )

        # 3. SAFETY: Check if Risk Manager Rejected the trade
        if trade_intent.volume <= 0 or trade_intent.action == "HOLD":
            self.rejection_stats[f"Risk: {trade_intent.comment}"] += 1
            return

        qty = trade_intent.volume
        stop_dist = trade_intent.stop_loss
        tp_dist = trade_intent.take_profit

        if qty < 0.01:
            self.rejection_stats['Zero Size (Risk)'] += 1
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
            comment=f"{trade_intent.comment}|Prob:{confidence:.2f}|{'DISC' if discovery_mode else 'AI'}"
        )
        broker.submit_order(order)
        
        # Record detailed trade event
        self.trade_events.append({
            'time': timestamp,
            'action': action,
            'price': price,
            'conf': confidence,
            'vpin': features.get('vpin', 0),
            'entropy': features.get('entropy', 0),
            'atr': current_atr,
            'volatility': volatility,
            'forced': discovery_mode
        })

    def generate_autopsy(self) -> str:
        """
        Generates a text report explaining WHY the strategy behaved this way.
        """
        if not self.trade_events:
            reject_str = ", ".join([f"{k}: {v}" for k, v in self.rejection_stats.items()])
            status = "Waiting for Warm-Up" if not self.burn_in_complete else "Analysis Paralysis"
            return f"AUTOPSY: No trades. Status: {status}. Rejections: {{{reject_str}}}. Bars processed: {self.bars_processed}"
        
        avg_conf = np.mean([t['conf'] for t in self.trade_events])
        avg_vpin = np.mean([t['vpin'] for t in self.trade_events])
        forced_count = sum(1 for t in self.trade_events if t['forced'])
        
        reject_str = ", ".join([f"{k}: {v}" for k, v in self.rejection_stats.items()])
        
        try:
            conf_values = [t['conf'] for t in self.trade_events]
            conf_bins = np.histogram(conf_values, bins=5, range=(0.0, 1.0))[0]
            dist_str = f"{list(conf_bins)}"
        except Exception:
            dist_str = "[]"

        report = (
            f"\n --- 💀 STRATEGY AUTOPSY ({self.symbol}) ---\n"
            f" Trades: {len(self.trade_events)} (Discovery Mode: {forced_count})\n"
            f" Avg Conf: {avg_conf:.2f}\n"
            f" Conf Dist: {dist_str}\n"
            f" Avg VPIN: {avg_vpin:.2f}\n"
            f" Rejections: {{{reject_str}}}\n"
            f" ----------------------------------------\n"
        )
        return report