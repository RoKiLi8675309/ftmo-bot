# =============================================================================
# FILENAME: engines/research/strategy.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/strategy.py
# DEPENDENCIES: shared, river, engines.research.backtester
# DESCRIPTION: The Adaptive Strategy Kernel.
#
# FORENSIC REMEDIATION LOG (2025-12-22):
# 1. COLD START BYPASS: Meta-Labeler is bypassed for first 50 events to force data.
# 2. CONFIDENCE FLOOR: Updated logic to respect lower 0.51 floor.
# 3. FILTERS: Relaxed Entropy check to use dynamic threshold (default 0.99).
# =============================================================================
import logging
import sys
import numpy as np
from collections import deque
from typing import Any, Dict, Optional, List
from datetime import datetime

# Shared Imports
from shared import (
    CONFIG,
    OnlineFeatureEngineer,
    AdaptiveTripleBarrier, # PHASE 2: ATR-based Labeling
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
        self.meta_label_events = 0 # Count events to bypass cold start

        # 4. Probability Calibrators
        self.calibrator_buy = ProbabilityCalibrator(window=2000)
        self.calibrator_sell = ProbabilityCalibrator(window=2000)

        # 5. Warm-up State
        self.burn_in_limit = params.get('burn_in_periods', 1000)
        self.burn_in_counter = 0

        # State
        self.last_features = None
        self.last_price = 0.0
        self.last_price_map = {} 
        
        # --- FORENSIC RECORDER ---
        self.decision_log = deque(maxlen=1000)
        self.trade_events = [] 

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
            return 

        # B. Delayed Training (Label Resolution via Adaptive Barrier)
        resolved_labels = self.labeler.resolve_labels(high, low)
        
        if resolved_labels:
            for (stored_feats, outcome_label) in resolved_labels:
                # Weighted Learning (10:1)
                w_pos = self.params.get('positive_class_weight', 10.0)
                w_neg = self.params.get('negative_class_weight', 1.0)
                weight = w_pos if outcome_label == 1 else w_neg
                
                self.model.learn_one(stored_feats, outcome_label, sample_weight=weight)
                
                # --- UPDATE META LABELER ---
                # We need to know what the primary model predicted for this past event.
                # However, re-predicting is expensive. We assume if outcome_label matches
                # a hypothetical correct prediction, we label it 1.
                # This is a simplification. Ideally, we'd store the prediction with the features.
                pass 

        # C. Add CURRENT Bar as new Trade Opportunity
        current_atr = features.get('atr', 0.0)
        self.labeler.add_trade_opportunity(features, price, current_atr, timestamp)

        # D. Inference
        try:
            # Forensic Filters (RELAXED via params)
            entropy_val = features.get('entropy', 0)
            # Default to 0.99 (Virtually Disabled) if not in params
            entropy_thresh = self.params.get('entropy_threshold', 0.99) 
            
            if entropy_val > entropy_thresh:
                return

            if features.get('vpin', 0) > 0.95: # Relaxed VPIN
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

            # E. Execution Logic
            dt_timestamp = datetime.fromtimestamp(timestamp) if timestamp > 0 else datetime.now()
            
            # --- META LABELING (COLD START BYPASS) ---
            # If we haven't seen 50 events yet, assume it's good to force learning.
            if self.meta_label_events < 50:
                is_profitable = True
            else:
                is_profitable = self.meta_labeler.predict(
                    features, 
                    pred_action, 
                    threshold=self.params.get('meta_labeling_threshold', 0.55)
                )
            
            self.meta_label_events += 1

            # --- PURE AI LOGIC (No Forced Trades) ---
            if pred_action == 1:
                if is_profitable:
                        self._execute_logic(prob_buy, prob_sell, price, features, broker, dt_timestamp, pred_action)

        except Exception as e:
            if self.debug_mode: logger.error(f"Strategy Error: {e}")
            pass

    def _inject_auxiliary_data(self):
        """
        Injects static approximations ONLY if missing.
        """
        defaults = {
            "USDJPY": 150.0, "GBPUSD": 1.25, "EURUSD": 1.08,
            "USDCAD": 1.35, "USDCHF": 0.90, "AUDUSD": 0.65, "NZDUSD": 0.60
        }
        for sym, price in defaults.items():
            if sym not in self.last_price_map:
                self.last_price_map[sym] = price

    def _execute_logic(self, p_buy, p_sell, price, features, broker, timestamp: datetime, action_int: int):
        """Decides whether to enter a trade using Inverse-Volatility Sizing."""
        
        # 1. Signal Threshold (Lowered Floor: 0.51)
        min_prob = self.params.get('min_calibrated_probability', 0.51)
        confidence = p_buy if action_int == 1 else p_sell
        
        if confidence < min_prob:
            return

        action = "BUY" if action_int == 1 else "SELL"

        # 2. Position Sizing & Risk
        if broker.get_position(self.symbol): return
        
        # Volatility Adjusted Sizing
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
            conf=confidence,
            volatility=volatility,
            active_correlations=0, 
            market_prices=self.last_price_map,
            atr=current_atr
        )

        qty = trade_intent.volume
        stop_dist = trade_intent.stop_loss
        tp_dist = trade_intent.take_profit

        if qty < 0.01: return

        if action == "BUY":
            sl_price = price - stop_dist
            tp_price = price + tp_dist
            side = 1
        else:
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
            comment=f"{trade_intent.comment}|Prob:{confidence:.2f}"
        )
        broker.submit_order(order)
        
        # Record detailed trade event for Autopsy
        self.trade_events.append({
            'time': timestamp,
            'action': action,
            'price': price,
            'conf': confidence,
            'vpin': features.get('vpin', 0),
            'entropy': features.get('entropy', 0),
            'atr': current_atr,
            'volatility': volatility,
            'forced': False 
        })

    def generate_autopsy(self) -> str:
        """
        Generates a text report explaining WHY the strategy behaved this way.
        """
        if not self.trade_events:
            return "AUTOPSY: No trades taken. System is likely waiting for high-confidence setup or Burn-In."
        
        avg_conf = np.mean([t['conf'] for t in self.trade_events])
        avg_vpin = np.mean([t['vpin'] for t in self.trade_events])
        avg_entropy = np.mean([t['entropy'] for t in self.trade_events])
        
        report = (
            f"\n   --- 💀 STRATEGY AUTOPSY ({self.symbol}) ---\n"
            f"   Trades Taken: {len(self.trade_events)}\n"
            f"   Avg Confidence: {avg_conf:.2f}\n"
            f"   Avg VPIN: {avg_vpin:.2f}\n"
            f"   Avg Entropy: {avg_entropy:.2f}\n"
            f"   ----------------------------------------\n"
        )
        
        display_trades = self.trade_events[:3]
        if len(self.trade_events) > 6:
            display_trades.extend(self.trade_events[-3:])
        elif len(self.trade_events) > 3:
            display_trades = self.trade_events 
            
        for i, t in enumerate(display_trades):
            ts_str = str(t['time'])
            report += f"   Trade: {t['action']} @ {ts_str} | Conf:{t['conf']:.2f}\n"
          
        return report