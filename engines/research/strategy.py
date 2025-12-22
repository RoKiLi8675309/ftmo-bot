# =============================================================================
# FILENAME: engines/research/strategy.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/strategy.py
# DEPENDENCIES: shared, river, engines.research.backtester
# DESCRIPTION: The Adaptive Strategy Kernel.
# AUDIT REMEDIATION (PHASE 2):
#   - WARM-UP: Enforced 1000-candle burn-in before Training/Inference.
#   - LABELING: Switched to AdaptiveTripleBarrier (ATR-based).
#   - LEARNING: Implemented Weighted Learning (10:1 Class Weights).
#   - FORCE UPDATE: Implemented 'Aggressive Mode' to match Live Engine (1 trade/day).
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
    LogSymbols
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
            window_size=params.get('feature_window', 50)
        )

        # 2. Adaptive Triple Barrier Labeler (The Teacher - Phase 2)
        tbm_conf = params.get('tbm', {})
        self.labeler = AdaptiveTripleBarrier(
            horizon_ticks=tbm_conf.get('horizon_minutes', 60), # Converted to ticks implicitly
            risk_mult=CONFIG['risk_management']['stop_loss_atr_mult'],
            reward_mult=tbm_conf.get('barrier_width', 2.0)
        )

        # 3. Meta Labeler (The Gatekeeper)
        self.meta_labeler = MetaLabeler()

        # 4. Probability Calibrators
        self.calibrator_buy = ProbabilityCalibrator(window=2000)
        self.calibrator_sell = ProbabilityCalibrator(window=2000)

        # 5. Warm-up State
        self.burn_in_limit = params.get('burn_in_periods', 1000)
        self.burn_in_counter = 0

        # State
        self.last_features = None
        self.last_price = 0.0
        self.last_price_map = {} # Cache for Risk Manager
       
        # --- FORENSIC RECORDER ---
        self.decision_log = deque(maxlen=1000)
        self.trade_events = [] 

        # --- FORCE TRADE STATE ---
        self.last_trade_date = None
        self.daily_trades = 0

    def on_data(self, snapshot: MarketSnapshot, broker: BacktestBroker):
        """
        Main Event Loop for the Strategy.
        """
        # Data Extraction
        price = snapshot.get_price(self.symbol, 'close')
        high = snapshot.get_high(self.symbol)
        low = snapshot.get_low(self.symbol)
        volume = snapshot.get_price(self.symbol, 'volume')
        
        # 0. Daily Reset Logic (Force Trade)
        # Snapshot timestamp is usually a datetime in Research
        current_date = snapshot.timestamp.date() if isinstance(snapshot.timestamp, datetime) else datetime.fromtimestamp(float(snapshot.timestamp)).date()
        
        if self.last_trade_date != current_date:
            self.daily_trades = 0
            self.last_trade_date = current_date
            
        # Determine Aggression
        aggressive = (self.daily_trades == 0)

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
        # Note: We MUST update features every bar to maintain recursive state (RSI/MACD)
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

        # --- PHASE 2: WARM-UP GATE ---
        if self.burn_in_counter < self.burn_in_limit:
            self.burn_in_counter += 1
            return # Skip Training/Inference, just warm up indicators

        # B. Delayed Training (Label Resolution via Adaptive Barrier)
        # Check if previous trades have concluded based on current High/Low
        resolved_labels = self.labeler.resolve_labels(high, low)
        
        if resolved_labels:
            for (stored_feats, outcome_label) in resolved_labels:
                # --- PHASE 2: WEIGHTED LEARNING ---
                # Positive (1) = 10.0, Negative (0) = 1.0
                w_pos = self.params.get('positive_class_weight', 10.0)
                w_neg = self.params.get('negative_class_weight', 1.0)
                
                weight = w_pos if outcome_label == 1 else w_neg
                
                # Train Primary Model
                self.model.learn_one(stored_feats, outcome_label, sample_weight=weight)

                # Train Meta-Labeler (Proxy)
                # Omitted in Strategy.py to match simplified training loop, 
                # but could be added if full fidelity is required.

        # C. Add CURRENT Bar as new Trade Opportunity
        # The labeler stores the features internally until resolution
        current_atr = features.get('atr', 0.0)
        self.labeler.add_trade_opportunity(features, price, current_atr, timestamp)

        # D. Inference
        try:
            # Forensic Filters
            # BYPASS IF AGGRESSIVE (Forced Entry)
            if not aggressive:
                entropy_val = features.get('entropy', 0)
                entropy_thresh = self.params.get('entropy_threshold', 0.70)
                
                if entropy_val > entropy_thresh:
                    return

                if features.get('vpin', 0) > 0.75:
                    return

            # Primary Prediction
            pred_class = self.model.predict_one(features)
            pred_proba = self.model.predict_proba_one(features)
            
            try:
                pred_action = int(pred_class)
            except:
                pred_action = 0
                
            prob_buy = pred_proba.get(1, 0.0)
            prob_sell = pred_proba.get(-1, 0.0) # Not used in Long-Only Phase 2 setup usually

            # E. Execution Logic
            dt_timestamp = datetime.fromtimestamp(timestamp) if timestamp > 0 else datetime.now()
            
            # Meta Check
            # BYPASS IF AGGRESSIVE (Forced Entry)
            is_profitable = True
            if not aggressive:
                is_profitable = self.meta_labeler.predict(
                    features, 
                    pred_action, 
                    threshold=self.params.get('meta_labeling_threshold', 0.60)
                )

            # --- FORCE TRADE LOGIC ---
            if aggressive:
                 # Lower confidence threshold (e.g. 0.40) to force activity
                 force_threshold = 0.40
                 if pred_action == 1 or prob_buy > force_threshold:
                     # Force Buy
                     self._execute_logic(prob_buy, prob_sell, price, features, broker, dt_timestamp, 1, aggressive=True)
            else:
                # Standard Logic
                if pred_action == 1:
                    if is_profitable:
                         self._execute_logic(prob_buy, prob_sell, price, features, broker, dt_timestamp, pred_action)

        except Exception as e:
            if self.debug_mode: logger.error(f"Strategy Error: {e}")
            pass

    def _inject_auxiliary_data(self):
        """
        Injects static approximations ONLY if missing, to allow optimization to proceed.
        """
        defaults = {
            "USDJPY": 150.0,
            "GBPUSD": 1.25,
            "EURUSD": 1.08,
            "USDCAD": 1.35,
            "USDCHF": 0.90,
            "AUDUSD": 0.65,
            "NZDUSD": 0.60
        }
        for sym, price in defaults.items():
            if sym not in self.last_price_map:
                self.last_price_map[sym] = price

    def _execute_logic(self, p_buy, p_sell, price, features, broker, timestamp: datetime, action_int: int, aggressive: bool = False):
        """Decides whether to enter a trade using Kelly-Vol Sizing."""
        
        # 1. Signal Threshold
        # If aggressive, we already qualified it in on_data via the force_threshold
        if not aggressive:
            min_prob = self.params.get('min_calibrated_probability', 0.75)
            confidence = p_buy if action_int == 1 else p_sell
            
            if confidence < min_prob:
                return
        else:
            confidence = p_buy if action_int == 1 else p_sell

        action = "BUY" if action_int == 1 else "SELL"

        # 2. Position Sizing & Risk
        if broker.get_position(self.symbol): return
       
        # Phase 2: Volatility Adjusted Sizing
        # We need ATR and Volatility from features
        volatility = features.get('volatility', 0.001)
        current_atr = features.get('atr', 0.001)

        ctx = TradeContext(
            symbol=self.symbol,
            price=price,
            stop_loss_price=0.0,
            account_equity=broker.equity,
            account_currency="USD",
            win_rate=0.55,
            risk_reward_ratio=2.0 # Updated to 2.0
        )

        trade_intent, risk_usd = RiskManager.calculate_rck_size(
            context=ctx,
            conf=confidence,
            volatility=volatility,
            active_correlations=0, # In backtest, we might not track this perfectly per-step
            market_prices=self.last_price_map,
            atr=current_atr # Phase 2: Explicitly pass ATR
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
            comment=f"{trade_intent.comment}|Prob:{confidence:.2f}|{'FORCED' if aggressive else 'NORMAL'}"
        )
        broker.submit_order(order)
        
        # Increment Daily Count on Submission
        self.daily_trades += 1
       
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
            'forced': aggressive
        })

    def generate_autopsy(self) -> str:
        """
        Generates a text report explaining WHY the strategy behaved this way.
        """
        if not self.trade_events:
            return "AUTOPSY: No trades taken. Strategy was too conservative (Entropy/VPIN blocked all signals) or data was noise."
       
        avg_conf = np.mean([t['conf'] for t in self.trade_events])
        avg_vpin = np.mean([t['vpin'] for t in self.trade_events])
        avg_entropy = np.mean([t['entropy'] for t in self.trade_events])
        forced_trades = len([t for t in self.trade_events if t.get('forced', False)])
       
        report = (
            f"\n   --- 💀 STRATEGY AUTOPSY ({self.symbol}) ---\n"
            f"   Trades Taken: {len(self.trade_events)}\n"
            f"   Forced Trades: {forced_trades}\n"
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
            forced_tag = "[FORCED]" if t.get('forced', False) else ""
            report += f"   Trade: {t['action']} @ {ts_str} | Conf:{t['conf']:.2f} | {forced_tag}\n"
          
        return report