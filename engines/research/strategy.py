# FILENAME: engines/research/strategy.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/strategy.py
# DEPENDENCIES: shared, river, engines.research.backtester
# DESCRIPTION: The Adaptive Strategy Kernel.
# FIX: Integrated MetaLabeler and RandomUnderSampler logic to match Live Engine.
# AUDIT REMEDIATION:
#   - FIXED: Simulation Data Gap. Injects synthetic cross-rates for single-symbol backtests.
#   - ADDED: Debug Mode flag for verbose error logging.
#   - IMPROVED: Autopsy generation for rich console telemetry.
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
    StreamingTripleBarrier,
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
    Manages its own Feature Engineering, TBM Labeler, and River Model.
    """
    def __init__(self, model: Any, symbol: str, params: dict[str, Any]):
        self.model = model
        self.symbol = symbol
        self.params = params
        self.debug_mode = False # AUDIT FIX: Can be enabled by main_research for deep dives

        # 1. Feature Engineer (The Eyes)
        self.fe = OnlineFeatureEngineer(
            window_size=params.get('feature_window', 50)
        )

        # 2. Triple Barrier Labeler (The Teacher)
        self.tbm = StreamingTripleBarrier(
            vol_multiplier=params.get('tbm', {}).get('barrier_width', 2.0),
            barrier_len=50,
            horizon_ticks=params.get('tbm', {}).get('horizon_minutes', 60)
        )

        # 3. Meta Labeler (The Gatekeeper)
        self.meta_labeler = MetaLabeler()

        # 4. Probability Calibrators
        self.calibrator_buy = ProbabilityCalibrator(window=2000)
        self.calibrator_sell = ProbabilityCalibrator(window=2000)

        # State
        # Tuple: (features, predicted_action)
        self.feature_history = {} 
        self.last_features = None
        self.last_price = 0.0
        self.last_price_map = {} # Cache for Risk Manager
      
        # --- FORENSIC RECORDER ---
        self.decision_log = deque(maxlen=1000)
        self.trade_events = [] 

    def on_data(self, snapshot: MarketSnapshot, broker: BacktestBroker):
        """
        Main Event Loop for the Strategy.
        """
        price = snapshot.get_price(self.symbol, 'close')
        volume = snapshot.get_price(self.symbol, 'volume')
        
        # Update Price Map for Risk Calculation
        self.last_price_map = snapshot.to_price_dict()
        # Ensure current symbol is in map if single-mode
        if self.symbol not in self.last_price_map:
            self.last_price_map[self.symbol] = price

        # AUDIT FIX: Inject Auxiliary Data if missing (e.g. for Single Symbol Backtest)
        # This prevents RiskManager from returning 0 size due to missing Cross Rates
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
      
        # Fallback to Tick Rule
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

        # A. Feature Engineering
        features = self.fe.update(
            price=price,
            timestamp=timestamp,
            volume=volume,
            buy_vol=buy_vol,
            sell_vol=sell_vol,
            time_feats={}
        )
        self.last_features = features

        # B. Labeling & Training
        resolved_events = self.tbm.update(price, timestamp)
        for label, origin_ts in resolved_events:
            if origin_ts in self.feature_history:
                X, pred_action = self.feature_history[origin_ts]
                y_primary = label # -1, 0, 1
              
                # Train Primary
                self.model.learn_one(X, y_primary)

                # Train Meta-Labeler
                # PnL Proxy: 1.0 if Label matches Pred (Win), -1.0 if not
                pnl_proxy = 0.0
                if pred_action != 0:
                    if pred_action == y_primary:
                        pnl_proxy = 1.0
                    else:
                        pnl_proxy = -1.0
                
                self.meta_labeler.update(X, pred_action, pnl_proxy)
              
                del self.feature_history[origin_ts]

        # C. Inference
        if len(self.fe.prices) < 50: return 

        try:
            # Forensic Filters
            entropy_val = features.get('entropy', 0)
            entropy_thresh = self.params.get('entropy_threshold', 0.95)
            
            if entropy_val > entropy_thresh:
                # Store history even if we hold, but pred is 0
                self.feature_history[timestamp] = (features, 0)
                return

            if features.get('vpin', 0) > 0.70:
                self.feature_history[timestamp] = (features, 0)
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

            # Store for Training
            self.feature_history[timestamp] = (features, pred_action)

            # D. Execution Logic
            dt_timestamp = datetime.fromtimestamp(timestamp) if timestamp > 0 else datetime.now()
            
            # Meta Check
            is_profitable = self.meta_labeler.predict(
                features, 
                pred_action, 
                threshold=self.params.get('meta_labeling_threshold', 0.60)
            )

            if pred_action != 0 and is_profitable:
                 self._execute_logic(prob_buy, prob_sell, price, features, broker, dt_timestamp, pred_action)

        except Exception as e:
            if self.debug_mode: logger.error(f"Strategy Error: {e}")
            pass

    def _inject_auxiliary_data(self):
        """
        AUDIT FIX:
        If backtesting a single symbol (e.g. AUDJPY), the snapshot will NOT contain USDJPY.
        This causes RiskManager to return 0 size (Conversion Rate 0).
        We inject static approximations ONLY if missing, to allow optimization to proceed.
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

    def _execute_logic(self, p_buy, p_sell, price, features, broker, timestamp: datetime, action_int: int):
        """Decides whether to enter a trade using RCK Sizing."""
        
        # 1. Signal Threshold
        min_prob = self.params.get('min_calibrated_probability', 0.60)
        confidence = p_buy if action_int == 1 else p_sell
        
        if confidence < min_prob:
            return

        action = "BUY" if action_int == 1 else "SELL"

        # 2. Position Sizing & Risk
        if broker.get_position(self.symbol): return
      
        atr = features.get('volatility', 0.001)
        if atr == 0: atr = 0.001

        ctx = TradeContext(
            symbol=self.symbol,
            price=price,
            stop_loss_price=0.0,
            account_equity=broker.equity,
            account_currency="USD",
            win_rate=0.55,
            risk_reward_ratio=1.5
        )

        # AUDIT FIX: Pass last_price_map (now enriched) to RiskManager
        trade_intent, risk_usd = RiskManager.calculate_rck_size(
            context=ctx,
            conf=confidence,
            volatility=atr,
            active_correlations=0,
            market_prices=self.last_price_map
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
            'hurst': features.get('hurst', 0.5),
            'volatility': atr
        })

    def generate_autopsy(self) -> str:
        """
        Generates a text report explaining WHY the strategy behaved this way.
        Used by main_research.py to debug low-scoring trials.
        """
        if not self.trade_events:
            return "AUTOPSY: No trades taken. Strategy was too conservative (Entropy/VPIN blocked all signals) or data was noise."
      
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
      
        # Show first 3 and last 3 trades for brevity in console
        display_trades = self.trade_events[:3]
        if len(self.trade_events) > 6:
            display_trades.extend(self.trade_events[-3:])
        elif len(self.trade_events) > 3:
            display_trades = self.trade_events # Show all if small enough
            
        for i, t in enumerate(display_trades):
            ts_str = str(t['time'])
            report += f"   Trade: {t['action']} @ {ts_str} | Conf:{t['conf']:.2f} | VPIN:{t['vpin']:.2f}\n"
          
        return report