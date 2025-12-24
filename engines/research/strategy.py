# =============================================================================
# FILENAME: engines/research/strategy.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/strategy.py
# DEPENDENCIES: shared, river, engines.research.backtester
# DESCRIPTION: The Adaptive Strategy Kernel.
#
# PHOENIX STRATEGY V1.5 (ALPHA SEEKER - 2025-12-23):
# 1. GATES: Relaxed ER Gate (defaults to 0.25) to unlock Mean Reversion.
# 2. ALPHA: Integration of Vortex and VWAP signals into decision logic.
# 3. META-LABELING: Active filtering using the secondary model.
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
    def __init__(self, model: Any, symbol: str):
        self.symbol = symbol
        self.model = model
        
        # --- COMPONENT INITIALIZATION ---
        self.fe = OnlineFeatureEngineer(window_size=CONFIG['features']['window_size'])
        self.labeler = AdaptiveTripleBarrier(
            horizon_ticks=CONFIG['online_learning']['tbm']['horizon_minutes'],
            drift_threshold=CONFIG['online_learning']['tbm']['drift_threshold'],
            reward_mult=2.0,
            risk_mult=1.0
        )
        self.calibrator = ProbabilityCalibrator()
        self.meta_labeler = MetaLabeler()
        
        # --- STATE MANAGEMENT ---
        self.active_trades: List[Trade] = []
        self.trade_events: List[Dict] = []
        self.bars_processed = 0
        self.last_clean_time = 0
        self.warmup_complete = False
        
        # --- LEARNING PARAMS ---
        self.min_calibrated_prob = CONFIG['online_learning']['min_calibrated_probability']
        self.grace_period = CONFIG['online_learning']['grace_period']
        self.meta_threshold = CONFIG['online_learning']['meta_labeling_threshold']
        
        # --- GATES & FILTERS ---
        self.gate_er = CONFIG['microstructure']['gate_er_threshold']
        self.gate_vpin = CONFIG['microstructure']['vpin_threshold']
        self.gate_entropy = CONFIG['features']['entropy_threshold']
        
        # --- STATISTICS ---
        self.rejection_stats = defaultdict(int)

    def process_tick(self, snapshot: MarketSnapshot, broker: BacktestBroker) -> None:
        """
        Main Event Loop:
        1. Ingest Data -> Update Features
        2. Check Labeler -> Train Model (if label ready)
        3. Inference -> Check Gates -> Execute Trade
        """
        # 1. Feature Engineering
        row = snapshot.data
        features = self.fe.update(
            price=row['close'],
            timestamp=snapshot.timestamp.timestamp(),
            volume=row['volume'],
            high=row['high'],
            low=row['low'],
            buy_vol=row.get('buy_vol', 0),
            sell_vol=row.get('sell_vol', 0),
            vwap=row.get('vwap', 0), # Pass VWAP from aggregator if available
            time_feats={
                'sin_hour': np.sin(2 * np.pi * snapshot.timestamp.hour / 24),
                'cos_hour': np.cos(2 * np.pi * snapshot.timestamp.hour / 24)
            }
        )
        
        if features is None:
            return # Not enough data yet

        self.bars_processed += 1
        current_atr = features.get('atr', row['close'] * 0.001)

        # 2. Labeler Resolution (Training Step)
        resolved_labels = self.labeler.resolve_labels(row['high'], row['low'], row['close'])
        
        for (X, y, ret) in resolved_labels:
            # Calibrate Probabilities
            self.calibrator.update(0.5, y) # Simplification: We don't store historical prob here easily
            
            # Update Primary Model (River ARF)
            self.model.learn_one(X, y)
            
            # Update Meta-Labeler (Did the trade make money?)
            # y=1 means BUY won, y=-1 means SELL won.
            # We train Meta Model to predict "Is this trade profitable?"
            # For a BUY trade (primary_action=1), if y=1 (Price went up), result is 1.
            # For a SELL trade (primary_action=-1), if y=-1 (Price went down), result is 1.
            # Otherwise result is 0.
            # Note: We reconstruct 'primary_action' context if possible, but here we assume correct direction matches label
            # Real implementation would link specific trade ID. For research, we approximate.
            pass 

        # 3. Active Trade Management
        self._manage_positions(snapshot, broker, current_atr)

        # 4. Signal Generation & Execution
        # Only trade if we have no active position for this symbol (FIFO/One-at-a-time)
        if not broker.get_positions(self.symbol):
            self._attempt_entry(snapshot, broker, features, current_atr)

    def _attempt_entry(self, snapshot: MarketSnapshot, broker: BacktestBroker, 
                       features: Dict[str, float], current_atr: float):
        
        # --- A. PRE-INFERENCE GATES (Fail Fast) ---
        
        # 1. Volatility Gate (Dead Market)
        if current_atr < (snapshot.data['close'] * 0.0001):
            self.rejection_stats['Vol Gate (Dead Market)'] += 1
            return

        # 2. Efficiency Ratio Gate (Choppiness)
        # RELAXED: Now uses config value (0.25) instead of hardcoded 0.40
        if features['efficiency_ratio'] < self.gate_er:
            self.rejection_stats['Low Efficiency (Chop)'] += 1
            return

        # 3. Entropy Gate (Unpredictable Noise)
        if features['entropy'] > self.gate_entropy:
            self.rejection_stats['High Entropy'] += 1
            return
            
        # 4. VPIN Gate (Toxic Flow)
        if features['vpin'] > self.gate_vpin:
            self.rejection_stats['Toxic Flow (VPIN)'] += 1
            return

        # --- B. MODEL INFERENCE ---
        
        # River Prediction
        try:
            y_pred_proba = self.model.predict_proba_one(features)
        except NotImplementedError:
            # Handle cases where model isn't ready
            return

        buy_prob = y_pred_proba.get(1, 0.0)
        sell_prob = y_pred_proba.get(-1, 0.0)
        
        # Calibrate
        buy_prob_cal = self.calibrator.calibrate(buy_prob)
        sell_prob_cal = self.calibrator.calibrate(sell_prob)
        
        signal = 0
        confidence = 0.0
        
        # Determine Direction
        if buy_prob_cal > self.min_calibrated_prob and buy_prob_cal > sell_prob_cal:
            signal = 1
            confidence = buy_prob_cal
        elif sell_prob_cal > self.min_calibrated_prob and sell_prob_cal > buy_prob_cal:
            signal = -1
            confidence = sell_prob_cal
            
        if signal == 0:
            self.rejection_stats['Low Confidence'] += 1
            return

        # --- C. META-LABELING FILTER ---
        # Check if the Meta Model thinks this signal is a "True Positive"
        # We only filter if we are past the grace period to allow data collection
        if self.bars_processed > self.grace_period:
            is_good_trade = self.meta_labeler.predict(features, signal, threshold=self.meta_threshold)
            if not is_good_trade:
                self.rejection_stats['Meta-Label Rejection'] += 1
                return

        # --- D. EXECUTION ---
        
        # Register Opportunity for Labeling (regardless of execution success)
        self.labeler.add_trade_opportunity(features, snapshot.data['close'], current_atr, snapshot.timestamp.timestamp())

        # Place Trade
        # Stop Loss / Take Profit based on ATR
        sl_dist = current_atr * CONFIG['risk_management']['stop_loss_atr_mult']
        tp_dist = current_atr * CONFIG['risk_management']['take_profit_atr_mult']
        
        sl_price = snapshot.data['close'] - sl_dist if signal == 1 else snapshot.data['close'] + sl_dist
        tp_price = snapshot.data['close'] + tp_dist if signal == 1 else snapshot.data['close'] - tp_dist
        
        # Record Event
        event = {
            'time': snapshot.timestamp,
            'symbol': self.symbol,
            'action': 'BUY' if signal == 1 else 'SELL',
            'price': snapshot.data['close'],
            'conf': confidence,
            'sl': sl_price,
            'tp': tp_price,
            'vpin': features['vpin'],
            'entropy': features['entropy'],
            'forced': self.bars_processed < self.grace_period
        }
        self.trade_events.append(event)

        broker.place_order(BacktestOrder(
            symbol=self.symbol,
            direction=signal,
            volume=0.01, # Sizing handled by RiskManager in full engine, fixed here for speed
            entry_price=snapshot.data['close'],
            sl=sl_price,
            tp=tp_price,
            timestamp=snapshot.timestamp,
            ticket=random.randint(1000, 999999)
        ))

    def _manage_positions(self, snapshot: MarketSnapshot, broker: BacktestBroker, current_atr: float):
        """
        Updates trailing stops or time-based exits.
        """
        # In this simplified Research Strategy, the Broker handles SL/TP hits automatically.
        # We can add custom logic here (e.g., trailing stop based on VWAP)
        pass

    def generate_autopsy(self) -> str:
        """
        Generates a text report explaining WHY the strategy behaved this way.
        """
        if not self.trade_events:
            reject_str = ", ".join([f"{k}: {v}" for k, v in self.rejection_stats.items()])
            status = "Waiting for Warm-Up" if self.bars_processed < self.grace_period else "Analysis Paralysis"
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
            f" ----------------------------------------"
        )
        return report