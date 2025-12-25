# =============================================================================
# FILENAME: engines/research/strategy.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/strategy.py
# DEPENDENCIES: shared, river, engines.research.backtester
# DESCRIPTION: The Adaptive Strategy Kernel.
#
# PHOENIX STRATEGY UPGRADE (2025-12-25 - DOCUMENT COMPLIANCE):
# 1. LOGIC OVERHAUL: Implemented "Regime-Adaptive Mean Reversion".
# 2. TRIGGER: Bollinger Band Reversion (2.5 SD).
# 3. FILTER: ADX < 25 (Ranging Market).
# 4. CONFIRMATION: Microstructure OFI Divergence.
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
        # Note: Now includes FracDiff, Microstructure, BB, and ADX kernels
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
        self.meta_warmup_limit = 50 
        
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
        self.rejection_stats = defaultdict(int) 
        self.feature_importance_counter = Counter() # Tracks top features
        
        # --- STRATEGY PARAMETERS (From Config/Document) ---
        self.adx_threshold = CONFIG['features']['adx']['threshold'] # 25
        self.bb_dev = CONFIG['features']['bollinger_bands']['std_dev'] # 2.5
        
        # Load Spread Assumptions for Gating
        self.spread_map = CONFIG.get('forensic_audit', {}).get('spread_pips', {})
        self.default_spread = self.spread_map.get('default', 1.5)

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

        # Flow Volumes (Critical for OFI)
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

        # A. Feature Engineering
        # (Generates frac_diff and micro_ofi internally)
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
        resolved_labels = self.labeler.resolve_labels(high, low, current_close=price)
        
        if resolved_labels:
            for (stored_feats, outcome_label, realized_ret) in resolved_labels:
                # --- PROFIT WEIGHTED LEARNING ---
                w_pos = self.params.get('positive_class_weight', 2.0)
                w_neg = self.params.get('negative_class_weight', 2.0)
                
                base_weight = w_pos if outcome_label != 0 else w_neg
                
                # Scale by Profit Magnitude
                ret_scalar = math.log1p(abs(realized_ret) * 100.0)
                ret_scalar = max(0.5, ret_scalar)
                
                final_weight = base_weight * ret_scalar
                
                self.model.learn_one(stored_feats, outcome_label, sample_weight=final_weight)
                
                # Double Learn for Positive outcomes
                if outcome_label != 0:
                     self.model.learn_one(stored_feats, outcome_label, sample_weight=final_weight)

                if outcome_label != 0:
                    self.meta_labeler.update(stored_feats, primary_action=outcome_label, outcome_pnl=realized_ret)

        # C. Add CURRENT Bar as new Trade Opportunity
        current_atr = features.get('atr', 0.0)
        self.labeler.add_trade_opportunity(features, price, current_atr, timestamp)

        # ============================================================
        # D. STRATEGY LOGIC: GBP/JPY Regime-Adaptive Mean Reversion
        # ============================================================
        
        # Extract Key Indicators
        adx_val = features.get('adx', 50.0)
        bb_upper = features.get('bb_upper', 999999.0)
        bb_lower = features.get('bb_lower', 0.0)
        micro_ofi = features.get('micro_ofi', 0.0)
        
        proposed_action = 0 # 0=HOLD, 1=BUY, -1=SELL
        rejection_reason = ""
        
        # --- CONDITION 1: REGIME IDENTIFICATION (FILTER) ---
        # Logic: If ADX > 25, Market is Trending -> DISABLE Mean Reversion.
        if adx_val > self.adx_threshold:
            self.rejection_stats[f"Trend Mode (ADX {adx_val:.1f})"] += 1
            # We exit early because this strategy is strictly Mean Reversion
            return 

        # --- CONDITION 2: THE TRIGGER (BOLLINGER BANDS) ---
        # Short Signal: Price > Upper Band
        if price > bb_upper:
            proposed_action = -1 # Sell
        # Long Signal: Price < Lower Band
        elif price < bb_lower:
            proposed_action = 1 # Buy
        else:
            # Inside bands -> No Trigger
            return

        # --- CONDITION 3: MICROSTRUCTURE CONFIRMATION (OFI) ---
        # Logic: Wait for OFI to contradict price direction.
        # If Price High (Short Trigger), we need Sellers (OFI < -threshold).
        # If Price Low (Long Trigger), we need Buyers (OFI > threshold).
        
        ofi_threshold = 2.0 # Aggressive flow required
        
        if proposed_action == -1: # Selling
            if micro_ofi >= -ofi_threshold: # Not enough selling pressure yet
                self.rejection_stats["OFI Wait (Price High, No Sellers)"] += 1
                return
        elif proposed_action == 1: # Buying
            if micro_ofi <= ofi_threshold: # Not enough buying pressure yet
                self.rejection_stats["OFI Wait (Price Low, No Buyers)"] += 1
                return

        # ============================================================
        # E. ML CONFIRMATION & EXECUTION
        # ============================================================
        
        # If we passed the Hard Rules, we consult the ML Model for Sizing/Confidence
        
        try:
            # Primary Prediction
            pred_class = self.model.predict_one(features)
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
            # We override the ML's directional decision with our Rule-Based Trigger,
            # but we use the ML's confidence for sizing.
            # If ML strongly disagrees (e.g. Rule=Buy, ML=Sell prob 0.9), we might skip.
            
            # Safety Check: If ML thinks probability is terrible (< 0.4), skip even if Rule triggers.
            if confidence < 0.40:
                self.rejection_stats["ML Disagreement"] += 1
                return

            if is_profitable:
                dt_timestamp = datetime.fromtimestamp(timestamp) if timestamp > 0 else datetime.now()
                # Discovery Mode is effectively disabled as we have hard rules now
                self._execute_logic(confidence, price, features, broker, dt_timestamp, proposed_action, discovery_mode=False)
            else:
                self.rejection_stats['Meta-Labeler'] += 1

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
        """Decides whether to enter a trade using Volatility Targeting."""
        
        # 1. Signal Threshold
        min_prob = self.params.get('min_calibrated_probability', 0.85)
        
        # Rule-based triggers are high quality, so we can relax ML confidence slightly
        # if the setup is perfect (e.g. Extreme BB deviation).
        # For now, we respect the config.
        if confidence < min_prob:
            self.rejection_stats['Low Confidence'] += 1
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
        
        # Track Features for Explainability
        imp_feats = []
        if features.get('adx', 0) < 25: imp_feats.append('Ranging_Mode')
        if features.get('bb_position', 0.5) > 1.0: imp_feats.append('BB_Upper_Break')
        elif features.get('bb_position', 0.5) < 0.0: imp_feats.append('BB_Lower_Break')
        
        micro_ofi = features.get('micro_ofi', 0.0)
        if micro_ofi > 2.0: imp_feats.append('OFI_Strong_Buy')
        elif micro_ofi < -2.0: imp_feats.append('OFI_Strong_Sell')
        
        for f in imp_feats:
            self.feature_importance_counter[f] += 1

        self.trade_events.append({
            'time': timestamp,
            'action': action,
            'price': price,
            'conf': confidence,
            'vpin': features.get('vpin', 0),
            'entropy': features.get('entropy', 0),
            'ker': features.get('ker', 0),
            'fdi': features.get('fdi', 0),
            'atr': current_atr,
            'volatility': volatility,
            'frac_diff': features.get('frac_diff', 0.0),
            'micro_ofi': micro_ofi,
            'forced': discovery_mode,
            'top_feats': imp_feats
        })

    def generate_autopsy(self) -> str:
        """
        Generates a text report explaining WHY the strategy behaved this way.
        """
        if not self.trade_events:
            reject_str = ", ".join([f"{k}: {v}" for k, v in self.rejection_stats.items()])
            status = "Waiting for Warm-Up" if not self.burn_in_complete else "No Trigger Conditions Met"
            return f"AUTOPSY: No trades. Status: {status}. Rejections: {{{reject_str}}}. Bars processed: {self.bars_processed}"
        
        avg_conf = np.mean([t['conf'] for t in self.trade_events])
        avg_vpin = np.mean([t['vpin'] for t in self.trade_events])
        avg_ofi = np.mean([t['micro_ofi'] for t in self.trade_events]) 
        avg_frac = np.mean([t['frac_diff'] for t in self.trade_events]) 
        
        forced_count = sum(1 for t in self.trade_events if t['forced'])
        
        reject_str = ", ".join([f"{k}: {v}" for k, v in self.rejection_stats.items()])
        
        # Explainability Report
        top_features = self.feature_importance_counter.most_common(5)
        feat_str = str(top_features)
        
        try:
            conf_values = [t['conf'] for t in self.trade_events]
            conf_bins = np.histogram(conf_values, bins=5, range=(0.0, 1.0))[0]
            dist_str = f"{list(conf_bins)}"
        except Exception:
            dist_str = "[]"

        report = (
            f"\n --- 💀 STRATEGY AUTOPSY ({self.symbol}) ---\n"
            f" Trades: {len(self.trade_events)} (Discovery Mode: {forced_count})\n"
            f" Avg Conf: {avg_conf:.2f} | Dist: {dist_str}\n"
            f" Avg VPIN: {avg_vpin:.2f} | Avg OFI: {avg_ofi:.2f}\n"
            f" Avg FracDiff: {avg_frac:.4f}\n"
            f" Top Drivers: {feat_str}\n"
            f" Rejections: {{{reject_str}}}\n"
            f" ----------------------------------------\n"
        )
        return report