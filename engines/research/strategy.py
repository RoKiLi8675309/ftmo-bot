# =============================================================================
# FILENAME: engines/research/strategy.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/strategy.py
# DEPENDENCIES: shared, river, engines.research.backtester
# DESCRIPTION: The Adaptive Strategy Kernel (Backtesting Version).
#
# PHOENIX STRATEGY UPGRADE (2025-12-28 - V3.0 IRON GATE):
# 1. PHILOSOPHY: "Iron Gate" - Filter out the noise.
# 2. FIX: H4 RSI Alignment Required (>50 for Long, <50 for Short).
# 3. FIX: KER Raised to 0.40 for Regime B.
# 4. FIX: ADX Filter Enabled (>20) to kill dead markets.
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
        
        # Sync Labeler Risk with Config (V3.0 uses 1.5 ATR Stop)
        risk_mult_conf = CONFIG['risk_management'].get('stop_loss_atr_mult', 1.5)
        
        self.labeler = AdaptiveTripleBarrier(
            horizon_ticks=tbm_conf.get('horizon_minutes', 60),
            risk_mult=risk_mult_conf, 
            reward_mult=tbm_conf.get('barrier_width', 1.5),
            drift_threshold=tbm_conf.get('drift_threshold', 1.2)
        )
        
        # 3. Meta Labeler (The Gatekeeper)
        self.meta_labeler = MetaLabeler()
        self.meta_label_events = 0 
        
        # 4. Probability Calibrators
        self.calibrator_buy = ProbabilityCalibrator(window=2000)
        self.calibrator_sell = ProbabilityCalibrator(window=2000)
        
        # 5. Warm-up State
        self.burn_in_limit = params.get('burn_in_periods', 200) # Fast Startup
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
        self.feature_importance_counter = Counter() 
        
        # --- PHOENIX STRATEGY PARAMETERS (V3.0 IRON GATE CONFIG) ---
        phx_conf = CONFIG.get('phoenix_strategy', {})
        
        # V3.0 GATES
        self.enable_regime_a = phx_conf.get('enable_regime_a_entries', True)
        self.require_d1_trend = phx_conf.get('require_d1_trend', True)
        self.require_h4_alignment = phx_conf.get('require_h4_alignment', True)
        
        # Safety Cap
        self.max_rvol_thresh = phx_conf.get('max_relative_volume', 3.5)
        
        # Thresholds (V3.0: TIGHTENED)
        self.ker_thresh = phx_conf.get('ker_trend_threshold', 0.40) # Raised to 0.40
        self.adx_threshold = CONFIG['features']['adx'].get('threshold', 20) # Enabled at 20
        
        # Volume Gate
        self.vol_gate_ratio = phx_conf.get('volume_gate_ratio', 1.5)
        
        # Momentum (V2.8: 0.65)
        self.aggressor_thresh = phx_conf.get('aggressor_threshold', 0.65)
        self.vol_exp_thresh = phx_conf.get('vol_expansion_threshold', 2.0) 
        
        self.limit_order_offset_pips = CONFIG.get('trading', {}).get('limit_order_offset_pips', 0.2)
        
        # Friday Liquidation
        self.friday_close_hour = CONFIG.get('risk_management', {}).get('friday_liquidation_hour_server', 21)

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

        # --- FRIDAY LIQUIDATION CHECK (GAP PROTECTION) ---
        if dt_ts.weekday() == 4 and dt_ts.hour >= self.friday_close_hour:
            # FIXED: Access broker.positions property (Dict[str, BacktestOrder])
            if self.symbol in broker.positions:
                pos = broker.positions[self.symbol]
                broker._close_partial_position(pos, pos.quantity, price, dt_ts, "Friday Liquidation")
                if self.debug_mode: logger.info(f"🚫 {self.symbol} Liquidated for Weekend (Friday {dt_ts.hour}:00)")
            return # Block new entries
        # -------------------------------------------------

        # Flow Volumes (L2 Proxies handled by Feature Engineer)
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
        # D. PROJECT PHOENIX: LOGIC GATES (V3.0 IRON GATE)
        # ============================================================
        
        # 1. Extract Phoenix Indicators
        rvol = features.get('rvol', 1.0)
        ker_val = features.get('ker', 0.5)
        aggressor = features.get('aggressor', 0.5)
        atr_val = features.get('atr', 0.0001)
        parkinson = features.get('parkinson_vol', 0.0)
        mtf_align = features.get('mtf_alignment', 0.0)
        adx_val = features.get('adx', 0.0)
        
        # --- CRITICAL FILTER 1: VOLUME EXHAUSTION FILTER ---
        if rvol > self.max_rvol_thresh:
            self.rejection_stats[f"Volume Climax (RVol {rvol:.2f} > {self.max_rvol_thresh})"] += 1
            return # Force HOLD
            
        # --- CRITICAL FILTER 2: MTF TREND LOCK ---
        d1_ema = context_data.get('d1', {}).get('ema200', 0.0)
        d1_trend_up = (price > d1_ema) if d1_ema > 0 else True
        d1_trend_down = (price < d1_ema) if d1_ema > 0 else True
        
        # --- NEW: H4 RSI ALIGNMENT (V3.0) ---
        h4_rsi = context_data.get('h4', {}).get('rsi', 50.0)
        h4_bull = h4_rsi > 50
        h4_bear = h4_rsi < 50
        
        # Gate Definitions
        # V3.0: 1.5x Avg Volume (Standard Institutional Flow)
        vol_gate = rvol > self.vol_gate_ratio
        
        # Momentum Direction (Aggressor Ratio)
        # V2.8: 0.65 Threshold
        is_bullish_candle = aggressor > self.aggressor_thresh
        is_bearish_candle = aggressor < (1.0 - self.aggressor_thresh)
        
        proposed_action = 0 # 0=HOLD, 1=BUY, -1=SELL
        regime_label = "C (Noise)"
        
        # --- REGIME A: MOMENTUM IGNITION (Unified Flow) ---
        # Re-enabled logic for catching moves that are not perfect "trends" yet
        if self.enable_regime_a:
            # Check for Flow Alignment: D1 Trend + Micro Structure + VOLUME GATE
            if d1_trend_up and is_bullish_candle and vol_gate:
                # Require Efficiency (V3.0: KER > 0.40) - Tightened again to stop chop
                if ker_val > self.ker_thresh:
                    proposed_action = 1
                    regime_label = "A (Mom-Long)"
                else:
                    self.rejection_stats[f"Regime A: Low Efficiency (<{self.ker_thresh})"] += 1
            
            elif d1_trend_down and is_bearish_candle and vol_gate:
                if ker_val > self.ker_thresh:
                    proposed_action = -1
                    regime_label = "A (Mom-Short)"
                else:
                    self.rejection_stats[f"Regime A: Low Efficiency (<{self.ker_thresh})"] += 1

            # Diagnostic for Volume Gate Failure
            elif (d1_trend_up and is_bullish_candle) or (d1_trend_down and is_bearish_candle):
                if not vol_gate:
                    self.rejection_stats[f"Volume Gate Fail (RVol {rvol:.2f} < {self.vol_gate_ratio})"] += 1
                
        # --- REGIME B: EFFICIENT TREND CONTINUATION ---
        # Logic: Efficiency (KER) + Momentum + D1 Align + ADX check
        # V3.0: Added H4 RSI Alignment & ADX Check
        
        # 1. ADX Filter: Market must be trending
        is_trending = adx_val > self.adx_threshold
        
        if proposed_action == 0 and ker_val > self.ker_thresh and vol_gate:
            if not is_trending:
                self.rejection_stats[f"Regime B: Low ADX ({adx_val:.1f} < {self.adx_threshold})"] += 1
            else:
                if is_bullish_candle and d1_trend_up:
                    if self.require_h4_alignment and not h4_bull:
                        self.rejection_stats["Regime B: H4 RSI Mismatch (Long)"] += 1
                    else:
                        proposed_action = 1
                        regime_label = "B (Trend-Long)"
                        
                elif is_bearish_candle and d1_trend_down:
                    if self.require_h4_alignment and not h4_bear:
                        self.rejection_stats["Regime B: H4 RSI Mismatch (Short)"] += 1
                    else:
                        proposed_action = -1
                        regime_label = "B (Trend-Short)"
                else:
                    self.rejection_stats["Regime B: Weak Candle / Counter Trend"] += 1
        
        # --- REGIME C: NOISE ---
        if proposed_action == 0:
            regime_label = "C (Noise)"
            self.rejection_stats[f"Noise (KER {ker_val:.2f})"] += 1
            return # Explicit HOLD

        # --- FINAL MTF SAFETY CHECK ---
        if proposed_action == 1 and not d1_trend_up and self.require_d1_trend:
            self.rejection_stats[f"MTF Lock (Price {price:.2f} < EMA {d1_ema:.2f})"] += 1
            return
        if proposed_action == -1 and not d1_trend_down and self.require_d1_trend:
            self.rejection_stats[f"MTF Lock (Price {price:.2f} > EMA {d1_ema:.2f})"] += 1
            return

        # ---------------------------------------------------------------------
        # MEAN REVERSION FILTER (Safety Check)
        # Prevent "Selling the Hole" (Shorting when Price < BB Lower or RSI < 30)
        # ---------------------------------------------------------------------
        bb_pos = features.get('bb_position', 0.5)
        rsi_val = features.get('rsi_norm', 0.5) * 100.0
        
        if proposed_action == -1: # SELL
            if bb_pos < 0.0 or rsi_val < 30:
                self.rejection_stats[f"Mean Rev Filter (BB:{bb_pos:.2f}|RSI:{rsi_val:.2f})"] += 1
                return
        elif proposed_action == 1: # BUY
            if bb_pos > 1.0 or rsi_val > 70:
                self.rejection_stats[f"Mean Rev Filter (BB:{bb_pos:.2f}|RSI:{rsi_val:.2f})"] += 1
                return
        # ---------------------------------------------------------------------

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
                threshold=self.params.get('meta_labeling_threshold', 0.55)
            )
            
            if proposed_action != 0:
                self.meta_label_events += 1

            # --- EXECUTION ---
            # V2.9: Defaults to 0.60 (Lowered from 0.70)
            min_prob = self.params.get('min_calibrated_probability', 0.60)
            
            if confidence < min_prob:
                self.rejection_stats[f"Low Confidence ({confidence:.2f} < {min_prob})"] += 1
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
        """
        # H4 Approximation (Every 48 M5 bars)
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
            # Standard EMA Calculation
            if len(arr) < 200:
                ema = np.mean(arr)
            else:
                # Use standard EMA formula over buffer
                span = 200
                alpha = 2 / (span + 1)
                # Quick approximate EMA on buffer (last value is latest)
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
            "USDCAD": 1.35, "USDCHF": 0.90, "AUDUSD": 0.65, "NZDUSD": 0.60
        }
        for sym, price in defaults.items():
            if sym not in self.last_price_map:
                self.last_price_map[sym] = price

    def _execute_logic(self, confidence, price, features, broker, timestamp: datetime, action_int: int, regime: str):
        """Decides whether to enter a trade using Fixed Risk (Prop Firm Mode)."""
        
        action = "BUY" if action_int == 1 else "SELL"

        # Hard Microstructure Filter (Amihud & OFI)
        amihud = features.get('amihud', 0.0)
        if amihud > 15.0: 
             self.rejection_stats["High Illiquidity (Amihud)"] += 1
             # return 

        # Position Sizing
        # FIXED: Access broker.positions property
        if self.symbol in broker.positions: return
        
        volatility = features.get('volatility', 0.001)
        current_atr = features.get('atr', 0.001)
        current_ker = features.get('ker', 1.0) 

        ctx = TradeContext(
            symbol=self.symbol,
            price=price,
            stop_loss_price=0.0,
            account_equity=broker.equity,
            account_currency="USD",
            win_rate=0.45, 
            risk_reward_ratio=2.0
        )

        # Calculate Size (Fixed Risk)
        trade_intent, risk_usd = RiskManager.calculate_rck_size(
            context=ctx,
            conf=confidence,
            volatility=volatility,
            active_correlations=0,
            market_prices=self.last_price_map,
            atr=current_atr, 
            ker=current_ker,
            account_size=broker.equity
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

        # Submit Order with Enriched Metadata
        order = BacktestOrder(
            symbol=self.symbol,
            side=side,
            quantity=qty,
            timestamp_created=timestamp,
            stop_loss=sl_price,
            take_profit=tp_price,
            comment=f"{trade_intent.comment}|Regime:{regime}|Limit:{self.limit_order_offset_pips}p",
            # METADATA FOR CSV LOGGING
            metadata={
                'regime': regime,
                'confidence': float(confidence),
                'rvol': features.get('rvol', 0),
                'parkinson': features.get('parkinson_vol', 0),
                'ker': current_ker,
                'atr': current_atr
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
            'time': timestamp,
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