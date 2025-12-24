# =============================================================================
# FILENAME: shared/financial/features.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/financial/features.py
# DEPENDENCIES: shared, numpy, numba, scipy, river (optional)
# DESCRIPTION: Mathematical kernels for Feature Engineering, Labeling, and Risk.
#
# PHOENIX STRATEGY UPGRADE (2025-12-23):
# 1. VPIN: Upgraded to Bulk Volume Classification (BVC) logic for toxicity detection.
# 2. STATIONARITY: Strict log-return normalization for all price-derived features.
# 3. REGIME: Enhanced Entropy calculation to detect market ordering vs chaos.
# 4. ROBUSTNESS: RecursiveEMA hardened against NaN/Inf injection.
#
# META-LABELING OPTIMIZATION (2025-12-24):
# 1. MODEL: Upgraded MetaLabeler to Adaptive Random Forest (ARF) for non-linear alpha.
# 2. INTERACTIONS: Added Volatility*Action and Hurst*Action interaction terms.
# =============================================================================
from __future__ import annotations
import math
import logging
import sys
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List, Tuple

# Shared Config
from shared.core.config import CONFIG

# Numba for high-performance JIT compilation (Guarded)
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func): return func

# Scipy for Entropy (Guarded)
try:
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# River / Sklearn imports (Guarded for Windows Producer compatibility)
try:
    from river import linear_model, forest, metrics
    from sklearn.isotonic import IsotonicRegression
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger("Features")

# --- 0. HELPER MATH KERNELS (ROBUST) ---
class RecursiveEMA:
    """
    Dependency-free Exponential Moving Average.
    Solves compatibility issues with changing River API signatures.
    Formula: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
    """
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = None

    def update(self, x: float):
        if x is None or math.isnan(x) or math.isinf(x):
            return # Skip bad values
            
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
    
    def get(self) -> float:
        return self.value if self.value is not None else 0.0

# --- 1. STREAMING INDICATORS (PHASE 2 CORE) ---
class StreamingIndicators:
    """
    Recursive implementation of technical indicators.
    Uses local RecursiveEMA to ensure stability and O(1) updates.
    """
    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26, macd_sig=9, atr_period=14):
        # MACD Components
        self.ema_fast = RecursiveEMA(alpha=2 / (macd_fast + 1))
        self.ema_slow = RecursiveEMA(alpha=2 / (macd_slow + 1))
        self.macd_signal = RecursiveEMA(alpha=2 / (macd_sig + 1))
        
        # RSI Components
        self.rsi_period = rsi_period
        self.rsi_avg_gain = RecursiveEMA(alpha=1 / rsi_period)
        self.rsi_avg_loss = RecursiveEMA(alpha=1 / rsi_period)
        self.prev_price = None
        
        # ATR Components (Volatility)
        self.atr_mean = RecursiveEMA(alpha=1 / atr_period)
        self.prev_close = None

    def update(self, price: float, high: float, low: float) -> Dict[str, float]:
        """
        Updates recursive state and returns current indicator values.
        """
        features = {}
        
        # 1. MACD Calculation
        self.ema_fast.update(price)
        self.ema_slow.update(price)
        self.macd_line = self.ema_fast.get() - self.ema_slow.get()
        self.macd_signal.update(self.macd_line)
        histogram = self.macd_line - self.macd_signal.get()
        
        features['macd_line'] = self.macd_line
        features['macd_hist'] = histogram
        
        # 2. RSI Calculation
        if self.prev_price is not None:
            change = price - self.prev_price
            gain = max(0.0, change)
            loss = max(0.0, -change)
            
            self.rsi_avg_gain.update(gain)
            self.rsi_avg_loss.update(loss)
            
            avg_gain = self.rsi_avg_gain.get()
            avg_loss = self.rsi_avg_loss.get()
            
            if avg_loss == 0:
                rsi = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            
            features['rsi'] = rsi
        else:
            features['rsi'] = 50.0
            
        # 3. ATR Calculation (True Range)
        if self.prev_close is not None:
            tr1 = high - low
            tr2 = abs(high - self.prev_close)
            tr3 = abs(low - self.prev_close)
            true_range = max(tr1, tr2, tr3)
            self.atr_mean.update(true_range)
            features['atr'] = self.atr_mean.get()
        else:
            # First tick fallback
            features['atr'] = high - low if (high > 0 and low > 0 and high != low) else 0.001
            
        # Update State
        self.prev_price = price
        self.prev_close = price
        return features

# --- 2. ADAPTIVE TRIPLE BARRIER (SELL SIGNAL ENABLED) ---
class AdaptiveTripleBarrier:
    """
    Volatility-Adaptive Labeling with Soft Drift Detection.
    Barriers expand/contract based on ATR.
    
    FIXED: Generates -1 labels for downside breaks to enable Shorting.
    AUDIT FIX: Drift Threshold Hardened to 0.75 to purge soft-label noise.
    """
    def __init__(self, horizon_ticks: int = 12, risk_mult: float = 1.0, reward_mult: float = 2.0, drift_threshold: float = 0.75):
        self.buffer = deque()
        self.time_limit = horizon_ticks
        self.risk_mult = risk_mult
        self.reward_mult = reward_mult
        self.drift_threshold = drift_threshold

    def add_trade_opportunity(self, features: Dict[str, float], entry_price: float, current_atr: float, timestamp: float):
        """
        Registers a potential trade setup (Hypothetical entry at current bar).
        """
        # Ensure ATR is valid to prevent zero-width barriers
        if current_atr <= 0: current_atr = entry_price * 0.0001
        
        # Dynamic Barriers based on ATR
        # Upper Barrier (Profit for Buy, Stop for Sell)
        upper_barrier = entry_price + (self.reward_mult * current_atr)
        
        # Lower Barrier (Stop for Buy, Profit for Sell)
        lower_barrier = entry_price - (self.risk_mult * current_atr)
        
        self.buffer.append({
            'features': features,
            'entry': entry_price,
            'tp': upper_barrier, # Top Barrier
            'sl': lower_barrier, # Bottom Barrier
            'atr': current_atr,  # Stored for Drift Calculation
            'start_time': timestamp,
            'age': 0
        })

    def resolve_labels(self, current_high: float, current_low: float, current_close: float = None) -> List[Tuple[Dict[str, float], int, float]]:
        """
        Checks active trades against current price action.
        Returns list of (features, label, realized_return).
        
        Label 1 = UP Move (Buy Signal)
        Label -1 = DOWN Move (Sell Signal)
        Label 0 = Noise/Hold
        """
        resolved = []
        active = deque()
        
        # Fallback if close not provided (use mid of high/low)
        if current_close is None:
            current_close = (current_high + current_low) / 2.0

        while self.buffer:
            trade = self.buffer.popleft()
            trade['age'] += 1
            
            label = None
            realized_ret = 0.0

            # 1. Did price hit Upper Barrier?
            if current_high >= trade['tp']:
                label = 1 # BUY SIGNAL
                realized_ret = (trade['tp'] - trade['entry']) / trade['entry']
            
            # 2. Did price hit Lower Barrier?
            elif current_low <= trade['sl']:
                label = -1 # SELL SIGNAL
                realized_ret = (trade['entry'] - trade['sl']) / trade['entry'] # Positive return for short
            
            # 3. Timeout? (Vertical Barrier)
            elif trade['age'] >= self.time_limit:
                # --- DRIFT LOGIC (Soft Labeling) ---
                # Calculate movement relative to entry
                drift = current_close - trade['entry']
                
                # Threshold to consider this a signal
                # AUDIT FIX: drift_threshold defaults to 0.75 now
                drift_req = trade['atr'] * self.drift_threshold
                
                if drift > drift_req:
                    label = 1 # SOFT BUY (Upward Drift)
                    realized_ret = (current_close - trade['entry']) / trade['entry']
                elif drift < -drift_req:
                    label = -1 # SOFT SELL (Downward Drift)
                    realized_ret = (trade['entry'] - current_close) / trade['entry']
                else:
                    label = 0 # TIMEOUT (True Noise)
                    realized_ret = 0.0

            if label is not None:
                resolved.append((trade['features'], label, realized_ret))
            else:
                active.append(trade)
        
        self.buffer = active
        return resolved

# --- 3. PROBABILITY CALIBRATOR ---
class ProbabilityCalibrator:
    def __init__(self, window: int = 1000):
        self.window = window
        self.y_true = deque(maxlen=window)
        self.y_prob = deque(maxlen=window)
        self.calibrator = None
        
        if ML_AVAILABLE:
            self.calibrator = IsotonicRegression(out_of_bounds='clip')

    def update(self, prob: float, label: int):
        self.y_prob.append(prob)
        self.y_true.append(label)

    def calibrate(self, raw_prob: float) -> float:
        if not ML_AVAILABLE or len(self.y_true) < 100:
            return raw_prob
        
        try:
            self.calibrator.fit(list(self.y_prob), list(self.y_true))
            return float(self.calibrator.predict([raw_prob])[0])
        except Exception:
            return raw_prob

# --- 4. META LABELER (UPGRADED) ---
class MetaLabeler:
    """
    Secondary model that learns whether a primary signal resulted in profit.
    Acts as a 'Gatekeeper' to filter False Positives.
    
    UPGRADE (2025-12-24): Switched to ARFClassifier for non-linear alpha detection.
    Interaction terms added to detect context-specific failures.
    """
    def __init__(self):
        self.model = None
        self.buffer = deque(maxlen=1000)
        
        if ML_AVAILABLE:
            # OPTIMIZATION: ARF is more robust than LogisticRegression for regime detection
            # We use a small ensemble (10 trees) for speed but high adaptability
            self.model = forest.ARFClassifier(
                n_models=10, 
                seed=42, 
                metric=metrics.F1()
            )

    def update(self, features: Dict[str, float], primary_action: int, outcome_pnl: float):
        if not ML_AVAILABLE or primary_action == 0:
            return
            
        # Target: 1 if PnL > 0 (Win), 0 if Loss
        y_meta = 1 if outcome_pnl > 0 else 0
        
        try:
            # Feature Engineering for Meta-Model
            meta_feats = self._enrich(features, primary_action)
            
            clean_features = self._sanitize(meta_feats)
            self.model.learn_one(clean_features, y_meta)
            self.buffer.append((clean_features, y_meta))
        except Exception as e:
            logger.error(f"MetaLabeler Update Error: {e}")

    def predict(self, features: Dict[str, float], primary_action: int, threshold: float = 0.55) -> bool:
        if not ML_AVAILABLE or primary_action == 0:
            return False
            
        try:
            meta_feats = self._enrich(features, primary_action)
            clean_features = self._sanitize(meta_feats)
            
            probs = self.model.predict_proba_one(clean_features)
            prob_profit = probs.get(1, 0.0)
            
            # If the model is confident (> threshold) that this trade will PROFIT (class 1)
            return prob_profit > threshold
        except Exception as e:
            logger.error(f"MetaLabeler Predict Error: {e}")
            return False

    def _enrich(self, features: Dict[str, float], action: int) -> Dict[str, float]:
        """Creates interaction features to help the model find specific failure modes."""
        clean = features.copy()
        
        # 1. Action Encoding (Essential: Is it a Buy or Sell?)
        clean['primary_action'] = float(action)
        
        # 2. Interaction: Volatility context for the action
        # Does this strategy fail in high vol?
        vol = clean.get('volatility', 0.0)
        clean['vol_x_action'] = vol * action
        
        # 3. Interaction: Trend vs Action
        # Are we fighting the trend? (Hurst < 0.5 = Mean Revert, > 0.5 = Trend)
        hurst = clean.get('hurst', 0.5)
        clean['hurst_x_action'] = (hurst - 0.5) * action
        
        # 4. Interaction: VPIN (Toxicity) x Action
        vpin = clean.get('vpin', 0.5)
        clean['vpin_x_action'] = vpin * action
        
        return clean

    def _sanitize(self, features: Dict[str, float]) -> Dict[str, float]:
        clean = {}
        for k, v in features.items():
            if math.isfinite(v):
                clean[k] = float(v)
            else:
                clean[k] = 0.0
        return clean

# --- 5. ONLINE FEATURE ENGINEER (PHOENIX: STATIONARY + PHYSICS) ---
class OnlineFeatureEngineer:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.prices = deque(maxlen=window_size)
        self.volumes = deque(maxlen=window_size)
        self.returns = deque(maxlen=window_size)
        
        # Phase 2: Integrated Streaming Indicators (Uses RecursiveEMA)
        self.indicators = StreamingIndicators()
        
        # Legacy components (Retained for continuity)
        self.entropy = EntropyMonitor(window=window_size)
        self.vpin = VPINMonitor(bucket_size=1000)
        self.frac_diff = IncrementalFracDiff(d=0.3, window=window_size)
        self.vol_monitor = VolatilityMonitor(window=20)
        self.last_price = None
        
        # --- AUDIT FIX: STATIONARITY & CONTEXT ---
        # 1. Cumulative OFI Window
        self.ofi_window = deque(maxlen=20)
        
        # 2. Volatility Trend EMA (for breakout detection)
        self.atr_ema = RecursiveEMA(alpha=0.05)
        
        # 3. Long-term Volatility Baseline (for Vol Ratio)
        self.vol_baseline = RecursiveEMA(alpha=0.001)
        
        # 4. OFI Trend (New Feature: Flow Momentum)
        self.ofi_ema = RecursiveEMA(alpha=CONFIG['features'].get('ofi_alpha', 0.1))

    def update(self, price: float, timestamp: float, volume: float,
               high: Optional[float] = None, low: Optional[float] = None,
               buy_vol: float = 0.0, sell_vol: float = 0.0,
               time_feats: Dict[str, float] = None) -> Dict[str, float]:
        
        if time_feats is None:
            time_feats = {'sin_hour': 0.0, 'cos_hour': 0.0}
            
        # Fallback if high/low not provided
        if high is None: high = price
        if low is None: low = price

        # Validate Inputs (Nan/Inf Protection)
        if not math.isfinite(price) or price <= 0: return None
        if not math.isfinite(volume): volume = 0.0

        self.prices.append(price)
        self.volumes.append(volume)
        
        # 1. Log Returns (Stationary)
        # Replaces raw linear returns for ML stability
        ret_log = 0.0
        if self.last_price and self.last_price > 0:
            try:
                ret_log = math.log(price / self.last_price)
            except ValueError:
                ret_log = 0.0
            self.returns.append(ret_log)
        else:
            self.returns.append(0.0)

        # 2. Update Recursive Indicators (Phase 2 Core)
        tech_feats = self.indicators.update(price, high, low)
        current_atr = tech_feats.get('atr', 0.001)
        
        # 3. Update Legacy Metrics
        entropy_val = self.entropy.update(price)
        # PHOENIX UPGRADE: Pass explicit Buy/Sell volume for VPIN
        vpin_val = self.vpin.update(volume, price, buy_vol, sell_vol)
        fd_price = self.frac_diff.update(price)
        volatility_val = self.vol_monitor.update(ret_log)

        # 4. Volatility Ratio (Stationary)
        # Current Short-term Vol vs Long-term Baseline
        self.vol_baseline.update(volatility_val)
        baseline_vol = self.vol_baseline.get()
        vol_ratio = volatility_val / baseline_vol if baseline_vol > 1e-9 else 1.0

        # 5. Hurst Exponent (Market Memory)
        hurst_val = 0.5
        if len(self.returns) >= 20:
            ret_arr = np.array(list(self.returns), dtype=np.float64)
            hurst_val = calculate_hurst(ret_arr)

        # 6. Order Flow Imbalance (OFI)
        # Normalized by volume to ensure stationarity
        denominator = volume if volume > 0 else 1.0
        ofi_val = (buy_vol - sell_vol) / denominator
        
        # --- Context Features ---
        self.ofi_window.append(ofi_val)
        cum_ofi = sum(self.ofi_window) / len(self.ofi_window) if self.ofi_window else 0.0
        
        # OFI Trend (Momentum)
        self.ofi_ema.update(ofi_val)
        ofi_trend = self.ofi_ema.get()

        # Regime Detection (Trend vs Mean Reversion)
        regime_val = 1.0 if hurst_val > 0.55 else (-1.0 if hurst_val < 0.45 else 0.0)
        
        # Volatility Breakout (Expansion)
        self.atr_ema.update(current_atr)
        atr_trend = self.atr_ema.get()
        vol_breakout = 1.0 if current_atr > (atr_trend * 1.05) else 0.0

        # Efficiency Ratio (ER)
        er_val = 0.5
        if len(self.prices) >= 10:
            price_list = list(self.prices)
            changes = np.diff(price_list)
            abs_change_sum = np.sum(np.abs(changes))
            net_change = abs(price_list[-1] - price_list[0])
            er_val = net_change / abs_change_sum if abs_change_sum > 0 else 0.0

        # --- AUDIT FIX: CANDLE PHYSICS (Stationary) ---
        # Normalize candle features by range or previous close
        candle_range = max(high - low, 1e-9)
        prev_close = self.prices[-2] if len(self.prices) > 1 else price
        
        # Body Ratio (Body / Range)
        body_size = abs(price - prev_close)
        body_ratio = body_size / candle_range
        
        # Wick Ratios
        upper_wick = high - max(price, prev_close)
        lower_wick = min(price, prev_close) - low
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range

        # Normalized Oscillators
        # MACD is absolute price difference, normalize by price to make stationary
        macd_norm = tech_feats['macd_line'] / price
        rsi_norm = tech_feats['rsi'] / 100.0 # Scale 0-1

        self.last_price = price

        # --- LAG GENERATION (Memory) ---
        lagged_feats = {}
        r_list = list(self.returns)
        for i in range(1, 6): # Lag 1 to 5
            if len(r_list) > i:
                lagged_feats[f'ret_lag_{i}'] = r_list[-(i+1)]
            else:
                lagged_feats[f'ret_lag_{i}'] = 0.0

        # Construct Final Feature Vector
        raw_features = {
            # --- CRITICAL FIX: RESTORE RAW ATR/VOL FOR GATE & RISK ---
            'atr': current_atr,
            'volatility': volatility_val,
            # ---------------------------------------------------------
            
            # Stationary Technicals
            'rsi_norm': rsi_norm,
            'macd_norm': macd_norm,
            'macd_hist_norm': tech_feats['macd_hist'] / price,
            'atr_pct': current_atr / price, # ATR as % of price
            
            # Stationary Volatility
            'vol_ratio': vol_ratio,
            'volatility_log': math.log(volatility_val + 1e-9),
            'log_ret': ret_log,
            
            # Statistical / Microstructure
            'frac_diff': fd_price / price, # Normalize FracDiff price approx
            'entropy': entropy_val,
            'vpin': vpin_val,
            'hurst': hurst_val,
            'ofi': ofi_val,
            'efficiency_ratio': er_val,
            
            # Context
            'cum_ofi': cum_ofi,
            'ofi_trend': ofi_trend, # New
            'regime': regime_val,
            'vol_breakout': vol_breakout,
            
            # Candle Physics
            'body_ratio': body_ratio,
            'upper_wick_ratio': upper_wick_ratio,
            'lower_wick_ratio': lower_wick_ratio,
            
            # Original inputs (Normalized Z-Scores)
            # We calculate simple Z-scores on the fly for price/vol
            # This is locally stationary
            'volume_z': (volume - np.mean(self.volumes)) / (np.std(self.volumes) + 1e-9) if len(self.volumes) > 2 else 0.0,
            
            **lagged_feats,
            **time_feats
        }
        
        return self._sanitize_features(raw_features)

    def _sanitize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        clean = {}
        for k, v in features.items():
            if math.isfinite(v):
                clean[k] = float(v)
            else:
                clean[k] = 0.0
        return clean

# --- 6. LEGACY MONITORS (RETAINED) ---
class StreamingTripleBarrier:
    """
    Legacy barrier logic. Retained for backward compatibility if needed.
    """
    def __init__(self, vol_multiplier: float = 2.0, barrier_len: int = 50, horizon_ticks: int = 100):
        self.vol_multiplier = vol_multiplier
        self.horizon_ticks = horizon_ticks
        self.history = deque(maxlen=barrier_len)
        self.pending_events = {}

    def update(self, price: float, timestamp: float) -> List[Tuple[int, float]]:
        self.history.append(price)
        resolved = []
        to_remove = []
        
        for origin_ts, params in self.pending_events.items():
            if price >= params['top']:
                resolved.append((1, origin_ts))
                to_remove.append(origin_ts)
            elif price <= params['bot']:
                resolved.append((-1, origin_ts))
                to_remove.append(origin_ts)
            elif timestamp >= params['expiry']:
                resolved.append((0, origin_ts))
                to_remove.append(origin_ts)
        
        for ts in to_remove:
            del self.pending_events[ts]
            
        if len(self.history) >= 20:
            vol = np.std(list(self.history))
            if vol < 1e-9: vol = price * 0.001
            width = vol * self.vol_multiplier
            
            self.pending_events[timestamp] = {
                'entry': price,
                'top': price + width,
                'bot': price - width,
                'expiry': timestamp + (self.horizon_ticks * 60)
            }
        
        return resolved

class EntropyMonitor:
    """
    Calculates Shannon Entropy of price distribution.
    Phoenix Upgrade: Uses log-returns histogram for better stationarity.
    """
    def __init__(self, window: int = 50):
        self.buffer = deque(maxlen=window)
    
    def update(self, price: float) -> float:
        self.buffer.append(price)
        if len(self.buffer) < 20: return 0.5
        
        try:
            # PHOENIX UPGRADE: Use returns instead of raw prices for histogram
            prices = list(self.buffer)
            returns = np.diff(prices)
            if np.std(returns) < 1e-9: return 0.0
            
            hist, _ = np.histogram(returns, bins=10, density=True)
            ent = entropy(hist)
            if math.isnan(ent): return 0.5
            
            # Normalize to 0-1
            return ent / np.log(10)
        except Exception:
            return 0.5

class VPINMonitor:
    """
    Calculates Volume-Synchronized Probability of Informed Trading (VPIN).
    PHOENIX UPGRADE: Uses Bulk Volume Classification (BVC) approximation.
    High VPIN (>0.8) indicates Toxic Flow / Adverse Selection Risk.
    """
    def __init__(self, bucket_size: float = 1000):
        self.bucket_size = bucket_size
        
        # Accumulators
        self.current_bucket_vol = 0.0
        self.current_buy_vol = 0.0
        self.current_sell_vol = 0.0
        
        # Store buckets as (buy_vol, sell_vol)
        self.buckets = deque(maxlen=50) 
        self.last_price = None

    def update(self, volume: float, price: float, buy_vol_input: float = 0.0, sell_vol_input: float = 0.0) -> float:
        """
        Updates VPIN state.
        If buy_vol_input/sell_vol_input are provided (from Aggregator), uses them.
        Otherwise, infers using Tick Rule (BVC approximation).
        """
        if self.last_price is None:
            self.last_price = price
            return 0.5

        # Determine Flow Split
        b_vol = 0.0
        s_vol = 0.0

        if buy_vol_input > 0 or sell_vol_input > 0:
            b_vol = buy_vol_input
            s_vol = sell_vol_input
        else:
            # Fallback Tick Rule
            if price > self.last_price:
                b_vol = volume
            elif price < self.last_price:
                s_vol = volume
            else:
                # Neutral tick: 50/50 split
                b_vol = volume / 2
                s_vol = volume / 2
        
        self.last_price = price

        # Update Accumulators
        self.current_buy_vol += b_vol
        self.current_sell_vol += s_vol
        self.current_bucket_vol += volume
        
        # Bucket Management
        if self.current_bucket_vol >= self.bucket_size:
            # Commit Bucket
            self.buckets.append((self.current_buy_vol, self.current_sell_vol))
            
            # Reset
            self.current_bucket_vol = 0.0
            self.current_buy_vol = 0.0
            self.current_sell_vol = 0.0
        
        return self.get_vpin()

    def get_vpin(self) -> float:
        if not self.buckets: return 0.5
        
        total_vol = 0.0
        absolute_imbalance = 0.0
        
        for b_buy, b_sell in self.buckets:
            total_vol += (b_buy + b_sell)
            absolute_imbalance += abs(b_buy - b_sell)
        
        if total_vol < 1e-9: return 0.5
        
        return absolute_imbalance / total_vol

class IncrementalFracDiff:
    def __init__(self, d: float = 0.6, window: int = 20):
        self.d = d
        self.window = window
        self.weights = self._get_weights_floored(d, window)
        self.memory = deque(maxlen=window)

    def _get_weights_floored(self, d, size):
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (d - k + 1) / k
            w.append(w_k)
        return np.array(w[::-1])

    def update(self, price: float) -> float:
        self.memory.append(price)
        if len(self.memory) < self.window: return 0.0
        series = np.array(self.memory)
        return float(np.dot(self.weights, series))

class VolatilityMonitor:
    def __init__(self, window: int = 20):
        self.returns = deque(maxlen=window)
    
    def update(self, ret: float) -> float:
        self.returns.append(ret)
        if len(self.returns) < 5: return 0.001
        val = np.std(self.returns)
        return val if math.isfinite(val) else 0.001

@njit
def calculate_hurst(ts):
    n = len(ts)
    if n < 20: return 0.5
    if np.std(ts) < 1e-9: return 0.5
    
    lags = np.arange(2, 20)
    tau = np.zeros(len(lags))
    for i in range(len(lags)):
        lag = lags[i]
        diff = ts[lag:] - ts[:-lag]
        std_diff = np.std(diff)
        if std_diff < 1e-9:
            tau[i] = 1e-9
        else:
            tau[i] = std_diff
    
    x = np.log(lags.astype(np.float64))
    y = np.log(tau)
    A = np.column_stack((x, np.ones(len(x))))
    m, c = np.linalg.lstsq(A, y)[0]
    return m

def enrich_with_d1_data(features: Dict[str, float], d1_data: Dict[str, float], current_price: float) -> Dict[str, float]:
    if not d1_data: return features
    
    prev_high = d1_data.get('high', 0)
    prev_low = d1_data.get('low', 0)
    
    if prev_high == 0: return features
    
    # Normalize distances by price to ensure stationarity
    features['dist_d1_high'] = (prev_high - current_price) / current_price
    features['dist_d1_low'] = (current_price - prev_low) / current_price
    
    return features