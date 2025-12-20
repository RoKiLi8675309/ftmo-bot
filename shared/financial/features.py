# FILENAME: shared/financial/features.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/financial/features.py
# DEPENDENCIES: shared, numpy, numba, scipy, river (optional)
# DESCRIPTION: Mathematical kernels for Feature Engineering, Labeling, and Risk metrics.
# AUDIT REMEDIATION (GROK):
#   - ADDED: MetaLabeler class for profitability filtering.
#   - FIXED: Robust OFI and NaN sanitization.
# CRITICAL: Python 3.9 Compatible. Graceful degradation if ML libs missing.
# =============================================================================
from __future__ import annotations
import math
import logging
import sys
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List, Tuple

# Numba for high-performance JIT compilation of Hurst exponent
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):
        return func

# Scipy for Entropy
try:
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# River / Sklearn imports
try:
    from river import stats, utils, linear_model
    from sklearn.isotonic import IsotonicRegression
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger("Features")

# --- 1. PROBABILITY CALIBRATOR ---
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


# --- 2. META LABELER (GROK REMEDIATION) ---
class MetaLabeler:
    """
    Secondary model that learns whether a primary signal (Buy/Sell) actually
    resulted in profit after costs. Acts as a gatekeeper.
    """
    def __init__(self):
        self.model = None
        self.buffer = deque(maxlen=1000)
        
        if ML_AVAILABLE:
            # Simple Logistic Regression for binary profitability check
            # Input: Features + Primary Action. Output: 1 (Profitable) / 0 (Loss)
            self.model = linear_model.LogisticRegression()

    def update(self, features: Dict[str, float], primary_action: int, outcome_pnl: float):
        """
        Train the meta-model.
        primary_action: 1 (Buy), -1 (Sell), 0 (Hold)
        outcome_pnl: Realized PnL (Net of costs)
        """
        if not ML_AVAILABLE or primary_action == 0:
            return

        # Target: 1 if profitable, 0 if loss
        y_meta = 1 if outcome_pnl > 0 else 0
        
        try:
            # Inject primary action as a feature for the meta learner
            augmented_features = features.copy()
            augmented_features['primary_action'] = float(primary_action)
            
            clean_features = self._sanitize(augmented_features)
            self.model.learn_one(clean_features, y_meta)
            self.buffer.append((clean_features, y_meta))
        except Exception as e:
            logger.error(f"MetaLabeler Update Error: {e}")

    def predict(self, features: Dict[str, float], primary_action: int, threshold: float = 0.55) -> bool:
        """
        Returns True if the trade is likely to be profitable.
        """
        if not ML_AVAILABLE or primary_action == 0:
            return False # SAFE DEFAULT: Do not trade if ML is broken
            
        try:
            augmented_features = features.copy()
            augmented_features['primary_action'] = float(primary_action)
            clean_features = self._sanitize(augmented_features)
            
            # predict_proba_one returns {0: prob_loss, 1: prob_profit}
            probs = self.model.predict_proba_one(clean_features)
            prob_profit = probs.get(1, 0.0)
            
            return prob_profit > threshold
        except Exception as e:
            logger.error(f"MetaLabeler Predict Error: {e}")
            return False # Block on error

    def _sanitize(self, features: Dict[str, float]) -> Dict[str, float]:
        clean = {}
        for k, v in features.items():
            if math.isfinite(v):
                clean[k] = v
            else:
                clean[k] = 0.0
        return clean


# --- 3. ONLINE FEATURE ENGINEER ---
class OnlineFeatureEngineer:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.prices = deque(maxlen=window_size)
        self.volumes = deque(maxlen=window_size)
        self.returns = deque(maxlen=window_size)
        self.entropy = EntropyMonitor(window=window_size)
        self.vpin = VPINMonitor(bucket_size=1000)
        self.frac_diff = IncrementalFracDiff(d=0.6, window=window_size)
        self.vol_monitor = VolatilityMonitor(window=20)
        self.last_price = None

    def update(self, price: float, timestamp: float, volume: float, 
               buy_vol: float = 0.0, sell_vol: float = 0.0, 
               time_feats: Dict[str, float] = None) -> Dict[str, float]:
        
        if time_feats is None:
            time_feats = {'sin_hour': 0.0, 'cos_hour': 0.0}

        self.prices.append(price)
        self.volumes.append(volume)
        if self.last_price and self.last_price > 0:
            ret = math.log(price / self.last_price) if price > 0 else 0.0
            self.returns.append(ret)
        else:
            ret = 0.0
            self.returns.append(0.0)

        entropy_val = self.entropy.update(price)
        vpin_val = self.vpin.update(volume, price, buy_vol, sell_vol)
        fd_price = self.frac_diff.update(price)
        volatility_val = self.vol_monitor.update(ret)

        hurst_val = 0.5
        if len(self.returns) >= 20:
            ret_arr = np.array(list(self.returns), dtype=np.float64)
            hurst_val = calculate_hurst(ret_arr)

        denominator = volume if volume > 0 else 1.0
        ofi_val = (buy_vol - sell_vol) / denominator

        er_val = 0.5
        if len(self.prices) >= 10:
            price_list = list(self.prices)
            changes = np.diff(price_list)
            abs_change_sum = np.sum(np.abs(changes))
            net_change = abs(price_list[-1] - price_list[0])
            er_val = net_change / abs_change_sum if abs_change_sum > 0 else 0.0

        z_score_val = 0.0
        if len(self.prices) >= 20:
            mu = np.mean(self.prices)
            sigma = np.std(self.prices)
            if sigma > 1e-9:
                z_score_val = (price - mu) / sigma

        price_z = 0.0
        volume_z = 0.0
        if len(self.prices) > 1:
            price_mean = np.mean(self.prices)
            price_std = np.std(self.prices)
            price_z = (price - price_mean) / price_std if price_std > 1e-9 else 0.0
            price_z = np.clip(price_z, -3, 3)
            
            volume_mean = np.mean(self.volumes)
            volume_std = np.std(self.volumes)
            volume_z = (volume - volume_mean) / volume_std if volume_std > 1e-9 else 0.0
            volume_z = np.clip(volume_z, -3, 3)

        self.last_price = price

        raw_features = {
            'frac_diff': fd_price,
            'volatility': volatility_val,
            'entropy': entropy_val,
            'vpin': vpin_val,
            'hurst': hurst_val,
            'ofi': ofi_val,
            'efficiency_ratio': er_val,
            'z_score': z_score_val,
            'price_z': price_z,
            'volume_z': volume_z,
            'price_raw': price,
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


# --- 4. STREAMING TRIPLE BARRIER ---
class StreamingTripleBarrier:
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


# --- 5. ENTROPY MONITOR ---
class EntropyMonitor:
    def __init__(self, window: int = 50):
        self.buffer = deque(maxlen=window)
    
    def update(self, x: float) -> float:
        self.buffer.append(x)
        if len(self.buffer) < 10: return 0.5
        if np.std(list(self.buffer)) < 1e-9: return 0.0
        try:
            hist, _ = np.histogram(self.buffer, bins=10, density=True)
            ent = entropy(hist)
            if math.isnan(ent): return 0.5
            return ent / np.log(10)
        except Exception:
            return 0.5


# --- 6. VPIN MONITOR ---
class VPINMonitor:
    def __init__(self, bucket_size: float = 1000):
        self.bucket_size = bucket_size
        self.current_bucket_vol = 0.0
        self.buy_vol = 0.0
        self.sell_vol = 0.0
        self.buckets = deque(maxlen=50)
        self.last_price = 0.0

    def update(self, volume: float, price: float, buy_vol_input: float = 0.0, sell_vol_input: float = 0.0) -> float:
        if buy_vol_input > 0 or sell_vol_input > 0:
            self.buy_vol += buy_vol_input
            self.sell_vol += sell_vol_input
        else:
            if price > self.last_price:
                self.buy_vol += volume
            elif price < self.last_price:
                self.sell_vol += volume
            else:
                self.buy_vol += volume / 2
                self.sell_vol += volume / 2
        
        self.current_bucket_vol += volume
        self.last_price = price

        if self.current_bucket_vol >= self.bucket_size:
            self.buckets.append((self.buy_vol, self.sell_vol))
            self.current_bucket_vol = 0.0
            self.buy_vol = 0.0
            self.sell_vol = 0.0
        
        if not self.buckets: return 0.5

        total_vol = 0.0
        diff_vol = 0.0
        for b_buy, b_sell in self.buckets:
            total_vol += (b_buy + b_sell)
            diff_vol += abs(b_buy - b_sell)
        
        if total_vol < 1e-9: return 0.5
        return diff_vol / total_vol


# --- 7. INCREMENTAL FRAC DIFF ---
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
        return np.dot(self.weights, series)


# --- 8. VOLATILITY MONITOR ---
class VolatilityMonitor:
    def __init__(self, window: int = 20):
        self.returns = deque(maxlen=window)
    
    def update(self, ret: float) -> float:
        self.returns.append(ret)
        if len(self.returns) < 5: return 0.001
        val = np.std(self.returns)
        return val if math.isfinite(val) else 0.001


# --- 9. UTILS & JIT FUNCTIONS ---
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
    features['dist_d1_high'] = (prev_high - current_price) / current_price
    features['dist_d1_low'] = (current_price - prev_low) / current_price
    return features