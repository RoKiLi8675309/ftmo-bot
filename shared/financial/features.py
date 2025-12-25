# =============================================================================
# FILENAME: shared/financial/features.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/financial/features.py
# DEPENDENCIES: shared, numpy, numba, scipy, river (optional), hmmlearn
# DESCRIPTION: Mathematical kernels for Feature Engineering, Labeling, and Risk.
#
# PHOENIX STRATEGY UPGRADE (2025-12-24 - GOLDEN CONFIG COMPLIANCE):
# 1. SCALING: Implemented WelfordScaler for O(1) online Z-score calculation.
# 2. REGIME: KER & FDI optimized for 1.45-1.55 Inhibition Zone detection.
# 3. STATIONARITY: Volume Z-Score now uses Welford's Algorithm (Infinite Window).
# 4. ROBUSTNESS: HMM and Entropy monitors hardened against sparse data.
# =============================================================================
from __future__ import annotations
import math
import logging
import sys
import os
import warnings
import contextlib
import io
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

# HMM Learn (Guarded)
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

# River / Sklearn imports (Guarded for Windows Producer compatibility)
try:
    from river import linear_model, forest, metrics
    from sklearn.isotonic import IsotonicRegression
    from sklearn.exceptions import ConvergenceWarning
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger("Features")

# --- 0. HELPER MATH KERNELS (ROBUST) ---

class WelfordScaler:
    """
    Online Standardization using Welford's Algorithm.
    Computes running Mean and Variance in a single pass (O(1)).
    Used to normalize non-stationary features (e.g., Volume) dynamically
    without the memory cost or lag of a sliding window.
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squares of differences from the current mean

    def update(self, x: float) -> float:
        """Updates statistics and returns the Z-Score of the new value."""
        if x is None or not math.isfinite(x):
            return 0.0
            
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
        if self.n < 2:
            return 0.0
            
        # Variance = M2 / (n - 1)
        variance = self.M2 / (self.n - 1)
        stdev = math.sqrt(variance)
        
        if stdev < 1e-9:
            return 0.0
            
        return (x - self.mean) / stdev

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

# --- 1. REGIME INDICATORS (NEW) ---
class KaufmanEfficiencyRatio:
    """
    Quantifies trend efficiency (Signal vs Noise).
    Range: 0.0 (Pure Noise) to 1.0 (Perfect Efficiency).
    Formula: |Net Change| / Sum(|Individual Changes|)
    """
    def __init__(self, window: int = 10):
        self.window = window
        self.prices = deque(maxlen=window + 1)

    def update(self, price: float) -> float:
        self.prices.append(price)
        if len(self.prices) < self.window + 1:
            return 0.5 # Default to neutral/uncertain
        
        # Signal: Absolute difference between price now and n periods ago
        signal = abs(self.prices[-1] - self.prices[0])
        
        # Noise: Sum of absolute differences between consecutive bars
        arr = np.array(self.prices)
        noise = np.sum(np.abs(np.diff(arr)))
        
        if noise == 0:
            return 1.0 # Pure efficiency
            
        return signal / noise

class FractalDimensionIndex:
    """
    Measures market complexity/dimensionality based on Chaos Theory.
    Range: ~1.0 (Linear/Trend) to ~2.0 (Jagged/Mean Reversion).
    CRITICAL: Values between 1.45 and 1.55 indicate Random Walk (Inhibition Zone).
    """
    def __init__(self, window: int = 30):
        self.window = window
        self.prices = deque(maxlen=window)

    def update(self, price: float) -> float:
        self.prices.append(price)
        if len(self.prices) < self.window:
            return 1.5 # Default to Random Walk barrier
        
        data = np.array(self.prices)
        
        # 1. Rescale data to unit square [0,1]x[0,1]
        min_p, max_p = np.min(data), np.max(data)
        if max_p == min_p:
            return 1.0 # Flat line = 1D
            
        scaled = (data - min_p) / (max_p - min_p)
        
        # 2. Calculate Path Length (L) in normalized space
        dt = 1.0 / (self.window - 1)
        diffs = np.diff(scaled)
        
        # Pythagorean theorem for each segment length
        length = np.sum(np.sqrt(diffs**2 + dt**2))
        
        if length <= 0: return 1.5
        
        # 3. Calculate FDI
        # Formula: FDI = 1 + (log(L) + log(2)) / log(2*(N-1))
        try:
            numerator = np.log(length) + np.log(2)
            denominator = np.log(2 * (self.window - 1))
            if denominator == 0: return 1.5
            
            fdi = 1 + (numerator / denominator)
            return fdi
        except Exception:
            return 1.5

class RegimeDetector:
    """
    Online Hidden Markov Model (HMM) for market regime classification.
    Identifies latent states (e.g., Low Vol/Range, High Vol/Trend).
    """
    def __init__(self, n_states: int = 2, window: int = 100):
        self.n_states = n_states
        self.window = window
        self.returns = deque(maxlen=window)
        self.last_regime = 0
        self.model = None
        self.fit_failures = 0
        self.fit_counter = 0 
        
        if HMM_AVAILABLE:
            try:
                self.model = hmm.GaussianHMM(
                    n_components=n_states, 
                    covariance_type="diag", 
                    n_iter=100, 
                    random_state=42,
                    init_params="stmc", 
                    verbose=False,
                    min_covar=1e-3, 
                    tol=1e-4
                )
            except Exception as e:
                logger.error(f"HMM Init Failed: {e}")
                self.model = None
    
    def update(self, ret: float) -> int:
        self.returns.append(ret * 100.0)
        self.fit_counter += 1
        
        if not HMM_AVAILABLE or self.model is None:
            return 0
            
        if len(self.returns) < self.window:
            return self.last_regime
            
        try:
            data = np.array(self.returns).reshape(-1, 1)
            
            # Optimization: Fit periodically
            should_fit = (self.fit_counter % 50 == 0) or (self.fit_counter < 200)

            if should_fit:
                if np.var(data) < 1e-6 or np.isnan(data).any():
                    return self.last_regime

                if self.fit_failures > 5:
                    self.model.init_params = "stmc"
                    self.fit_failures = 0
                elif hasattr(self.model, 'startprob_'):
                    self.model.init_params = ""

                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        self.model.fit(data)
                
                if not self.model.monitor_.converged:
                    self.fit_failures += 1
                else:
                    self.fit_failures = 0

            current_state = int(self.model.predict(data[-1].reshape(1, -1))[0])
            self.last_regime = current_state
            return current_state

        except Exception:
            self.fit_failures += 1
            return self.last_regime

# --- 2. STREAMING INDICATORS ---
class StreamingIndicators:
    """
    Recursive implementation of technical indicators.
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
        
        # ATR Components
        self.atr_mean = RecursiveEMA(alpha=1 / atr_period)
        self.prev_close = None

    def update(self, price: float, high: float, low: float) -> Dict[str, float]:
        features = {}
        
        # 1. MACD
        self.ema_fast.update(price)
        self.ema_slow.update(price)
        self.macd_line = self.ema_fast.get() - self.ema_slow.get()
        self.macd_signal.update(self.macd_line)
        histogram = self.macd_line - self.macd_signal.get()
        
        features['macd_line'] = self.macd_line
        features['macd_hist'] = histogram
        
        # 2. RSI
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
            
        # 3. ATR
        if self.prev_close is not None:
            tr1 = high - low
            tr2 = abs(high - self.prev_close)
            tr3 = abs(low - self.prev_close)
            true_range = max(tr1, tr2, tr3)
            self.atr_mean.update(true_range)
            features['atr'] = self.atr_mean.get()
        else:
            features['atr'] = high - low if (high > 0 and low > 0 and high != low) else 0.001
            
        self.prev_price = price
        self.prev_close = price
        return features

# --- 3. ADAPTIVE TRIPLE BARRIER ---
class AdaptiveTripleBarrier:
    """
    Volatility-Adaptive Labeling.
    Barriers expand/contract based on ATR and Volatility Regime.
    """
    def __init__(self, horizon_ticks: int = 12, risk_mult: float = 1.0, reward_mult: float = 2.0, drift_threshold: float = 0.75):
        self.buffer = deque()
        self.time_limit = horizon_ticks
        self.risk_mult = risk_mult
        self.reward_mult = reward_mult
        self.drift_threshold = drift_threshold

    def add_trade_opportunity(self, features: Dict[str, float], entry_price: float, current_atr: float, timestamp: float):
        if current_atr <= 0: current_atr = entry_price * 0.0001
        
        volatility = features.get('volatility', 0.0)
        # Adaptive Scaling: 1 + (Vol * 100) to breathe with market
        adaptive_scalar = 1.0 + (volatility * 100.0)
        
        effective_risk_mult = self.risk_mult * adaptive_scalar
        effective_reward_mult = self.reward_mult * adaptive_scalar
        
        upper_barrier = entry_price + (effective_reward_mult * current_atr)
        lower_barrier = entry_price - (effective_risk_mult * current_atr)
        
        self.buffer.append({
            'features': features,
            'entry': entry_price,
            'tp': upper_barrier,
            'sl': lower_barrier,
            'atr': current_atr,
            'start_time': timestamp,
            'age': 0
        })

    def resolve_labels(self, current_high: float, current_low: float, current_close: float = None) -> List[Tuple[Dict[str, float], int, float]]:
        resolved = []
        active = deque()
        
        if current_close is None:
            current_close = (current_high + current_low) / 2.0

        while self.buffer:
            trade = self.buffer.popleft()
            trade['age'] += 1
            
            label = None
            realized_ret = 0.0
            
            if current_high >= trade['tp']:
                label = 1 # BUY
                realized_ret = (trade['tp'] - trade['entry']) / trade['entry']
            elif current_low <= trade['sl']:
                label = -1 # SELL
                realized_ret = (trade['entry'] - trade['sl']) / trade['entry']
            elif trade['age'] >= self.time_limit:
                # Drift Check
                drift = current_close - trade['entry']
                drift_req = trade['atr'] * self.drift_threshold
                
                if drift > drift_req:
                    label = 1 # Soft BUY
                    realized_ret = (current_close - trade['entry']) / trade['entry']
                elif drift < -drift_req:
                    label = -1 # Soft SELL
                    realized_ret = (trade['entry'] - current_close) / trade['entry']
                else:
                    label = 0 # Noise
                    realized_ret = 0.0

            if label is not None:
                resolved.append((trade['features'], label, realized_ret))
            else:
                active.append(trade)
        
        self.buffer = active
        return resolved

# --- 4. PROBABILITY CALIBRATOR ---
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

# --- 5. META LABELER ---
class MetaLabeler:
    def __init__(self):
        self.model = None
        self.buffer = deque(maxlen=1000)
        
        if ML_AVAILABLE:
            self.model = forest.ARFClassifier(
                n_models=10, 
                seed=42, 
                metric=metrics.F1()
            )

    def update(self, features: Dict[str, float], primary_action: int, outcome_pnl: float):
        if not ML_AVAILABLE or primary_action == 0:
            return
        
        y_meta = 1 if outcome_pnl > 0 else 0
        try:
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
            return prob_profit > threshold
        except Exception as e:
            logger.error(f"MetaLabeler Predict Error: {e}")
            return False

    def _enrich(self, features: Dict[str, float], action: int) -> Dict[str, float]:
        clean = features.copy()
        clean['primary_action'] = float(action)
        clean['vol_x_action'] = clean.get('volatility', 0.0) * action
        clean['hurst_x_action'] = (clean.get('hurst', 0.5) - 0.5) * action
        clean['vpin_x_action'] = clean.get('vpin', 0.5) * action
        clean['ker_x_action'] = clean.get('ker', 0.5) * action
        return clean

    def _sanitize(self, features: Dict[str, float]) -> Dict[str, float]:
        clean = {}
        for k, v in features.items():
            if math.isfinite(v):
                clean[k] = float(v)
            else:
                clean[k] = 0.0
        return clean

# --- 6. ONLINE FEATURE ENGINEER ---
class OnlineFeatureEngineer:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.prices = deque(maxlen=window_size)
        self.volumes = deque(maxlen=window_size)
        self.returns = deque(maxlen=window_size)
        
        self.indicators = StreamingIndicators()
        
        # Welford Scaler for Volume (Infinite Window Stationarity)
        self.welford_volume = WelfordScaler()
        
        self.ker = KaufmanEfficiencyRatio(window=10)
        self.fdi = FractalDimensionIndex(window=30)
        self.regime_detector = RegimeDetector(n_states=2, window=100)
        
        self.entropy = EntropyMonitor(window=window_size)
        self.vpin = VPINMonitor(bucket_size=1000)
        self.frac_diff = IncrementalFracDiff(d=0.3, window=window_size)
        self.vol_monitor = VolatilityMonitor(window=20)
        self.last_price = None
        
        self.ofi_window = deque(maxlen=20)
        self.atr_ema = RecursiveEMA(alpha=0.05)
        self.vol_baseline = RecursiveEMA(alpha=0.001)
        self.ofi_ema = RecursiveEMA(alpha=CONFIG['features'].get('ofi_alpha', 0.1))

    def update(self, price: float, timestamp: float, volume: float, 
               high: Optional[float] = None, low: Optional[float] = None, 
               buy_vol: float = 0.0, sell_vol: float = 0.0, 
               time_feats: Dict[str, float] = None,
               sentiment: float = 0.0) -> Dict[str, float]:
        
        if time_feats is None:
            time_feats = {'sin_hour': 0.0, 'cos_hour': 0.0}
        if high is None: high = price
        if low is None: low = price

        if not math.isfinite(price) or price <= 0: return None
        if not math.isfinite(volume): volume = 0.0

        self.prices.append(price)
        self.volumes.append(volume)
        
        # Log Returns
        ret_log = 0.0
        if self.last_price and self.last_price > 0:
            try:
                ret_log = math.log(price / self.last_price)
            except ValueError:
                ret_log = 0.0
            self.returns.append(ret_log)
        else:
            self.returns.append(0.0)

        # Update Indicators
        tech_feats = self.indicators.update(price, high, low)
        current_atr = tech_feats.get('atr', 0.001)
        
        # Update Regime
        ker_val = self.ker.update(price)
        fdi_val = self.fdi.update(price)
        regime_hmm = self.regime_detector.update(ret_log)
        
        # Update Other Metrics
        entropy_val = self.entropy.update(price)
        vpin_val = self.vpin.update(volume, price, buy_vol, sell_vol)
        fd_price = self.frac_diff.update(price)
        volatility_val = self.vol_monitor.update(ret_log)

        # Volatility Ratio
        self.vol_baseline.update(volatility_val)
        baseline_vol = self.vol_baseline.get()
        vol_ratio = volatility_val / baseline_vol if baseline_vol > 1e-9 else 1.0

        # Hurst
        hurst_val = 0.5
        if len(self.returns) >= 20:
            ret_arr = np.array(list(self.returns), dtype=np.float64)
            hurst_val = calculate_hurst(ret_arr)

        # OFI
        denominator = volume if volume > 0 else 1.0
        ofi_val = (buy_vol - sell_vol) / denominator
        self.ofi_window.append(ofi_val)
        cum_ofi = sum(self.ofi_window) / len(self.ofi_window) if self.ofi_window else 0.0
        self.ofi_ema.update(ofi_val)
        ofi_trend = self.ofi_ema.get()

        # Regime / Trends
        regime_val = 1.0 if hurst_val > 0.55 else (-1.0 if hurst_val < 0.45 else 0.0)
        self.atr_ema.update(current_atr)
        atr_trend = self.atr_ema.get()
        vol_breakout = 1.0 if current_atr > (atr_trend * 1.05) else 0.0

        # ER
        er_val = 0.5
        if len(self.prices) >= 10:
            price_list = list(self.prices)
            changes = np.diff(price_list)
            abs_change_sum = np.sum(np.abs(changes))
            net_change = abs(price_list[-1] - price_list[0])
            er_val = net_change / abs_change_sum if abs_change_sum > 0 else 0.0

        # Candle Physics
        candle_range = max(high - low, 1e-9)
        prev_close = self.prices[-2] if len(self.prices) > 1 else price
        body_size = abs(price - prev_close)
        body_ratio = body_size / candle_range
        upper_wick = high - max(price, prev_close)
        lower_wick = min(price, prev_close) - low
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range

        # Normalized Oscillators
        macd_norm = tech_feats['macd_line'] / price
        rsi_norm = tech_feats['rsi'] / 100.0

        self.last_price = price
        
        # --- WELFORD SCALING (REPLACED MANUAL Z-SCORE) ---
        volume_z = self.welford_volume.update(volume)
        # -------------------------------------------------

        # Lag Features
        lagged_feats = {}
        r_list = list(self.returns)
        for i in range(1, 6):
            if len(r_list) > i:
                lagged_feats[f'ret_lag_{i}'] = r_list[-(i+1)]
            else:
                lagged_feats[f'ret_lag_{i}'] = 0.0

        raw_features = {
            'atr': current_atr,
            'volatility': volatility_val,
            
            # Regime
            'ker': ker_val,
            'fdi': fdi_val,
            'hmm_regime': float(regime_hmm) / (self.regime_detector.n_states - 1) if self.regime_detector.n_states > 1 else 0.0,
            'sentiment': sentiment,
            
            # Technicals (Stationary)
            'rsi_norm': rsi_norm,
            'macd_norm': macd_norm,
            'macd_hist_norm': tech_feats['macd_hist'] / price,
            'atr_pct': current_atr / price, 
            
            # Volatility
            'vol_ratio': vol_ratio,
            'volatility_log': math.log(volatility_val + 1e-9),
            'log_ret': ret_log,
            
            # Microstructure
            'frac_diff': fd_price / price, 
            'entropy': entropy_val,
            'vpin': vpin_val,
            'hurst': hurst_val,
            'ofi': ofi_val,
            'efficiency_ratio': er_val,
            
            # Context
            'cum_ofi': cum_ofi,
            'ofi_trend': ofi_trend,
            'regime': regime_val,
            'vol_breakout': vol_breakout,
            
            # Physics
            'body_ratio': body_ratio,
            'upper_wick_ratio': upper_wick_ratio,
            'lower_wick_ratio': lower_wick_ratio,
            
            # Scaled Volume
            'volume_z': volume_z,
            
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

# --- 7. LEGACY MONITORS ---
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

class EntropyMonitor:
    def __init__(self, window: int = 50):
        self.buffer = deque(maxlen=window)
    
    def update(self, price: float) -> float:
        self.buffer.append(price)
        if len(self.buffer) < 20: return 0.5
        try:
            prices = list(self.buffer)
            returns = np.diff(prices)
            if np.std(returns) < 1e-9: return 0.0
            hist, _ = np.histogram(returns, bins=10, density=True)
            ent = entropy(hist)
            if math.isnan(ent): return 0.5
            return ent / np.log(10)
        except Exception:
            return 0.5

class VPINMonitor:
    def __init__(self, bucket_size: float = 1000):
        self.bucket_size = bucket_size
        self.current_bucket_vol = 0.0
        self.current_buy_vol = 0.0
        self.current_sell_vol = 0.0
        self.buckets = deque(maxlen=50) 
        self.last_price = None

    def update(self, volume: float, price: float, buy_vol_input: float = 0.0, sell_vol_input: float = 0.0) -> float:
        if self.last_price is None:
            self.last_price = price
            return 0.5
        
        b_vol = 0.0
        s_vol = 0.0
        if buy_vol_input > 0 or sell_vol_input > 0:
            b_vol = buy_vol_input
            s_vol = sell_vol_input
        else:
            if price > self.last_price:
                b_vol = volume
            elif price < self.last_price:
                s_vol = volume
            else:
                b_vol = volume / 2
                s_vol = volume / 2
        
        self.last_price = price
        self.current_buy_vol += b_vol
        self.current_sell_vol += s_vol
        self.current_bucket_vol += volume
        
        if self.current_bucket_vol >= self.bucket_size:
            self.buckets.append((self.current_buy_vol, self.current_sell_vol))
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
    hurst = m * 2.0 
    return max(0.0, min(1.0, hurst))

def enrich_with_d1_data(features: Dict[str, float], d1_data: Dict[str, float], current_price: float) -> Dict[str, float]:
    if not d1_data: return features
    
    prev_high = d1_data.get('high', 0)
    prev_low = d1_data.get('low', 0)
    
    if prev_high == 0: return features
    
    # Normalize distances by price to ensure stationarity
    features['dist_d1_high'] = (prev_high - current_price) / current_price
    features['dist_d1_low'] = (current_price - prev_low) / current_price
    
    return features