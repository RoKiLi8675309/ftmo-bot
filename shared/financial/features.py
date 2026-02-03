# =============================================================================
# FILENAME: shared/financial/features.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/financial/features.py
# DEPENDENCIES: shared, numpy, numba, scipy, river (optional), hmmlearn
# DESCRIPTION: Mathematical kernels for Feature Engineering, Labeling, and Risk.
# 
# PHOENIX V16.20 AUDIT FIX (THE MATH BREAKER CURE):
# 1. NUMBA STABILITY: Replaced 'np.linalg.lstsq' with explicit linear regression
#    in 'calculate_hurst'. This fixes the "Tuple vs Array" return crash across 
#    different Numpy versions.
# 2. TYPE SAFETY: Enforced float64 casting in JIT blocks.
# 3. ROBUSTNESS: Preserved HMM variance checks and zero-division guards.
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
    Formula: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
    """
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = None

    def update(self, x: float):
        if x is None or math.isnan(x) or math.isinf(x):
            return  # Skip bad values
            
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
    
    def get(self) -> float:
        return self.value if self.value is not None else 0.0

# --- 1. PROJECT PHOENIX V9.0 MOMENTUM INDICATORS ---

class StreamingBollingerBands:
    """
    V12.4 UPDATE: Sniper Mode Breakout Indicator.
    Calculates Upper/Lower Bands and Width for Breakout & Squeeze Detection.
    Defaults adjusted to 1.5 Std Dev for Aggressor entries.
    """
    def __init__(self, window: int = 20, num_std: float = 1.5):
        self.window = window
        self.num_std = num_std
        self.buffer = deque(maxlen=window)
        
    def update(self, price: float) -> Dict[str, float]:
        self.buffer.append(price)
        
        if len(self.buffer) < self.window:
            return {
                'bb_upper': price, 
                'bb_lower': price, 
                'bb_mid': price, 
                'bb_width': 0.0, 
                'bb_pct_b': 0.5,
                'bb_breakout': 0.0
            }
            
        # Calculate Stats
        mu = np.mean(self.buffer)
        std = np.std(self.buffer)
        
        upper = mu + (self.num_std * std)
        lower = mu - (self.num_std * std)
        width = (upper - lower) / mu if mu > 1e-9 else 0.0
        
        # Percent B: Where is price relative to bands? (0=Lower, 1=Upper)
        band_range = upper - lower
        pct_b = (price - lower) / band_range if band_range > 1e-9 else 0.5
        
        # Breakout Signal: >0 if above upper, <0 if below lower
        breakout = 0.0
        if price > upper:
            breakout = (price - upper) / price
        elif price < lower:
            breakout = (price - lower) / price # Negative value
            
        return {
            'bb_upper': upper,
            'bb_lower': lower,
            'bb_mid': mu,
            'bb_width': width,
            'bb_pct_b': pct_b,
            'bb_breakout': breakout
        }

class StreamingParkinsonVolatility:
    """
    Estimates volatility using High/Low range.
    More efficient than Close-to-Close volatility for detecting expansion.
    """
    def __init__(self, alpha: float = 0.1):
        self.factor = 1.0 / (4.0 * math.log(2.0))
        self.ema = RecursiveEMA(alpha)

    def update(self, high: float, low: float) -> float:
        if low <= 0 or high <= 0 or low > high:
            return 0.0
        
        try:
            # Raw Parkinson Variance for this bar
            # Guard against flat bar log(1) = 0
            if abs(high - low) < 1e-9: return self.ema.get()

            log_hl = math.log(high / low)
            variance = self.factor * (log_hl ** 2)
            vol = math.sqrt(variance)
            
            # Smooth it for stability
            self.ema.update(vol)
            return self.ema.get()
        except ValueError:
            return 0.0

class StreamingAmihudLiquidity:
    """
    Proxy for Illiquidity using L1 data.
    Ratio of absolute return to dollar volume.
    """
    def __init__(self, alpha: float = 0.05):
        self.ema = RecursiveEMA(alpha)

    def update(self, abs_return: float, price: float, volume: float) -> float:
        if volume <= 0 or price <= 0:
            return 0.0
            
        dollar_vol = price * volume
        if dollar_vol < 1.0: 
            return 0.0
            
        illiquidity = abs_return / dollar_vol
        
        # Scale up for feature stability
        scaled_illiquidity = illiquidity * 1e6
        
        self.ema.update(scaled_illiquidity)
        return self.ema.get()

class StreamingRelativeVolume:
    """
    Measures Duration Intensity for Volume Bars.
    Since volume is constant in Volume Bars, we measure Time to Fill.
    Intensity = AvgDuration / CurrentDuration.
    High Intensity (> 2.0) = FUEL GAUGE for Aggressor Strategy.
    """
    def __init__(self, window: int = 20):
        self.dur_ema = RecursiveEMA(alpha=2.0/(window+1))
        self.last_ts = None

    def update(self, timestamp: float) -> float:
        if self.last_ts is None:
            self.last_ts = timestamp
            return 1.0 # First bar baseline
            
        duration = timestamp - self.last_ts
        self.last_ts = timestamp
        
        if duration <= 1e-3: duration = 1e-3 # Prevent div by zero (instant fill)
        
        avg_dur = self.dur_ema.get()
        
        # Initialize EMA if first real update
        if avg_dur == 0.0:
            self.dur_ema.update(duration)
            return 1.0
            
        self.dur_ema.update(duration)
        
        # Intensity: If Avg is 60s and Current is 30s -> Intensity = 2.0 (Fuel Gauge Met)
        return avg_dur / duration

class StreamingAggressorRatio:
    """
    Price Action Aggressor: (Close - Low) / (High - Low)
    1.0 = Bulls closed at High.
    0.0 = Bears closed at Low.
    """
    def update(self, high: float, low: float, close: float) -> float:
        rng = high - low
        if rng <= 1e-9:
            return 0.5 # Flat bar
        
        ratio = (close - low) / rng
        return max(0.0, min(ratio, 1.0)) # Clamp 0-1

# --- 2. QUANTITATIVE MATH KERNELS ---

class StreamingFracDiff:
    """
    Computes the Fractionally Differentiated value of a time series.
    """
    def __init__(self, d=0.4, window=2000, tolerance=1e-5):
        self.d = d
        self.window = window
        self.tolerance = tolerance
        self.weights = self._calculate_weights()
        self.history = deque(maxlen=len(self.weights))
        self.warmup_complete = False
        
    def _calculate_weights(self):
        w = [1.0]
        k = 1
        while True:
            w_next = -w[-1] * ((self.d - k + 1) / k)
            if abs(w_next) < self.tolerance or k >= self.window:
                break
            w.append(w_next)
            k += 1
        return np.array(w[::-1])

    def update(self, price):
        if not isinstance(price, (int, float)) or not math.isfinite(price):
            return 0.0
        self.history.append(price)
        if len(self.history) < len(self.weights):
            return 0.0
        if not self.warmup_complete:
            self.warmup_complete = True
        window_array = np.array(self.history)
        frac_diff_value = np.dot(self.weights, window_array)
        return float(frac_diff_value)

class MicrostructureAnalyzer:
    """
    Analyzes Volume Bars to calculate Order Flow Imbalance (OFI).
    Used for the Aggressor Trigger.
    """
    def __init__(self, ema_alpha=0.1):
        self.ofi_smoothed = 0.0
        self.alpha = ema_alpha

    def process_bar(self, buy_vol: float, sell_vol: float) -> float:
        current_imbalance = buy_vol - sell_vol
        self.ofi_smoothed = (self.alpha * current_imbalance) + ((1 - self.alpha) * self.ofi_smoothed)
        return self.ofi_smoothed

# --- 3. REGIME INDICATORS ---

class KaufmanEfficiencyRatio:
    """
    Quantifies trend efficiency (Signal vs Noise).
    CRITICAL: Must be > 0.10 for V9 Breakout Strategy.
    """
    def __init__(self, window: int = 10):
        self.window = window
        self.prices = deque(maxlen=window + 1)

    def update(self, price: float) -> float:
        self.prices.append(price)
        if len(self.prices) < self.window + 1:
            return 0.5
        signal = abs(self.prices[-1] - self.prices[0])
        arr = np.array(self.prices)
        noise = np.sum(np.abs(np.diff(arr)))
        if noise == 0:
            return 1.0
        return signal / noise

class FractalDimensionIndex:
    """
    Measures market complexity. Range: ~1.0 (Trend) to ~2.0 (Mean Rev).
    """
    def __init__(self, window: int = 30):
        self.window = window
        self.prices = deque(maxlen=window)

    def update(self, price: float) -> float:
        self.prices.append(price)
        if len(self.prices) < self.window:
            return 1.5
        data = np.array(self.prices)
        min_p, max_p = np.min(data), np.max(data)
        if max_p == min_p: return 1.0
        scaled = (data - min_p) / (max_p - min_p)
        dt = 1.0 / (self.window - 1)
        diffs = np.diff(scaled)
        length = np.sum(np.sqrt(diffs**2 + dt**2))
        if length <= 0: return 1.5
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
    Online Hidden Markov Model (HMM) for regime classification.
    Robust against convergence failures (Hard Reset Logic).
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
            logging.getLogger("hmmlearn").setLevel(logging.ERROR)
            logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)
            
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
            
            # MATH HARDENING: Check variance. If data is flat, HMM will crash.
            if np.var(data) < 1e-9 or np.isnan(data).any():
                return self.last_regime

            # Retrain periodically or early on
            should_fit = (self.fit_counter % 50 == 0) or (self.fit_counter < 200)
            
            if should_fit:
                # HARD RESET LOGIC: If model fails to converge repeatedly, scramble it.
                if self.fit_failures > 5:
                    self.model.init_params = "stmc" # Re-init weights
                    self.fit_failures = 0
                elif hasattr(self.model, 'startprob_'):
                    self.model.init_params = "" # Keep weights, refine them
                
                # Suppress Stdout/Stderr from C-level HMM code
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.model.fit(data)
                
                if self.model.monitor_.converged:
                    # Sort states by Variance (Low Var = 0 = Quiet, High Var = 1 = Volatile)
                    variances = np.array([np.mean(np.diag(c)) for c in self.model.covars_])
                    sort_order = np.argsort(variances)
                    self.state_map = {old: new for new, old in enumerate(sort_order)}
                    self.fit_failures = 0
                else:
                    self.fit_failures += 1
                    # Fallback Identity map
                    self.state_map = {i: i for i in range(self.n_states)}
            
            # Predict current state
            raw_state = int(self.model.predict(data[-1].reshape(1, -1))[0])
            mapped_state = self.state_map.get(raw_state, raw_state)
            
            self.last_regime = mapped_state
            return mapped_state
            
        except Exception:
            self.fit_failures += 1
            return self.last_regime

# --- 4. STREAMING INDICATORS ---

class StreamingVortex:
    """
    Vortex Indicator (VI) for Regime Detection.
    VI > 1.0 implies Trend, VI < 1.0 implies Chop.
    Uses sliding window summation for stability.
    """
    def __init__(self, period: int = 14):
        self.period = period
        self.tr_buffer = deque(maxlen=period)
        self.vm_plus_buffer = deque(maxlen=period)
        self.vm_minus_buffer = deque(maxlen=period)
        self.prev_low = None
        self.prev_high = None
        self.prev_close = None

    def update(self, high: float, low: float, close: float) -> Tuple[float, float]:
        if self.prev_close is None:
            self.prev_high = high
            self.prev_low = low
            self.prev_close = close
            return 1.0, 1.0

        # 1. True Range
        tr1 = high - low
        tr2 = abs(high - self.prev_close)
        tr3 = abs(low - self.prev_close)
        tr = max(tr1, tr2, tr3)
        
        # 2. Vortex Movements
        vm_plus = abs(high - self.prev_low)
        vm_minus = abs(low - self.prev_high)
        
        # 3. Update Buffers
        self.tr_buffer.append(tr)
        self.vm_plus_buffer.append(vm_plus)
        self.vm_minus_buffer.append(vm_minus)
        
        # 4. Update State
        self.prev_high = high
        self.prev_low = low
        self.prev_close = close
        
        if len(self.tr_buffer) < self.period:
            return 1.0, 1.0
            
        sum_tr = sum(self.tr_buffer)
        if sum_tr < 1e-9:
            return 1.0, 1.0
            
        vi_plus = sum(self.vm_plus_buffer) / sum_tr
        vi_minus = sum(self.vm_minus_buffer) / sum_tr
        
        return vi_plus, vi_minus

class StreamingChoppiness:
    """
    Choppiness Index (CHOP) for Regime Detection.
    CHOP > 50 implies Sideways/Consolidation.
    CHOP < 38 implies Trending.
    Formula: 100 * LOG10( SUM(ATR, n) / ( MaxHi(n) - MinLo(n) ) ) / LOG10(n)
    """
    def __init__(self, period: int = 14):
        self.period = period
        self.tr_buffer = deque(maxlen=period)
        self.high_buffer = deque(maxlen=period)
        self.low_buffer = deque(maxlen=period)
        self.prev_close = None

    def update(self, high: float, low: float, close: float) -> float:
        if self.prev_close is None:
            self.prev_close = close
            self.high_buffer.append(high)
            self.low_buffer.append(low)
            return 50.0 # Neutral start

        # 1. True Range
        tr1 = high - low
        tr2 = abs(high - self.prev_close)
        tr3 = abs(low - self.prev_close)
        tr = max(tr1, tr2, tr3)
        
        # 2. Update Buffers
        self.tr_buffer.append(tr)
        self.high_buffer.append(high)
        self.low_buffer.append(low)
        self.prev_close = close
        
        if len(self.tr_buffer) < self.period:
            return 50.0
            
        # 3. Calculate CHOP
        sum_tr = sum(self.tr_buffer)
        max_hi = max(self.high_buffer)
        min_lo = min(self.low_buffer)
        rng = max_hi - min_lo
        
        if rng <= 1e-9 or sum_tr <= 1e-9:
            return 50.0
            
        try:
            # 100 * Log10(SumTR / Range) / Log10(n)
            ratio = sum_tr / rng
            if ratio <= 0: return 50.0 # Guard log domain error
            
            numerator = math.log10(ratio)
            denominator = math.log10(self.period)
            chop = 100.0 * (numerator / denominator)
            return max(0.0, min(100.0, chop))
        except ValueError:
            return 50.0

class StreamingADX:
    """
    Critical for Trend Detection.
    """
    def __init__(self, period: int = 14):
        self.period = period
        self.dm_plus_ema = RecursiveEMA(alpha=1/period)
        self.dm_minus_ema = RecursiveEMA(alpha=1/period)
        self.tr_ema = RecursiveEMA(alpha=1/period)
        self.dx_ema = RecursiveEMA(alpha=1/period)
        
        self.prev_high = None
        self.prev_low = None
        self.prev_close = None
        
    def update(self, high: float, low: float, close: float) -> float:
        if self.prev_close is None:
            self.prev_high = high
            self.prev_low = low
            self.prev_close = close
            return 0.0
            
        tr1 = high - low
        tr2 = abs(high - self.prev_close)
        tr3 = abs(low - self.prev_close)
        true_range = max(tr1, tr2, tr3)
        
        up_move = high - self.prev_high
        down_move = self.prev_low - low
        
        dm_plus = up_move if (up_move > down_move and up_move > 0) else 0.0
        dm_minus = down_move if (down_move > up_move and down_move > 0) else 0.0
        
        self.tr_ema.update(true_range)
        self.dm_plus_ema.update(dm_plus)
        self.dm_minus_ema.update(dm_minus)
        
        avg_tr = self.tr_ema.get()
        avg_dm_plus = self.dm_plus_ema.get()
        avg_dm_minus = self.dm_minus_ema.get()
        
        if avg_tr > 0:
            di_plus = 100 * (avg_dm_plus / avg_tr)
            di_minus = 100 * (avg_dm_minus / avg_tr)
        else:
            di_plus = 0.0
            di_minus = 0.0
            
        sum_di = di_plus + di_minus
        if sum_di > 0:
            dx = 100 * abs(di_plus - di_minus) / sum_di
        else:
            dx = 0.0
            
        self.dx_ema.update(dx)
        self.prev_high = high
        self.prev_low = low
        self.prev_close = close
        
        return self.dx_ema.get()

class StreamingIndicators:
    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26, macd_sig=9, atr_period=14):
        self.ema_fast = RecursiveEMA(alpha=2 / (macd_fast + 1))
        self.ema_slow = RecursiveEMA(alpha=2 / (macd_slow + 1))
        self.macd_signal = RecursiveEMA(alpha=2 / (macd_sig + 1))
        
        self.rsi_period = rsi_period
        self.rsi_avg_gain = RecursiveEMA(alpha=1 / rsi_period)
        self.rsi_avg_loss = RecursiveEMA(alpha=1 / rsi_period)
        self.prev_price = None
        
        self.atr_mean = RecursiveEMA(alpha=1 / atr_period)
        self.prev_close = None
        
        self.adx = StreamingADX(period=14)

    def update(self, price: float, high: float, low: float) -> Dict[str, float]:
        features = {}
        
        # MACD
        self.ema_fast.update(price)
        self.ema_slow.update(price)
        self.macd_line = self.ema_fast.get() - self.ema_slow.get()
        self.macd_signal.update(self.macd_line)
        histogram = self.macd_line - self.macd_signal.get()
        
        features['macd_line'] = self.macd_line
        features['macd_hist'] = histogram
        
        # RSI
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
            
        # ATR
        if self.prev_close is not None:
            tr1 = high - low
            tr2 = abs(high - self.prev_close)
            tr3 = abs(low - self.prev_close)
            true_range = max(tr1, tr2, tr3)
            self.atr_mean.update(true_range)
            features['atr'] = self.atr_mean.get()
        else:
            features['atr'] = high - low if (high > 0 and low > 0 and high != low) else 0.001
            
        # ADX Only
        adx_val = self.adx.update(high, low, price)
        features['adx'] = adx_val
            
        self.prev_price = price
        self.prev_close = price
        return features

# --- 5. ADAPTIVE TRIPLE BARRIER ---

class AdaptiveTripleBarrier:
    """
    Labeling engine.
    Updated for FTMO Sniper Mode: Extended horizon to capture Swing moves.
    """
    def __init__(self, horizon_ticks: int = 144, risk_mult: float = 1.0, reward_mult: float = 2.0, drift_threshold: float = 0.75):
        # V12.4 ALIGNMENT: Default horizon increased to ~12 hours (144 ticks @ M5)
        # matches the winning "Alpha Asset" profile from backtesting.
        self.buffer = deque()
        self.time_limit = horizon_ticks
        self.risk_mult = risk_mult
        self.reward_mult = reward_mult
        self.drift_threshold = drift_threshold

    def add_trade_opportunity(self, features: Dict[str, float], entry_price: float, current_atr: float, timestamp: float, parkinson_vol: float = 0.0):
        if current_atr <= 0: current_atr = entry_price * 0.0001
        
        # Base Volatility Scaling
        volatility = features.get('volatility', 0.0)
        adaptive_scalar = 1.0 + (volatility * 100.0)
        
        # Parkinson Volatility Boost (Expansion Logic)
        p_vol = parkinson_vol if parkinson_vol > 0 else features.get('parkinson_vol', 0.0)
        
        vol_boost = 0.0
        if p_vol > 0.002:
            vol_boost = 0.5 # Add 0.5x ATR room during expansion
        
        effective_risk_mult = (self.risk_mult + vol_boost) * adaptive_scalar
        effective_reward_mult = (self.reward_mult + vol_boost) * adaptive_scalar
        
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
                label = 1  # BUY
                realized_ret = (trade['tp'] - trade['entry']) / trade['entry']
            elif current_low <= trade['sl']:
                label = -1 # SELL
                realized_ret = (trade['entry'] - trade['sl']) / trade['entry']
            elif trade['age'] >= self.time_limit:
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

# --- 6. PROBABILITY CALIBRATOR & META LABELER ---

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

# --- 7. ONLINE FEATURE ENGINEER (PROJECT PHOENIX V9.0) ---

class OnlineFeatureEngineer:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.prices = deque(maxlen=window_size)
        self.volumes = deque(maxlen=window_size)
        self.returns = deque(maxlen=window_size)
        
        self.indicators = StreamingIndicators()
        
        # Welford Scaler for Volume (Infinite Window Stationarity)
        self.welford_volume = WelfordScaler()
        self.welford_ofi = WelfordScaler()
        
        # Regime Indicators
        self.ker = KaufmanEfficiencyRatio(window=10)
        self.fdi = FractalDimensionIndex(window=30)
        self.regime_detector = RegimeDetector(n_states=2, window=100)
        
        # New Regime Filters (Anti-Chop)
        self.vortex = StreamingVortex(period=14)
        self.choppiness = StreamingChoppiness(period=14)
        
        # Project Phoenix L1 Proxies
        self.parkinson = StreamingParkinsonVolatility(alpha=0.1)
        self.amihud = StreamingAmihudLiquidity(alpha=0.05)
        self.rvol = StreamingRelativeVolume(window=20)
        self.aggressor = StreamingAggressorRatio()
        
        # V12.4 MOMENTUM LOGIC: Bollinger Bands
        self.bb = StreamingBollingerBands(window=20, num_std=1.5)

        # Microstructure & Math Engines
        self.entropy = EntropyMonitor(window=window_size)
        self.vpin = VPINMonitor(bucket_size=1000)
        self.frac_diff = StreamingFracDiff(d=0.4, window=window_size)
        self.microstructure = MicrostructureAnalyzer(ema_alpha=0.1)
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
               sentiment: float = 0.0,
               context_data: Dict[str, Any] = None) -> Dict[str, float]:
        
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
        
        # Update Anti-Chop Filters
        vi_plus, vi_minus = self.vortex.update(high, low, price)
        chop_index = self.choppiness.update(high, low, price)
        
        # Update L1 Proxies (Project Phoenix)
        parkinson_val = self.parkinson.update(high, low)
        amihud_val = self.amihud.update(abs(ret_log), price, volume)
        rvol_val = self.rvol.update(timestamp) # Fuel Gauge (Duration Intensity)
        aggressor_val = self.aggressor.update(high, low, price)
        
        # Update V9 Momentum
        bb_feats = self.bb.update(price)

        # Update Other Metrics
        entropy_val = self.entropy.update(price)
        vpin_val = self.vpin.update(volume, price, buy_vol, sell_vol)
        volatility_val = self.vol_monitor.update(ret_log)

        # FracDiff
        fd_price = self.frac_diff.update(price)

        # Microstructure (OFI)
        raw_ofi_smoothed = self.microstructure.process_bar(buy_vol, sell_vol)
        micro_ofi_z = self.welford_ofi.update(raw_ofi_smoothed)

        # Volatility Ratio
        self.vol_baseline.update(volatility_val)
        baseline_vol = self.vol_baseline.get()
        vol_ratio = volatility_val / baseline_vol if baseline_vol > 1e-9 else 1.0

        # Hurst
        hurst_val = 0.5
        if len(self.returns) >= 20:
            ret_arr = np.array(list(self.returns), dtype=np.float64)
            hurst_val = calculate_hurst(ret_arr)

        # Legacy OFI
        denominator = volume if volume > 0 else 1.0
        ofi_simple = (buy_vol - sell_vol) / denominator
        self.ofi_window.append(ofi_simple)
        cum_ofi = sum(self.ofi_window) / len(self.ofi_window) if self.ofi_window else 0.0
        self.ofi_ema.update(ofi_simple)
        ofi_trend = self.ofi_ema.get()

        # Trends
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
        volume_z = self.welford_volume.update(volume)

        # MTF Context
        d1_trend = 0.0
        mtf_align = 0.0
        
        if context_data:
            d1 = context_data.get('d1', {})
            d1_ema = d1.get('ema200', 0.0)
            if d1_ema > 0:
                d1_trend = 1.0 if price > d1_ema else -1.0
            
            h4 = context_data.get('h4', {})
            h4_rsi = h4.get('rsi', 50.0)
            h4_trend = 1.0 if h4_rsi > 50 else -1.0
            
            m5_trend = 1.0 if tech_feats['macd_line'] > 0 else -1.0
            
            if d1_trend != 0 and (d1_trend == h4_trend == m5_trend):
                mtf_align = 1.0

        # --- FLOW IMBALANCE FEATURES (AGGRESSOR) ---
        safe_total_vol = buy_vol + sell_vol
        flow_imbalance = (buy_vol - sell_vol) / safe_total_vol if safe_total_vol > 0 else 0.0
        
        safe_sell = sell_vol if sell_vol > 0 else 1.0
        flow_ratio = buy_vol / safe_sell
        
        raw_features = {
            # Core
            'log_ret': ret_log,
            'volatility': volatility_val,
            'atr': current_atr,
            'atr_pct': current_atr / price,
            
            # Project Phoenix L1 Proxies
            'parkinson_vol': parkinson_val,
            'amihud': amihud_val,
            'rvol': rvol_val,
            'aggressor': aggressor_val,
            'flow_imbalance': flow_imbalance, 
            'flow_ratio': flow_ratio,           
            
            # V9 Momentum Features (NEW)
            'bb_breakout': bb_feats['bb_breakout'],
            'bb_width': bb_feats['bb_width'],
            'bb_pct_b': bb_feats['bb_pct_b'],
            
            # Regime & Math
            'ker': ker_val,
            'fdi': fdi_val,
            'hmm_regime': float(regime_hmm) / (self.regime_detector.n_states - 1) if self.regime_detector.n_states > 1 else 0.0,
            'entropy': entropy_val,
            'hurst': hurst_val,
            'frac_diff': fd_price,
            'vpin': vpin_val,
            'choppiness': chop_index,    # NEW: 0-100 (50+ = Chop)
            'vortex_spread': vi_plus - vi_minus, # NEW: >0 Bullish, <0 Bearish
            
            # Technicals (Trend Focused)
            'rsi_norm': rsi_norm,
            'macd_norm': macd_norm,
            'macd_hist_norm': tech_feats['macd_hist'] / price,
            'adx': tech_feats.get('adx', 0.0),
            
            # Context / Legacy
            'vol_ratio': vol_ratio,
            'volatility_log': math.log(volatility_val + 1e-9),
            'micro_ofi': micro_ofi_z,
            'ofi_simple': ofi_simple,
            'efficiency_ratio': er_val,
            'cum_ofi': cum_ofi,
            'ofi_trend': ofi_trend,
            'regime': regime_val,
            'vol_breakout': vol_breakout,
            'sentiment': sentiment,
            
            # MTF
            'd1_trend': d1_trend,
            'mtf_alignment': mtf_align,
            
            # Physics
            'body_ratio': body_ratio,
            'upper_wick_ratio': upper_wick_ratio,
            'lower_wick_ratio': lower_wick_ratio,
            'volume_z': volume_z,
            
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

# --- 8. LEGACY MONITORS (PRESERVED) ---

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
    """Wrapper for StreamingFracDiff legacy compatibility."""
    def __init__(self, d: float = 0.6, window: int = 20):
        self.impl = StreamingFracDiff(d=d, window=window)
    def update(self, price: float) -> float:
        return self.impl.update(price)

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
    """
    Calculates the Hurst Exponent using Numba-optimized standard deviation analysis.
    V16.20 FIX: Uses explicit manual linear regression (O(N)) to replace 
    np.linalg.lstsq, which is unstable across Numpy versions in Numba.
    """
    n = len(ts)
    if n < 20: return 0.5
    
    # 1. Variance Check (Math Guard)
    if np.std(ts) < 1e-9: return 0.5
    
    lags = np.arange(2, 20, dtype=np.float64)
    tau = np.zeros(len(lags), dtype=np.float64)
    
    for i in range(len(lags)):
        lag = int(lags[i])
        # Manually compute diff to avoid creation of large arrays if possible, 
        # but here slicing is cleaner.
        diff = ts[lag:] - ts[:-lag]
        std_diff = np.std(diff)
        
        # Guard against zero std dev in flat markets
        if std_diff < 1e-9:
            tau[i] = 1e-9
        else:
            tau[i] = std_diff
    
    # Safe Log (Avoid log(0))
    # Using np.maximum is safe in Numba
    log_lags = np.log(lags)
    log_tau = np.log(tau)
    
    # 2. MANUAL LINEAR REGRESSION (Least Squares)
    # y = mx + c
    # m = (N * sum(xy) - sum(x) * sum(y)) / (N * sum(x^2) - (sum(x))^2)
    
    N = float(len(lags))
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_tau)
    sum_xy = np.sum(log_lags * log_tau)
    sum_xx = np.sum(log_lags * log_lags)
    
    denominator = (N * sum_xx) - (sum_x * sum_x)
    
    if abs(denominator) < 1e-9:
        return 0.5
        
    slope = ((N * sum_xy) - (sum_x * sum_y)) / denominator
    
    # Hurst = Slope (in this specific R/S proxy method)
    hurst = slope
    
    # Clamp result
    if hurst < 0.0: return 0.0
    if hurst > 1.0: return 1.0
    return hurst

def enrich_with_d1_data(features: Dict[str, float], d1_data: Dict[str, float], current_price: float) -> Dict[str, float]:
    if not d1_data: return features
    prev_high = d1_data.get('high', 0)
    prev_low = d1_data.get('low', 0)
    if prev_high == 0: return features
    
    features['dist_d1_high'] = (prev_high - current_price) / current_price
    features['dist_d1_low'] = (current_price - prev_low) / current_price
    return features