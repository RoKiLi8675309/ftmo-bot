from __future__ import annotations
import warnings
import pandas as pd
import numpy as np
import pytz
import sys
import logging
import math
import os
import threading
import copy
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# Shared Imports
from shared.core.config import CONFIG
from shared.domain.models import VolumeBar

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
    from river import linear_model, forest, metrics, optim, drift, compose, preprocessing
    from sklearn.isotonic import IsotonicRegression
    from sklearn.exceptions import ConvergenceWarning
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger("Features")

# --- 0. HELPER MATH KERNELS (ROBUST) ---

class WelfordScaler:
    """
    V20.6 FIX: Upgraded from Infinite Welford to Exponential Moving Average (EMA) Scaler.
    Prevents Stationarity Lockup where N becomes so large that recent volatility 
    expansions are ignored due to the sheer mass of historical data.
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.mean = 0.0
        self.var = 0.0
        self.initialized = False

    def update(self, x: float) -> float:
        """Updates statistics via EWMA and returns the Z-Score of the new value."""
        if x is None or not math.isfinite(x):
            return 0.0
            
        if not self.initialized:
            self.mean = x
            self.var = 0.0
            self.initialized = True
            return 0.0
            
        # EWMA Variance and Mean Update
        diff = x - self.mean
        incr = self.alpha * diff
        self.mean += incr
        self.var = (1 - self.alpha) * (self.var + diff * incr)
        
        stdev = math.sqrt(self.var)
        
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

# --- 1. MOMENTUM INDICATORS ---

class StreamingBollingerBands:
    """
    Sniper Mode Breakout Indicator.
    Calculates Upper/Lower Bands, Width, and Squeeze Percentile
    for Breakout & Volatility Contraction Detection.
    """
    def __init__(self, window: int = 20, num_std: float = 2.0, squeeze_window: int = 100):
        self.window = window
        self.num_std = num_std
        self.buffer = deque(maxlen=window)
        self.width_history = deque(maxlen=squeeze_window) # Track historical width for squeeze
        
    def update(self, price: float) -> Dict[str, float]:
        self.buffer.append(price)
        
        if len(self.buffer) < self.window:
            return {
                'bb_upper': price, 
                'bb_lower': price, 
                'bb_mid': price, 
                'bb_width': 0.0, 
                'bb_pct_b': 0.5,
                'bb_breakout': 0.0,
                'bb_squeeze_percentile': 0.5
            }
            
        # Calculate Stats
        mu = np.mean(self.buffer)
        std = np.std(self.buffer)
        if std < 1e-9: std = 1e-9 # Prevent Div/0 Flatline Collapse
        
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
            
        # --- SQUEEZE DETECTOR ---
        # Measures how tight current bands are compared to the last N periods.
        if len(self.width_history) > 10:
            hist_array = np.array(self.width_history)
            squeeze_pct = np.sum(hist_array < width) / len(hist_array)
        else:
            squeeze_pct = 0.5
            
        self.width_history.append(width)
            
        return {
            'bb_upper': upper,
            'bb_lower': lower,
            'bb_mid': mu,
            'bb_width': width,
            'bb_pct_b': pct_b,
            'bb_breakout': breakout,
            'bb_squeeze_percentile': float(squeeze_pct)
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

class AdaptiveImbalanceBarGenerator:
    """
    V20.18 Parity: Generates Tick Imbalance Bars (TIBs).
    Replaces time-based sampling with information-driven sampling.
    """
    def __init__(self, symbol: str, initial_threshold: float = 10.0, alpha: float = 0.05):
        self.symbol = symbol
        
        # 🚨 V20.18 FIX: Store the dynamic minimum threshold derived from the calibration phase.
        # This prevents the hardcap bug from destroying retail FX dense data streams.
        self.min_threshold = max(0.1, initial_threshold * 0.1)
        
        # State variables for the current bar
        self.current_imbalance = 0.0
        self.ticks_in_bar = 0
        self.open_price = None
        self.high_price = -float('inf')
        self.low_price = float('inf')
        self.close_price = None
        self.volume_accum = 0.0
        self.start_timestamp = None
        
        # Extended VPIN states
        self.current_buy_vol = 0.0
        self.current_sell_vol = 0.0
        self.vwap_sum = 0.0

        # State for Tick Rule (Aggressor Logic)
        self.prev_price = None
        self.prev_tick_rule = 1  # Default to Buy (1)

        # Adaptive Threshold Logic (EWMA)
        self.expected_imbalance = initial_threshold
        self.alpha = alpha   # Forgetting factor for EWMA

    def process_tick(self, price: float, volume: float, timestamp: float, 
                     external_buy_vol: float = 0.0, external_sell_vol: float = 0.0) -> Optional[VolumeBar]:
        """
        Ingest a single tick and determine if a bar should be closed based on Imbalance.
        """
        # Initialize if first tick
        if self.prev_price is None:
            self.prev_price = price
            self.open_price = price
            self.start_timestamp = timestamp
            return None

        # 1. Apply Tick Rule (Infer Aggressor)
        if external_buy_vol > 0 or external_sell_vol > 0:
            tick_rule = 1 if external_buy_vol > external_sell_vol else -1
            if external_buy_vol == external_sell_vol: tick_rule = 0
            self.prev_tick_rule = tick_rule
        else:
            price_change = price - self.prev_price
            if price_change != 0:
                tick_rule = np.sign(price_change)
                self.prev_tick_rule = tick_rule
            else:
                tick_rule = self.prev_tick_rule

        # 2. Update Accumulators
        self.current_imbalance += (tick_rule * volume)
        self.ticks_in_bar += 1
        self.volume_accum += volume
        self.vwap_sum += (price * volume)
        
        if self.open_price is None: self.open_price = price
        self.high_price = max(self.high_price, price)
        self.low_price = min(self.low_price, price)
        self.close_price = price
        self.prev_price = price

        if tick_rule == 1:
            self.current_buy_vol += volume
        elif tick_rule == -1:
            self.current_sell_vol += volume
        else:
            self.current_buy_vol += (volume / 2)
            self.current_sell_vol += (volume / 2)

        # 3. Check Threshold Condition
        if abs(self.current_imbalance) >= self.expected_imbalance or self.ticks_in_bar >= 1000:
            # ⚠️ WARNING FIX: Calculate volume spillover to prevent data distortion
            excess_imbalance = abs(self.current_imbalance) - self.expected_imbalance
            spill_vol = 0.0
            if abs(self.current_imbalance) > 0:
                spill_ratio = excess_imbalance / abs(self.current_imbalance)
                spill_vol = volume * spill_ratio

            bar = self._finalize_bar(timestamp, price)
            
            # Apply spillover to next bar state
            if excess_imbalance > 0:
                self.current_imbalance = tick_rule * excess_imbalance
                self.volume_accum = spill_vol
                self.vwap_sum = price * spill_vol
                self.ticks_in_bar = 1
                self.open_price = price
                if tick_rule == 1: self.current_buy_vol = spill_vol
                elif tick_rule == -1: self.current_sell_vol = spill_vol
                else:
                    self.current_buy_vol = spill_vol / 2.0
                    self.current_sell_vol = spill_vol / 2.0

            return bar

        return None

    def _finalize_bar(self, timestamp: float, close_price: float) -> VolumeBar:
        if isinstance(timestamp, (float, int)):
            dt_ts = datetime.fromtimestamp(timestamp, pytz.utc)
        else:
            dt_ts = timestamp

        vwap = self.vwap_sum / self.volume_accum if self.volume_accum > 0 else close_price

        bar = VolumeBar(
            timestamp=dt_ts,
            open=self.open_price,
            high=self.high_price,
            low=self.low_price,
            close=close_price,
            volume=self.volume_accum,
            vwap=vwap,
            tick_count=self.ticks_in_bar,
            buy_vol=self.current_buy_vol,
            sell_vol=self.current_sell_vol
        )

        current_abs_imb = abs(self.current_imbalance)
        self.expected_imbalance = (self.alpha * current_abs_imb) + ((1 - self.alpha) * self.expected_imbalance)
        
        # 🚨 V20.18 FIX: Utilize self.min_threshold instead of a hardcoded 10.0!
        # This completely cures the Data Starvation bug on retail fractional tick volume feeds.
        self.expected_imbalance = max(self.min_threshold, min(self.expected_imbalance, 10000.0))

        self.current_imbalance = 0.0
        self.ticks_in_bar = 0
        self.open_price = None
        self.high_price = -float('inf')
        self.low_price = float('inf')
        self.volume_accum = 0.0
        self.vwap_sum = 0.0
        self.current_buy_vol = 0.0
        self.current_sell_vol = 0.0
        self.start_timestamp = timestamp

        return bar

# --- 3. REGIME INDICATORS ---

class KaufmanEfficiencyRatio:
    """
    Quantifies trend efficiency (Signal vs Noise).
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
    """
    def __init__(self, n_states: int = 2, window: int = 100, executor: Optional[ThreadPoolExecutor] = None):
        self.n_states = n_states
        self.window = window
        self.returns = deque(maxlen=window)
        self.last_regime = 0
        self.model = None
        self.fit_failures = 0
        self.fit_counter = 0
        self.state_map = {i: i for i in range(n_states)}
        
        # Concurrency protections
        self.lock = threading.Lock()
        
        # 🚨 V20.18 FIX: Track whether we own the executor or if it was injected
        self._owns_executor = False
        self.executor = executor
        self.is_fitting = False
        
        if HMM_AVAILABLE:
            logging.getLogger("hmmlearn").setLevel(logging.ERROR)
            logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)
            
            try:
                self.model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",
                    n_iter=20, # Reduced from 100 to prevent GIL blocking
                    random_state=42,
                    init_params="stmc",
                    verbose=False,
                    min_covar=1e-3,
                    tol=1e-4
                )
            except Exception as e:
                logger.error(f"HMM Init Failed: {e}")
                self.model = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'lock' in state: del state['lock']
        if 'executor' in state: del state['executor']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()
        self.executor = None
        self._owns_executor = False
        self.is_fitting = False
    
    def update(self, ret: float) -> int:
        self.returns.append(ret * 100.0)
        self.fit_counter += 1
        
        if not HMM_AVAILABLE or self.model is None:
            return 0
            
        if len(self.returns) < self.window:
            return self.last_regime
            
        try:
            data = np.array(self.returns).reshape(-1, 1)
            
            if np.var(data) < 1e-9 or np.isnan(data).any():
                return self.last_regime

            should_fit = (self.fit_counter % 50 == 0) or (self.fit_counter < 200 and self.fit_counter % 10 == 0)
            
            if should_fit and not self.is_fitting:
                self.is_fitting = True
                fit_data = data.copy()
                
                # 🚨 V20.18 FIX: Synchronous Execution during Backtests!
                # If no executor is provided, we MUST run synchronously to prevent cross-trial
                # state contamination and assure pure determinism during Optuna search.
                if self.executor is not None:
                    self.executor.submit(self._async_fit, fit_data)
                else:
                    self._async_fit(fit_data)
            
            with self.lock:
                if hasattr(self.model, 'startprob_'):
                    raw_state = int(self.model.predict(data[-1].reshape(1, -1))[0])
                else:
                    raw_state = 0
                    
            mapped_state = self.state_map.get(raw_state, raw_state)
            
            self.last_regime = mapped_state
            return mapped_state
            
        except Exception:
            return self.last_regime
            
    def _async_fit(self, data):
        try:
            with self.lock:
                model_clone = copy.deepcopy(self.model) if hasattr(self.model, 'startprob_') else self.model
            
            if self.fit_failures > 5:
                model_clone.init_params = "stmc" # Re-init weights
            elif hasattr(model_clone, 'startprob_'):
                model_clone.init_params = "" # Keep weights, refine them
                
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_clone.fit(data)
            
            if model_clone.monitor_.converged:
                variances = np.array([np.mean(np.diag(c)) for c in model_clone.covars_])
                sort_order = np.argsort(variances)
                new_map = {old: new for new, old in enumerate(sort_order)}
                
                with self.lock:
                    self.model = model_clone
                    self.state_map = new_map
                    self.fit_failures = 0
            else:
                self.fit_failures += 1
        except Exception:
            self.fit_failures += 1
        finally:
            self.is_fitting = False

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

        tr1 = high - low
        tr2 = abs(high - self.prev_close)
        tr3 = abs(low - self.prev_close)
        tr = max(tr1, tr2, tr3)
        
        vm_plus = abs(high - self.prev_low)
        vm_minus = abs(low - self.prev_high)
        
        self.tr_buffer.append(tr)
        self.vm_plus_buffer.append(vm_plus)
        self.vm_minus_buffer.append(vm_minus)
        
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
            return 50.0

        tr1 = high - low
        tr2 = abs(high - self.prev_close)
        tr3 = abs(low - self.prev_close)
        tr = max(tr1, tr2, tr3)
        
        self.tr_buffer.append(tr)
        self.high_buffer.append(high)
        self.low_buffer.append(low)
        self.prev_close = close
        
        if len(self.tr_buffer) < self.period:
            return 50.0
            
        sum_tr = sum(self.tr_buffer)
        max_hi = max(self.high_buffer)
        min_lo = min(self.low_buffer)
        rng = max_hi - min_lo
        
        if rng <= 1e-9 or sum_tr <= 1e-9:
            return 50.0
            
        try:
            ratio = sum_tr / rng
            if ratio <= 0: return 50.0 
            
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

# --- 5. META LABELING & PROBABILITY CALIBRATION ---

class ProbabilityCalibrator:
    """
    Platt Scaling Calibrator adapted for swing trading confidence.
    """
    def __init__(self, window: int = 1000):
        self.calibrator = None
        if ML_AVAILABLE:
            self.calibrator = linear_model.LogisticRegression(
                optimizer=optim.SGD(0.05),
                loss=optim.losses.Log(),
                l2=0.1
            )
        self.samples_seen = 0
        self.pos_count = 0
        self.neg_count = 0

    def update(self, prob: float, label: int):
        if ML_AVAILABLE and math.isfinite(prob):
            self.calibrator.learn_one({'raw_prob': prob}, label)
            self.samples_seen += 1
            if label == 1:
                self.pos_count += 1
            else:
                self.neg_count += 1

    def calibrate(self, raw_prob: float) -> float:
        if not ML_AVAILABLE: return raw_prob
        # Return RAW PROBABILITY until mature so the base model can filter natively
        if self.pos_count < 10 or self.neg_count < 10:
            return raw_prob 
            
        try:
            calibrated_dict = self.calibrator.predict_proba_one({'raw_prob': raw_prob})
            calibrated = calibrated_dict.get(1, raw_prob)
            return max(0.01, min(calibrated, 0.99))
        except Exception:
            return raw_prob

class MetaLabeler:
    """
    Secondary model that learns when the primary model is likely to be right or wrong.
    """
    def __init__(self):
        self.model = None
        self.buffer = deque(maxlen=1000)
        self.pos_count = 0
        self.neg_count = 0
        
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
        
        if y_meta == 1:
            self.pos_count += 1
        else:
            self.neg_count += 1
            
        try:
            meta_feats = self._enrich(features, primary_action)
            clean_features = self._sanitize(meta_feats)
            self.model.learn_one(clean_features, y_meta)
            self.buffer.append((clean_features, y_meta))
        except Exception as e:
            logger.debug(f"MetaLabeler Update Error: {e}")

    def predict(self, features: Dict[str, float], primary_action: int, threshold: float = 0.50) -> bool:
        if not ML_AVAILABLE or primary_action == 0:
            return False
            
        if self.pos_count < 10 or self.neg_count < 10: 
            return True 
            
        try:
            meta_feats = self._enrich(features, primary_action)
            clean_features = self._sanitize(meta_feats)
            probs = self.model.predict_proba_one(clean_features)
            prob_profit = probs.get(1, 0.0)
            return prob_profit >= threshold
        except Exception as e:
            logger.debug(f"MetaLabeler Predict Error: {e}")
            return True

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
            if v is not None and math.isfinite(v):
                clean[k] = float(v)
            else:
                clean[k] = 0.0
        return clean

# --- 6. ADAPTIVE TRIPLE BARRIER ---

class AdaptiveTripleBarrier:
    """
    Reality-Injected Labeling Engine.
    V20.8 FIX: Restored signal_id and current_timestamp arguments to ensure strict IPC compatibility.
    """
    def __init__(self, horizon_ticks: int = 144, risk_mult: float = 1.0, reward_mult: float = 2.0, 
                 drift_threshold: float = 0.75, horizon_type: str = 'TIME', horizon_value: float = 0.0):
        self.buffer = deque()
        self.time_limit = horizon_ticks
        self.risk_mult = risk_mult
        self.reward_mult = reward_mult
        self.drift_threshold = drift_threshold
        self.horizon_type = horizon_type.upper()
        self.horizon_threshold = horizon_value
        if self.horizon_type == 'TIME' and self.horizon_threshold == 0:
            self.horizon_threshold = horizon_ticks

    def add_trade_opportunity(self, features: Dict[str, float], entry_price: float, current_atr: float, 
                              timestamp: float, parkinson_vol: float = 0.0, 
                              min_stop_dist: float = 0.0, spread_in_price: float = 0.0, 
                              comm_in_price: float = 0.0, min_profit_dist: float = 0.0,
                              proposed_action: int = 0, pred_proba: float = 0.5,
                              override_reward_mult: float = None, signal_id: str = None):
        if current_atr <= 0: current_atr = entry_price * 0.0001
        
        volatility = features.get('volatility', 0.0)
        adaptive_scalar = 1.0 + (volatility * 100.0)
        
        p_vol = parkinson_vol if parkinson_vol > 0 else features.get('parkinson_vol', 0.0)
        vol_boost = 0.5 if p_vol > 0.002 else 0.0
        
        actual_reward_mult = override_reward_mult if override_reward_mult else self.reward_mult
        
        effective_risk_mult = (self.risk_mult + vol_boost) * adaptive_scalar
        effective_reward_mult = (actual_reward_mult + vol_boost) * adaptive_scalar
        
        raw_risk_dist = effective_risk_mult * current_atr
        actual_risk_dist = max(raw_risk_dist, min_stop_dist) 
        
        actual_reward_dist = actual_risk_dist * (effective_reward_mult / effective_risk_mult)
        
        buy_tp = entry_price + actual_reward_dist + spread_in_price
        buy_sl = entry_price - actual_risk_dist + spread_in_price
        
        sell_tp = entry_price - actual_reward_dist - spread_in_price
        sell_sl = entry_price + actual_risk_dist - spread_in_price

        min_profit_pct = (min_profit_dist + spread_in_price + comm_in_price) / entry_price if entry_price > 0 else 0.0
        
        feats_copy = features.copy()
        feats_copy['min_profit_pct'] = min_profit_pct

        self.buffer.append({
            'signal_id': signal_id,       # V20.8 IPC Fix
            'is_executed': False,         # V20.8 IPC Fix
            'features': feats_copy,
            'entry': entry_price,
            'buy_tp': buy_tp,
            'buy_sl': buy_sl,
            'sell_tp': sell_tp,
            'sell_sl': sell_sl,
            'atr': current_atr,
            'start_time': timestamp,
            'age': 0,
            'cum_vol': 0.0,
            'cum_volatility': 0.0,
            'spread_pct': spread_in_price / entry_price if entry_price > 0 else 0.0,
            'comm_pct': comm_in_price / entry_price if entry_price > 0 else 0.0,
            'proposed_action': proposed_action,
            'pred_proba': pred_proba
        })

    def resolve_labels(self, current_high: float, current_low: float, current_close: float = None, 
                       current_volume: float = 0.0, current_log_ret: float = 0.0,
                       current_timestamp: float = 0.0) -> List[Tuple[Dict[str, float], int, int, float, float, int, float, float, bool]]:
        """
        Evaluates independent virtual BUY and SELL trades to solve Asymmetry.
        Returns: (features, buy_label, sell_label, buy_ret, sell_ret, proposed_action, pred_proba, proposed_ret, is_executed)
        """
        resolved = []
        active = deque()
        if current_close is None: current_close = (current_high + current_low) / 2.0

        while self.buffer:
            trade = self.buffer.popleft()
            trade['age'] += 1
            trade['cum_vol'] += current_volume
            trade['cum_volatility'] += (current_log_ret ** 2)

            is_expired = False
            if self.horizon_type == 'TIME':
                if current_timestamp > 0 and trade.get('start_time', 0) > 0:
                    duration_sec = current_timestamp - trade['start_time']
                    if duration_sec >= (self.time_limit * 60): is_expired = True
                else:
                    if trade['age'] >= self.time_limit: is_expired = True
            elif self.horizon_type == 'VOLUME':
                thresh = self.horizon_threshold if self.horizon_threshold > 0 else current_volume * 100
                if trade['cum_vol'] >= thresh: is_expired = True
            elif self.horizon_type == 'VOLATILITY':
                thresh = self.horizon_threshold if self.horizon_threshold > 0 else 0.0005
                if trade['cum_volatility'] >= thresh: is_expired = True

            buy_status = 0
            sell_status = 0

            # Evaluate VIRTUAL BUY
            if current_low <= trade['buy_sl']: buy_status = -1
            elif current_high >= trade['buy_tp']: buy_status = 1
            
            # Evaluate VIRTUAL SELL
            if current_high >= trade['sell_sl']: sell_status = -1
            elif current_low <= trade['sell_tp']: sell_status = 1

            if buy_status != 0 or sell_status != 0 or is_expired:
                buy_ret = 0.0
                sell_ret = 0.0
                spread_pct = trade['spread_pct']
                comm_pct = trade['comm_pct']

                # Calculate Net Return (BUY) with Slippage
                if buy_status == 1: 
                    buy_ret = (trade['buy_tp'] - trade['entry']) / trade['entry'] - spread_pct - comm_pct
                elif buy_status == -1: 
                    buy_ret = (trade['buy_sl'] - trade['entry']) / trade['entry'] - spread_pct - comm_pct
                elif is_expired: 
                    buy_ret = (current_close - trade['entry']) / trade['entry'] - spread_pct - comm_pct

                # Calculate Net Return (SELL) with Slippage
                if sell_status == 1: 
                    sell_ret = (trade['entry'] - trade['sell_tp']) / trade['entry'] - spread_pct - comm_pct
                elif sell_status == -1: 
                    sell_ret = (trade['entry'] - trade['sell_sl']) / trade['entry'] - spread_pct - comm_pct
                elif is_expired: 
                    sell_ret = (trade['entry'] - current_close) / trade['entry'] - spread_pct - comm_pct

                # --- V20.18 FIX: DUAL-MODEL ASYMMETRY (TUPLE UNPACKING CURE) ---
                buy_label = 1 if buy_ret > 0 else 0
                sell_label = 1 if sell_ret > 0 else 0
                    
                proposed_action = trade.get('proposed_action', 0)
                pred_proba = trade.get('pred_proba', 0.5)
                proposed_ret = 0.0
                
                if proposed_action == 1:
                    proposed_ret = buy_ret
                elif proposed_action == -1:
                    proposed_ret = sell_ret

                is_executed = trade.get('is_executed', False) # V20.8 IPC Fix

                resolved.append((
                    trade['features'], 
                    buy_label,
                    sell_label,
                    buy_ret,
                    sell_ret,
                    proposed_action,
                    pred_proba,
                    proposed_ret,
                    is_executed
                ))
            else:
                active.append(trade)

        self.buffer = active
        return resolved

# --- 7. PROXIES & UTILS ---

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

    def get(self) -> float:
        if len(self.returns) < 5: return 0.001
        val = np.std(self.returns)
        return val if math.isfinite(val) else 0.001

@njit
def calculate_hurst(returns_ts):
    """
    V20.6 FIX: Calculates the Hurst Exponent correctly by integrating returns 
    into a random walk (price path) before computing lag standard deviations.
    """
    n = len(returns_ts)
    if n < 20: return 0.5
    
    # 1. Reconstruct the price path from log returns (Cumulative Sum)
    ts = np.cumsum(returns_ts)
    
    # 1b. Variance Check (Math Guard)
    if np.std(ts) < 1e-9: return 0.5
    
    lags = np.arange(2, 20, dtype=np.float64)
    tau = np.zeros(len(lags), dtype=np.float64)
    
    for i in range(len(lags)):
        lag = int(lags[i])
        diff = ts[lag:] - ts[:-lag]
        std_diff = np.std(diff)
        
        if std_diff < 1e-9:
            tau[i] = 1e-9
        else:
            tau[i] = std_diff
    
    log_lags = np.log(lags)
    log_tau = np.log(tau)
    
    N = float(len(lags))
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_tau)
    sum_xy = np.sum(log_lags * log_tau)
    sum_xx = np.sum(log_lags * log_lags)
    
    denominator = (N * sum_xx) - (sum_x * sum_x)
    
    if abs(denominator) < 1e-9:
        return 0.5
        
    slope = ((N * sum_xy) - (sum_x * sum_y)) / denominator
    
    hurst = slope
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

# --- 8. GLOBAL FEATURE ORCHESTRATOR ---

class OnlineFeatureEngineer:
    """
    Orchestrates all streaming indicators and quantitative models to build a cohesive feature vector.
    🚨 CRITICAL FIX: The ThreadPoolExecutor is now instantiated dynamically strictly based on `live_mode` 
    to prevent Optuna background thread contamination across independent Swarm Trials.
    """
    def __init__(self, window_size: int = 50, live_mode: bool = False):
        self.window_size = window_size
        
        # Core monitors
        self.vol_monitor = VolatilityMonitor(window=window_size)
        vpin_bucket = CONFIG.get('phoenix_strategy', {}).get('vpin_bucket_size', 1000)
        self.vpin_monitor = VPINMonitor(bucket_size=vpin_bucket)
        entropy_win = CONFIG.get('features', {}).get('entropy_window', 100)
        self.entropy_monitor = EntropyMonitor(window=entropy_win)
        
        # 🚨 CRITICAL FIX: Isolate HMM fitting. Use ThreadPoolExecutor ONLY in Live mode.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="RegimeHMM") if live_mode else None
        self.regime_detector = RegimeDetector(n_states=2, window=100, executor=self.executor)
        
        # Streaming features
        self.bb = StreamingBollingerBands(window=20)
        self.parkinson = StreamingParkinsonVolatility()
        self.amihud = StreamingAmihudLiquidity()
        self.rvol = StreamingRelativeVolume()
        self.aggressor = StreamingAggressorRatio()
        self.indicators = StreamingIndicators()
        self.micro_analyzer = MicrostructureAnalyzer()
        
        self.prev_price = None

    def update(self, price: float, timestamp: float, volume: float, high: float, low: float, 
               buy_vol: float, sell_vol: float, time_feats: dict, context_data: dict) -> Optional[dict]:
        
        log_ret = math.log(price / self.prev_price) if self.prev_price and self.prev_price > 0 else 0.0
        self.prev_price = price
        
        volatility = self.vol_monitor.update(log_ret)
        vpin = self.vpin_monitor.update(volume, price, buy_vol, sell_vol)
        entropy = self.entropy_monitor.update(price)
        regime = self.regime_detector.update(log_ret)
        
        bb_feats = self.bb.update(price)
        park_vol = self.parkinson.update(high, low)
        amihud_val = self.amihud.update(abs(log_ret), price, volume)
        rvol_val = self.rvol.update(timestamp)
        agg_ratio = self.aggressor.update(high, low, price)
        ind_feats = self.indicators.update(price, high, low)
        ofi = self.micro_analyzer.process_bar(buy_vol, sell_vol)
        
        features = {
            'log_ret': log_ret,
            'volatility': volatility,
            'vpin': vpin,
            'entropy': entropy,
            'hmm_regime': float(regime),
            'parkinson_vol': park_vol,
            'amihud': amihud_val,
            'rvol': rvol_val,
            'aggressor_ratio': agg_ratio,
            'ofi': ofi
        }
        
        features.update(bb_feats)
        features.update(ind_feats)
        features.update(time_feats)
        
        # Context Data (Clusters, MTF)
        if context_data:
            if 'cluster' in context_data:
                features['cluster_flow'] = context_data['cluster'].get('cluster_flow', 0.0)
                features['global_corr'] = context_data['cluster'].get('global_corr', 0.0)
            features = enrich_with_d1_data(features, context_data.get('d1', {}), price)
        
        return features
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Ensure safe unpickling without carrying over thread pool context
        if hasattr(self, 'regime_detector'):
            self.regime_detector.executor = None
            self.regime_detector._owns_executor = False