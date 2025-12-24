# =============================================================================
# FILENAME: shared/financial/transformer.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/financial/transformer.py
# DEPENDENCIES: pandas, numpy, math
# DESCRIPTION: Transforms raw data into ML-ready features (Cyclical Time, Clusters).
#
# FORENSIC REMEDIATION LOG (2025-12-23):
# 1. CLUSTER CONTEXT: Enhanced Builder to track Correlation Matrices.
# 2. FEATURE ENRICHMENT: Added 'global_correlation' to context vector.
# 3. ROBUSTNESS: Added safety checks for missing correlation data.
# =============================================================================
import numpy as np
import pandas as pd
import math
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

class TimeFeatureTransformer:
    """
    Transforms linear timestamps into cyclical features to help ML models
    learn seasonality (e.g., Session overlaps, Friday closes).
     
    Concepts:
    - Hour of Day (0-23) -> Sine/Cosine pair
    - Day of Week (0-6) -> Sine/Cosine pair
    """
    def __init__(self):
        self.seconds_per_day = 24 * 60 * 60
        self.seconds_per_week = 7 * 24 * 60 * 60

    def transform_scalar(self, timestamp: float) -> Dict[str, float]:
        """
        Transforms a single unix timestamp into cyclical features.
        Used by the Online Feature Engineer (Real-time).
        """
        # Create a datetime object to extract struct fields (UTC)
        dt = datetime.fromtimestamp(timestamp, timezone.utc)
        
        # 1. Hour of Day Cycle (Micro Seasonality)
        # We use seconds past midnight for higher resolution
        seconds_past_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
        day_progress = seconds_past_midnight / self.seconds_per_day
        
        hour_sin = math.sin(2 * math.pi * day_progress)
        hour_cos = math.cos(2 * math.pi * day_progress)
        
        # 2. Day of Week Cycle (Macro Seasonality)
        # 0=Mon, 6=Sun. We treat the week as a continuous cycle.
        week_progress = (dt.weekday() * self.seconds_per_day + seconds_past_midnight) / self.seconds_per_week
        
        day_sin = math.sin(2 * math.pi * week_progress)
        day_cos = math.cos(2 * math.pi * week_progress)
        
        return {
            'time_hour_sin': hour_sin,
            'time_hour_cos': hour_cos,
            'time_day_sin': day_sin,
            'time_day_cos': day_cos
        }

    def transform_batch(self, df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
        """
        Transforms a DataFrame column of timestamps.
        Used by the Research Pipeline (Backtesting).
        """
        if df.empty or time_col not in df.columns:
            return df
        
        # Ensure column is datetime
        series = pd.to_datetime(df[time_col], utc=True)
        
        # Helper for vectorized calc
        def get_cyclical(dt_series, period):
            return np.sin(2 * np.pi * dt_series / period), np.cos(2 * np.pi * dt_series / period)

        # Seconds past midnight
        seconds_past_midnight = series.dt.hour * 3600 + series.dt.minute * 60 + series.dt.second
        df['time_hour_sin'], df['time_hour_cos'] = get_cyclical(seconds_past_midnight, self.seconds_per_day)
        
        # Seconds past week start (Monday)
        seconds_past_week = series.dt.dayofweek * self.seconds_per_day + seconds_past_midnight
        df['time_day_sin'], df['time_day_cos'] = get_cyclical(seconds_past_week, self.seconds_per_week)
        
        return df

class ClusterContextBuilder:
    """
    Analyzes asset groups to inject 'Regime Context' into the feature set.
    Example: If all USD pairs are moving up, we are in a 'USD Strong' regime.
     
    UPDATED (2025-12-23): Now tracks Inter-Symbol Correlations to detect
    market-wide stress or idiosyncratic moves.
    """
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        # Map symbols to base/quote to identify clusters
        # e.g. 'EURUSD' -> {'base': 'EUR', 'quote': 'USD'}
        self.meta = self._parse_symbols(symbols)
        self.correlation_map = {} # Store latest correlations

    def _parse_symbols(self, symbols: List[str]) -> Dict[str, Dict[str, str]]:
        meta = {}
        for sym in symbols:
            if len(sym) == 6:
                meta[sym] = {'base': sym[:3], 'quote': sym[3:]}
            else:
                # Handle suffixes if present
                meta[sym] = {'base': sym[:3], 'quote': sym[3:6]}
        return meta

    def update_correlations(self, corr_matrix: pd.DataFrame):
        """
        Updates the internal correlation map from a pandas DataFrame (Correlation Matrix).
        Calculates the average correlation of each asset to the rest of the portfolio.
        """
        if corr_matrix.empty: return
        
        try:
            # Calculate mean correlation per symbol (excluding self-correlation of 1.0)
            # We subtract 1 and divide by N-1 to get avg of off-diagonals
            n = len(corr_matrix.columns)
            if n > 1:
                sums = corr_matrix.sum(axis=1) - 1.0
                avgs = sums / (n - 1)
                self.correlation_map = avgs.to_dict()
            else:
                self.correlation_map = {c: 0.0 for c in corr_matrix.columns}
        except Exception:
            pass # Fail silently, retain old map

    def calculate_cluster_coherence(self, returns_map: Dict[str, float]) -> Dict[str, float]:
        """
        Calculates the strength of specific currency moves based on the portfolio.
        Returns a dict of scores, e.g., {'USD': 0.8, 'EUR': -0.2}
         
        returns_map: Dict[symbol, log_return]
        """
        scores = {}
        counts = {}
        for sym, ret in returns_map.items():
            if sym not in self.meta: continue
            
            base = self.meta[sym]['base']
            quote = self.meta[sym]['quote']
            
            # Base moves proportional to return
            scores[base] = scores.get(base, 0.0) + ret
            counts[base] = counts.get(base, 0) + 1
            
            # Quote moves inverse to return
            scores[quote] = scores.get(quote, 0.0) - ret
            counts[quote] = counts.get(quote, 0) + 1
        
        # Normalize
        final_coherence = {}
        for ccy, score in scores.items():
            if counts[ccy] > 0:
                final_coherence[ccy] = score / counts[ccy]
            else:
                final_coherence[ccy] = 0.0
        
        return final_coherence

    def get_context_feature(self, symbol: str, coherence_map: Dict[str, float]) -> Dict[str, float]:
        """
        Returns a feature vector representing the Market Context for a specific pair.
        Includes:
        1. Cluster Flow ('Wind Behind the Back')
        2. Global Correlation (Systemic vs Idiosyncratic)
        """
        features = {'cluster_flow': 0.0, 'global_corr': 0.0}
        
        if symbol not in self.meta: return features
        
        # 1. Cluster Flow
        base = self.meta[symbol]['base']
        quote = self.meta[symbol]['quote']
        
        base_score = coherence_map.get(base, 0.0)
        quote_score = coherence_map.get(quote, 0.0)
        
        # If Base is strong (+) and Quote is weak (-), result is positive (Strong Buy context)
        features['cluster_flow'] = base_score - quote_score
        
        # 2. Correlation Context
        features['global_corr'] = self.correlation_map.get(symbol, 0.0)
        
        return features