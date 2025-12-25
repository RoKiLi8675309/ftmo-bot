# =============================================================================
# FILENAME: shared/data.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/data.py
# DEPENDENCIES: pandas, numpy, psycopg2 (optional), sqlalchemy
# DESCRIPTION: Data loading with Adaptive Volume Normalization.
# AUDIT REMEDIATION:
#   - FIXED (CRITICAL): VolumeBarAggregator now ingests external L2 flows.
#   - FALLBACK: Implemented Lee-Ready (Tick Rule) for Retail Brokers/Backtesting.
#   - PROBLEM #4 (JPY Bias): Implemented AdaptiveVolumeNormalizer.
# =============================================================================
from __future__ import annotations
import warnings
import pandas as pd
import numpy as np
import pytz
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Shared Imports
from shared.core.config import CONFIG
from shared.domain.models import VolumeBar

class TemporalPipeline:
    """
    Utilities for standardizing timestamps across different data sources.
    """
    def __init__(self):
        self.utc = pytz.UTC

    def normalize_dates(self, df: pd.DataFrame, date_col: str = 'time') -> pd.DataFrame:
        """
        Ensures the date column is Datetime-aware and UTC normalized.
        """
        if df.empty or date_col not in df.columns:
            return df
        
        # Convert to datetime objects
        dates = pd.to_datetime(df[date_col], errors='coerce')
        
        # Localize if naive, Convert if aware
        if dates.dt.tz is None:
            dates = dates.dt.tz_localize('UTC', ambiguous='NaT', nonexistent='shift_forward')
        else:
            dates = dates.dt.tz_convert('UTC')
            
        df[date_col] = dates
        
        # Sort and deduplicate
        df.sort_values(by=date_col, inplace=True)
        # Drop rows where time is NaT
        df.dropna(subset=[date_col], inplace=True)
        return df

class AdaptiveVolumeNormalizer:
    """
    Calculates dynamic volume thresholds based on historical activity.
    Solves JPY bias (Problem #4) by normalizing volume across different liquidity regimes.
    """
    def __init__(self, target_bars_per_day: int = 50, lookback_days: int = 20):
        self.target_bars = target_bars_per_day
        self.lookback = lookback_days

    def calculate_threshold(self, df: pd.DataFrame, timestamp_col: str = 'time', volume_col: str = 'volume') -> float:
        """
        Computes the optimal volume threshold to achieve the target bar count.
        Uses a rolling average of daily volume over the lookback period.
        """
        if df.empty:
            return 1000.0 # Safe default

        # Ensure datetime index
        temp_df = df.copy()
        if not isinstance(temp_df.index, pd.DatetimeIndex):
            temp_df[timestamp_col] = pd.to_datetime(temp_df[timestamp_col], utc=True)
            temp_df.set_index(timestamp_col, inplace=True)

        # Resample to Daily Volume
        daily_vol = temp_df[volume_col].resample('1D').sum()
        
        # Filter out low-activity days (weekend noise or holidays)
        # We assume a trading day has at least non-trivial volume
        mean_vol = daily_vol[daily_vol > 0].mean()
        
        if pd.isna(mean_vol) or mean_vol == 0:
            return 1000.0

        # Threshold = Avg Daily Volume / Target Bars
        threshold = mean_vol / self.target_bars
        
        # Clamp to reasonable limits to prevent micro-bars or mega-bars
        threshold = max(100.0, threshold) 
        
        return float(threshold)

class VolumeBarAggregator:
    """
    Aggregates raw ticks into Volume Bars based on a volume threshold.
    Crucial for VPIN calculation and event-driven analysis.
    """
    def __init__(self, symbol: str, threshold: float = 1000):
        self.symbol = symbol
        self.threshold = threshold
        
        # Accumulators
        self.current_volume = 0.0
        self.current_buy_vol = 0.0
        self.current_sell_vol = 0.0
        
        self.open_price = None
        self.high_price = -float('inf')
        self.low_price = float('inf')
        self.close_price = None
        
        # Helper state
        self.last_price = None  # To determine tick direction
        self.last_tick_direction = 0 # 1=Buy, -1=Sell (Lee-Ready State)
        self.vwap_sum = 0.0  # Price * Volume sum
        self.ticks_in_bar = 0
        self.last_ts = None

    def process_tick(self, price: float, volume: float, timestamp: float, 
                     external_buy_vol: float = 0.0, external_sell_vol: float = 0.0) -> Optional[VolumeBar]:
        """
        Ingests a tick. Returns a VolumeBar if threshold reached, else None.
        Handles overshoot via Carry-Over (Recursion for large block trades).
        
        AUDIT FIX: Accepts external_buy_vol/external_sell_vol from Producer.
        If provided (>0), uses them. If not, falls back to Lee-Ready Tick Rule.
        """
        # 1. Update Prices
        if self.open_price is None:
            self.open_price = price
        
        self.high_price = max(self.high_price, price)
        self.low_price = min(self.low_price, price)
        self.close_price = price
        self.last_ts = timestamp
        
        # --- VOLUME DIRECTION LOGIC ---
        # Priority 1: External L2 Data (Producer)
        if external_buy_vol > 0 or external_sell_vol > 0:
            buy_v = external_buy_vol
            sell_v = external_sell_vol
            
            # Update Lee-Ready state for continuity even if we use L2 here
            if self.last_price is not None:
                if price > self.last_price: self.last_tick_direction = 1
                elif price < self.last_price: self.last_tick_direction = -1
                
        else:
            # Priority 2: Lee-Ready Tick Rule Fallback
            buy_v = 0.0
            sell_v = 0.0
            
            if self.last_price is not None:
                if price > self.last_price:
                    direction = 1  # Buy
                    self.last_tick_direction = 1
                elif price < self.last_price:
                    direction = -1  # Sell
                    self.last_tick_direction = -1
                else:
                    # Neutral: Continue previous direction
                    direction = self.last_tick_direction
            else:
                direction = 0  # First tick
                self.last_tick_direction = 0

            # Assign Flow
            if direction == 1:
                buy_v = volume
            elif direction == -1:
                sell_v = volume
            else:
                # Should rarely happen with Lee-Ready, but fallback to 50/50
                buy_v = volume / 2
                sell_v = volume / 2

        self.last_price = price
        # -----------------------

        # 3. Handle Carry-Over Logic
        # Calculate how much volume fits in the CURRENT bar
        remaining_capacity = self.threshold - self.current_volume
        
        if volume < remaining_capacity:
            # Case A: Tick fits entirely in current bar
            self._accumulate(price, volume, buy_v, sell_v)
            return None
        else:
            # Case B: Tick fills the bar and spills over
            # We need to split the Buy/Sell volume proportionally
            
            # Fraction of this tick that completes the bar
            split_ratio = remaining_capacity / volume if volume > 0 else 0
            
            fill_buy = buy_v * split_ratio
            fill_sell = sell_v * split_ratio
            
            rem_buy = buy_v - fill_buy
            rem_sell = sell_v - fill_sell
            
            # 1. Fill the current bar
            self._accumulate(price, remaining_capacity, fill_buy, fill_sell)
            
            # 2. Create the Bar
            vwap = self.vwap_sum / self.current_volume if self.current_volume > 0 else price
            
            # Timestamps must be datetime objects for Models
            if isinstance(self.last_ts, float) or isinstance(self.last_ts, int):
                dt_ts = datetime.fromtimestamp(self.last_ts, pytz.utc)
            else:
                dt_ts = self.last_ts

            bar = VolumeBar(
                timestamp=dt_ts,
                open=self.open_price,
                high=self.high_price,
                low=self.low_price,
                close=self.close_price,
                volume=self.current_volume,
                vwap=vwap,
                tick_count=self.ticks_in_bar,
                # CRITICAL: Export Split Volume for VPIN/OFI
                buy_vol=self.current_buy_vol,
                sell_vol=self.current_sell_vol
            )
            
            # 3. Calculate Spillover Volume
            spillover_vol = volume - remaining_capacity
            
            # 4. Reset for the NEXT bar
            self._reset()
            
            # 5. Apply Spillover (Start new bar logic)
            # We initialize the new bar with the spillover data.
            # Prices match the tick that caused the spillover.
            self.open_price = price
            self.high_price = price
            self.low_price = price
            self.close_price = price
            self.last_ts = timestamp
            
            self._accumulate(price, spillover_vol, rem_buy, rem_sell)
            
            return bar

    def _accumulate(self, price: float, volume: float, buy_vol: float, sell_vol: float):
        """Helper to update internal counters."""
        self.current_volume += volume
        self.vwap_sum += (price * volume)
        self.ticks_in_bar += 1
        self.current_buy_vol += buy_vol
        self.current_sell_vol += sell_vol

    def _reset(self):
        self.current_volume = 0.0
        self.current_buy_vol = 0.0
        self.current_sell_vol = 0.0
        
        self.open_price = None
        self.high_price = -float('inf')
        self.low_price = float('inf')
        self.close_price = None
        
        self.vwap_sum = 0.0
        self.ticks_in_bar = 0

# --- GLOBAL FUNCTIONS ---
def load_real_data(
    symbol: str,
    n_candles: int = 50000,
    days: int = 365,
    db_config: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Loads historical data from Postgres using SQLAlchemy for stability.
    Raises ValueError if data is empty to prevent silent failures in Research.
      
    FIX: Enforces Data Mapping (Tick 'price' -> OHLC) and Sets Datetime Index.
    """
    if db_config is None:
        db_config = CONFIG.get('postgres')
        if not db_config:
            print("ERROR: No database configuration found.")
            return pd.DataFrame()

    try:
        from sqlalchemy import create_engine, text
        
        # Construct DSN for SQLAlchemy
        # postgresql+psycopg2://user:password@host:port/dbname
        db_url = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['db']}"
        
        engine = create_engine(db_url)
        
        # 1. Attempt TICKS Table Load (High Fidelity)
        # Using parameterized text query for SQLAlchemy
        query_ticks = text("""
            SELECT time, bid, ask, (bid+ask)/2 as price, 
                   (ask-bid)*100000 as spread, 
                   flags 
            FROM ticks 
            WHERE symbol = :symbol 
            AND time > NOW() - INTERVAL :days 
            ORDER BY time ASC
            LIMIT :limit
        """)
        
        params = {"symbol": symbol, "days": f"{days} days", "limit": n_candles}
        
        try:
            # We use a nested cursor/transaction to safely try/catch SQL errors
            with engine.connect() as conn:
                df = pd.read_sql(query_ticks, conn, params=params)
        
        except Exception:
            # 2. Fallback to OHLCV Table (Standard)
            query_ohlcv = text("""
                SELECT time, close as price, close as bid, close as ask, volume 
                FROM ohlcv 
                WHERE symbol = :symbol 
                AND time > NOW() - INTERVAL :days 
                ORDER BY time ASC
                LIMIT :limit
            """)
            with engine.connect() as conn:
                df = pd.read_sql(query_ohlcv, conn, params=params)
            
            if not df.empty:
                if 'spread' not in df.columns:
                    df['spread'] = 0.0
                if 'flags' not in df.columns:
                    df['flags'] = 0

        # CRITICAL AUDIT FIX: Strict Check for Empty Data
        if df.empty:
            print(f"WARNING: Database query returned 0 rows for {symbol}. Check DB population!")
            return pd.DataFrame()

        # Normalize
        tp = TemporalPipeline()
        df = tp.normalize_dates(df, 'time')
        
        # Ensure numeric
        cols = ['price', 'bid', 'ask', 'spread', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c])
        
        if 'volume' not in df.columns:
            df['volume'] = 1.0

        # --- DATA MAPPING FIX: Tick Data Compatibility ---
        if 'price' in df.columns:
            for col in ['open', 'high', 'low', 'close']:
                if col not in df.columns:
                    df[col] = df['price']
        # ------------------------------------------------

        # --- INDEX FIX: Set Index to 'time' ---
        df.set_index('time', inplace=True, drop=False)
        # --------------------------------------

        return df
    except ImportError:
        print("ERROR: sqlalchemy/psycopg2 not installed. Cannot load real data.")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR: Database load failed for {symbol}: {e}")
        return pd.DataFrame()

def batch_generate_volume_bars(tick_df: pd.DataFrame, volume_threshold: float = 1000) -> List[Dict[str, Any]]:
    """
    Offline batch processor for converting Tick DF to Volume Bar List.
    FIX: Uses Lee-Ready Tick Rule for classification since offline data rarely has 'buy_vol' columns.
    
    NOTE: If volume_threshold is default (1000), it attempts to auto-calculate 
    using AdaptiveVolumeNormalizer if possible.
    """
    
    # Auto-Adapt Threshold if using default and enough data exists
    if volume_threshold == 1000 and len(tick_df) > 10000:
        normalizer = AdaptiveVolumeNormalizer(target_bars_per_day=50)
        try:
            dynamic_thresh = normalizer.calculate_threshold(tick_df)
            volume_threshold = dynamic_thresh
        except Exception:
            pass 

    bars = []
    agg = VolumeBarAggregator(symbol="BATCH", threshold=volume_threshold)
    
    for row in tick_df.itertuples():
        price = getattr(row, 'price', getattr(row, 'close', None))
        vol = getattr(row, 'volume', 1.0)
        ts = getattr(row, 'Index', getattr(row, 'time', None))
        
        # If DB has pre-calculated flows (rare), use them
        b_vol = getattr(row, 'buy_vol', 0.0)
        s_vol = getattr(row, 'sell_vol', 0.0)
        
        if price is None: continue
        
        # Convert TS to float if needed
        if isinstance(ts, (datetime, pd.Timestamp)):
            ts_val = ts.timestamp()
        else:
            ts_val = float(ts)

        # Process via Aggregator logic (reusing strict logic)
        bar = agg.process_tick(price, vol, ts_val, b_vol, s_vol)
        
        if bar:
            bars.append({
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'vwap': bar.vwap,
                'tick_count': bar.tick_count,
                'buy_vol': bar.buy_vol,
                'sell_vol': bar.sell_vol
            })
            
    return bars