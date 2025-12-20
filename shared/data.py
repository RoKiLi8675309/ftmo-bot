# =============================================================================
# FILENAME: shared/data.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/data.py
# DEPENDENCIES: pandas, numpy, psycopg2 (optional), sqlalchemy
# DESCRIPTION: Data loading with SQLAlchemy Engine & Strict Empty Checks.
# AUDIT REMEDIATION:
#   - LG-1: Implemented Lee-Ready Algorithm for Tick Rule (No more 50/50 split).
#   - Ensures neutral ticks inherit direction from the aggressor.
# =============================================================================
from __future__ import annotations
import warnings
import pandas as pd
import numpy as np
import pytz
import sys
from datetime import datetime
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

    def process_tick(self, price: float, volume: float, timestamp: float) -> Optional[VolumeBar]:
        """
        Ingests a tick. Returns a VolumeBar if threshold reached, else None.
        Handles overshoot via Carry-Over (Recursion for large block trades).
        """
        # 1. Update Prices
        if self.open_price is None:
            self.open_price = price
       
        self.high_price = max(self.high_price, price)
        self.low_price = min(self.low_price, price)
        self.close_price = price
        self.last_ts = timestamp
       
        # --- TICK RULE LOGIC (LEE-READY) ---
        # AUDIT FIX: LG-1 Implemented memory for neutral ticks
        if self.last_price is not None:
            if price > self.last_price:
                direction = 1  # Buy
                self.last_tick_direction = 1
            elif price < self.last_price:
                direction = -1  # Sell
                self.last_tick_direction = -1
            else:
                # Neutral: Continue previous direction (Lee-Ready)
                direction = self.last_tick_direction
        else:
            direction = 0  # First tick
            self.last_tick_direction = 0

        self.last_price = price
        # -----------------------

        # 3. Handle Carry-Over Logic
        # Calculate how much volume fits in the CURRENT bar
        remaining_capacity = self.threshold - self.current_volume
       
        if volume < remaining_capacity:
            # Case A: Tick fits entirely in current bar
            self._accumulate(price, volume, direction)
            return None
        else:
            # Case B: Tick fills the bar and spills over
            # 1. Fill the current bar
            self._accumulate(price, remaining_capacity, direction)
           
            # 2. Create the Bar
            vwap = self.vwap_sum / self.current_volume if self.current_volume > 0 else price
           
            bar = VolumeBar(
                timestamp=self.last_ts,
                open=self.open_price,
                high=self.high_price,
                low=self.low_price,
                close=self.close_price,
                volume=self.current_volume,
                vwap=vwap,
                tick_count=self.ticks_in_bar,
                # Add extended attributes for VPIN
                buy_vol=self.current_buy_vol,
                sell_vol=self.current_sell_vol
            )
           
            # 3. Calculate Spillover
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
           
            self._accumulate(price, spillover_vol, direction)
           
            return bar

    def _accumulate(self, price: float, volume: float, direction: int):
        """Helper to update internal counters."""
        self.current_volume += volume
        self.vwap_sum += (price * volume)
        self.ticks_in_bar += 1
       
        if direction == 1:
            self.current_buy_vol += volume
        elif direction == -1:
            self.current_sell_vol += volume
        else:
            # Should rarely happen with Lee-Ready, but fallback to 50/50 if truly no history
            self.current_buy_vol += volume / 2
            self.current_sell_vol += volume / 2

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
                # Execute purely to check existence/validity, but read_sql handles the fetch
                # Optimization: pd.read_sql with params
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
            # Raise strict error so worker fails immediately rather than silently
            raise ValueError(f"CRITICAL: Database query returned 0 rows for {symbol}. Check DB population!")

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
        # If we loaded Ticks, we likely have 'price' but NOT 'close'/'open'/'high'/'low'.
        # The Backtester expects OHLC. We must map 'price' to these columns.
        if 'price' in df.columns:
            # Check and fill missing OHLC columns
            for col in ['open', 'high', 'low', 'close']:
                if col not in df.columns:
                    df[col] = df['price']
        # ------------------------------------------------

        # --- INDEX FIX: Set Index to 'time' ---
        # This ensures iterating the DataFrame yields a DatetimeIndex,
        # preventing "int object has no attribute timestamp" errors in Strategy.
        df.set_index('time', inplace=True, drop=False)
        # --------------------------------------

        return df
    except ImportError:
        print("ERROR: sqlalchemy/psycopg2 not installed. Cannot load real data.")
        return pd.DataFrame()
    except Exception as e:
        # Re-raise ValueError so it propagates to the Research pipeline
        if "CRITICAL" in str(e):
            raise e
        print(f"ERROR: Database load failed for {symbol}: {e}")
        return pd.DataFrame()

def batch_generate_volume_bars(tick_df: pd.DataFrame, volume_threshold: float = 1000) -> List[Dict[str, Any]]:
    """
    Offline batch processor for converting Tick DF to Volume Bar List.
    FIX: Now correctly calculates Buy/Sell Volume using Lee-Ready Tick Rule.
    """
    bars = []
    current_vol = 0.0
    current_buy_vol = 0.0
    current_sell_vol = 0.0
   
    open_p = None
    high_p = -float('inf')
    low_p = float('inf')
    vwap_sum = 0.0
    tick_count = 0
   
    # State for Tick Rule
    last_price = None
    last_direction = 0
   
    for row in tick_df.itertuples():
        price = getattr(row, 'price', getattr(row, 'close', None))
        vol = getattr(row, 'volume', 1.0)
        ts = getattr(row, 'Index', getattr(row, 'time', None))
       
        if price is None:
            continue
       
        # --- TICK RULE LOGIC (LEE-READY) ---
        if last_price is not None:
            if price > last_price:
                direction = 1  # Buy
                last_direction = 1
            elif price < last_price:
                direction = -1  # Sell
                last_direction = -1
            else:
                # Neutral: Inherit previous
                direction = last_direction
        else:
            direction = 0
            last_direction = 0
       
        last_price = price
        # -----------------------

        if open_p is None:
            open_p = price
        high_p = max(high_p, price)
        low_p = min(low_p, price)
       
        current_vol += vol
       
        # Accumulate Buy/Sell Vol
        if direction == 1:
            current_buy_vol += vol
        elif direction == -1:
            current_sell_vol += vol
        else:
            current_buy_vol += vol / 2
            current_sell_vol += vol / 2
        
        vwap_sum += (price * vol)
        tick_count += 1
       
        if current_vol >= volume_threshold:
            if isinstance(ts, (datetime, pd.Timestamp)):
                ts_val = ts.timestamp()
            else:
                ts_val = float(ts)
               
            bars.append({
                'timestamp': ts_val,
                'open': open_p,
                'high': high_p,
                'low': low_p,
                'close': price,
                'volume': current_vol,
                'vwap': vwap_sum / current_vol if current_vol > 0 else price,
                'tick_count': tick_count,
                # CRITICAL FIX: Export Split Volume for VPIN
                'buy_vol': current_buy_vol,
                'sell_vol': current_sell_vol
            })
           
            # Reset
            current_vol = 0.0
            current_buy_vol = 0.0
            current_sell_vol = 0.0
            open_p = None
            high_p = -float('inf')
            low_p = float('inf')
            vwap_sum = 0.0
            tick_count = 0
           
    return bars