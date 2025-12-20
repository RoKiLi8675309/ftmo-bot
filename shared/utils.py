# =============================================================================
# FILENAME: shared/utils.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/utils.py
# DEPENDENCIES: typing
# DESCRIPTION: General utilities and helpers (Precision, Diagnostics).
# CRITICAL: Python 3.9 Compatible.
# =============================================================================
from typing import Any, Optional, List, Dict
import math

class PrecisionGuard:
    """
    Ensures prices and volumes are rounded to the correct precision
    required by the broker (MetaTrader 5).
    """
    @staticmethod
    def get_digits(symbol: str, symbol_info: Any = None) -> int:
        """
        Returns the number of decimal places for a symbol.
        Prioritizes MT5 symbol_info if available.
        AUDIT FIX: Hardened heuristics for Indices and Crypto.
        """
        # 1. Authoritative Source
        if symbol_info and hasattr(symbol_info, 'digits'):
            return symbol_info.digits
       
        # 2. Fallback Heuristics
        s = symbol.upper()
       
        # JPY Pairs (Usually 3 digits, e.g. 145.123)
        if "JPY" in s:
            return 3
           
        # Gold/Silver (Usually 2 digits, e.g. 2045.50)
        if "XAU" in s or "XAG" in s:
            return 2
       
        # Indices (Usually 1 or 2 digits)
        # US30, GER40, SPX500, NAS100
        if any(idx in s for idx in ["US30", "GER30", "GER40", "FRA40", "UK100", "JP225"]):
            return 1
        if any(idx in s for idx in ["SPX", "NAS", "US500", "US100"]):
            return 2

        # Crypto (Usually 2 or 5 digits depending on price, assume 2 for major pairs like BTCUSD)
        if "BTC" in s or "ETH" in s:
            return 2

        # Most Forex pairs are 5 digits (e.g. 1.05234)
        return 5

    @staticmethod
    def normalize_price(price: float, symbol: str, symbol_info: Any = None) -> float:
        """
        Rounds price to the correct number of digits.
        """
        digits = PrecisionGuard.get_digits(symbol, symbol_info)
        return round(price, digits)

    @staticmethod
    def normalize_volume(volume: float, step: float = 0.01, min_vol: float = 0.01, max_vol: float = 100.0) -> float:
        """
        Rounds volume to the nearest step and clamps to limits.
        """
        if step == 0: return volume
       
        # Quantize
        steps = round(volume / step)
        quantized = steps * step
       
        # Clamp
        quantized = max(min_vol, min(quantized, max_vol))
       
        return round(quantized, 2)

class SystemDiagnose:
    """
    Container for diagnostic logic used in diagnose.py.
    Provides integrity checks for trade logs and data structures.
    """
    def __init__(self, trade_log: Any):
        # Handle cases where trade_log might be a list or DataFrame
        if hasattr(trade_log, 'copy'):
            self.df = trade_log.copy()
        else:
            self.df = trade_log

    def check_integrity(self) -> bool:
        """
        Checks if the data structure contains valid trade information.
        """
        # If it's a pandas DataFrame or similar
        if hasattr(self.df, 'empty'):
            return not self.df.empty
        # If it's a list (e.g., list of dicts)
        if isinstance(self.df, list):
            return len(self.df) > 0
        return False

    def get_summary(self) -> Dict[str, Any]:
        """Returns a basic summary of the trade log."""
        if not self.check_integrity():
            return {"status": "Empty"}
       
        return {"status": "Valid", "count": len(self.df)}