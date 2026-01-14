# =============================================================================
# FILENAME: shared/domain/models.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/domain/models.py
# DEPENDENCIES: dataclasses, typing, datetime
# DESCRIPTION: Core data structures (Trades, Bars, Events) used across the system.
# CRITICAL: Python 3.9 Compatible.
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class TradeContext:
    """
    Encapsulates the context required to calculate risk and position size.
    """
    symbol: str
    price: float
    stop_loss_price: float
    account_equity: float
    account_currency: str = "USD"
    win_rate: float = 0.55
    risk_reward_ratio: float = 1.0
    avg_win: float = 1.0
    avg_loss: float = 0.5

@dataclass
class Trade:
    """
    Represents a trade intent or an active trade.
    Passed from Linux (Logic) to Windows (Execution).
    """
    symbol: str
    action: str # "BUY" or "SELL"
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    comment: str
    entry_type: str = "MARKET"
    magic_number: int = 0
    ticket: int = 0
    
    # Advanced Trade Management
    limit_offset_atr: float = 0.5
    trail_enabled: bool = True
    trail_atr_mult: float = 1.0
    breakeven_at_rr: float = 1.0
    scale_out_levels: List[Tuple[float, float]] = field(default_factory=lambda: [(1.0, 0.5), (2.0, 1.0)])

@dataclass
class NewsEvent:
    """
    Represents a macroeconomic event from the economic calendar.
    """
    title: str
    country: str
    time_utc: datetime
    impact: str
    forecast: str = ""
    previous: str = ""

    @property
    def is_high_impact(self) -> bool:
        return self.impact.lower() == 'high'

    def to_dict(self) -> Dict[str, Any]:
        """Serialization helper for Redis/JSON."""
        return {
            'title': self.title,
            'country': self.country,
            'time_utc': self.time_utc.isoformat(),
            'impact': self.impact,
            'forecast': self.forecast,
            'previous': self.previous
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'NewsEvent':
        """Deserialization helper."""
        return NewsEvent(
            title=data['title'],
            country=data['country'],
            time_utc=datetime.fromisoformat(data['time_utc']),
            impact=data['impact'],
            forecast=data.get('forecast', ''),
            previous=data.get('previous', '')
        )

@dataclass
class VolumeBar:
    """
    Represents an aggregated bar based on volume rather than time.
    Crucial for VPIN calculation and event-driven analysis.
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    tick_count: int
    # Extended attributes for VPIN
    buy_vol: float = 0.0
    sell_vol: float = 0.0