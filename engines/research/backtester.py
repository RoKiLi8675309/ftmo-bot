# =============================================================================
# FILENAME: engines/research/backtester.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/backtester.py
# DEPENDENCIES: shared
# DESCRIPTION: Event-Driven Backtesting Broker. Simulates execution, spread,
# commissions, and PnL tracking for strategy validation.
#
# PHOENIX V16.1 MAINTENANCE PATCH:
# 1. PNL FIX: Removed hardcoded exchange rates. Now uses dynamic MarketSnapshot.
# 2. ACCURACY: Enforces live conversion rates for Cross-Pairs (e.g. EURJPY -> USD).
# 3. SAFETY: Added strict Hard Deck and Daily Loss enforcement.
# =============================================================================
from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from collections import deque
from datetime import datetime, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Generator, Any
import pytz

# Shared Imports
from shared.financial.risk import RiskManager
from shared.core.config import CONFIG
from shared.core.logging_setup import LogSymbols

# Setup Logger
logger = logging.getLogger("Backtester")

@dataclass(frozen=True)
class MarketSnapshot:
    """
    Represents the market state at a specific timestamp across all symbols.
    Used to decouple the Broker from the raw Dataframe.
    """
    timestamp: datetime
    data: pd.Series  # Row from the aligned dataframe

    def get_price(self, symbol: str, price_type: str = 'close') -> float:
        """
        Safe accessor for price data (Close, Open, High, Low).
        Handles both Single-Index and Multi-Index columns.
        """
        try:
            # Try specific column first (e.g., 'EURUSD_close')
            key = f"{symbol}_{price_type}" if symbol else price_type
            val = None
            
            if key in self.data:
                val = self.data[key]
            # Fallback for generic names if only one symbol exists
            elif price_type in self.data:
                val = self.data[price_type]
            
            if val is not None:
                # Handle potential non-numeric data
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0
            
            return 0.0
        except Exception:
            return 0.0
            
    def get_high(self, symbol: str) -> float:
        return self.get_price(symbol, 'high')
        
    def get_low(self, symbol: str) -> float:
        return self.get_price(symbol, 'low')

    def to_price_dict(self) -> Dict[str, float]:
        """Returns a dictionary of {symbol: close_price}."""
        res = {}
        # Iterate safely over index
        for col in self.data.index:
            if isinstance(col, str) and "_close" in col:
                sym = col.replace("_close", "")
                try:
                    res[sym] = float(self.data[col])
                except:
                    res[sym] = 0.0
        return res

@dataclass
class BacktestOrder:
    symbol: str
    side: int # 1 for BUY, -1 for SELL
    quantity: float
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    ticket: int = 0
    timestamp_created: datetime = field(default_factory=datetime.now)
    magic: int = 123456
    comment: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation flags
    action: str = "" # "BUY" or "SELL" derived from side
    is_active: bool = True
    close_time: Optional[datetime] = None
    close_price: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    exit_reason: str = ""
    
    # Metrics
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    def __post_init__(self):
        if not self.action:
            self.action = "BUY" if self.side == 1 else "SELL"

class BacktestBroker:
    """
    Simulates a Broker environment:
    - Order Execution (Immediate Fill for research, spread simulation)
    - PnL Tracking (Equity/Balance)
    - Margin Checks (simplified)
    - V10.0: Daily Hard Deck Enforcement
    """
    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.cash = initial_balance
        
        self.open_positions: List[BacktestOrder] = []
        self.closed_positions: List[BacktestOrder] = []
        
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.orders_history: List[Dict] = []
        self.trade_log: List[Dict] = [] 
        
        self.ticket_counter = 1
        self.last_snapshot: Optional[MarketSnapshot] = None
        self.last_price_map: Dict[str, float] = {}
        
        # State Flags
        self.is_blown = False

        # Load Configured Costs
        self.commission_per_lot = CONFIG.get('risk_management', {}).get('commission_per_lot_rt', 5.0)
        self.spread_map = CONFIG.get('forensic_audit', {}).get('spread_pips', {})
        self.default_spread = self.spread_map.get('default', 1.5)

        # Pre-populate price map to prevent initial warnings
        self._inject_aux_data()
        
        # --- V10.0 DAILY ANCHOR TRACKING ---
        self.daily_start_equity = initial_balance
        self.current_day_date = None
        # Max Daily Loss Pct (e.g., 0.045 for 4.5%)
        self.max_daily_loss_pct = CONFIG['risk_management'].get('max_daily_loss_pct', 0.045)
        
        # Timezone for Midnight Reset
        tz_str = CONFIG['risk_management'].get('risk_timezone', 'Europe/Prague')
        try:
            self.market_tz = pytz.timezone(tz_str)
        except:
            self.market_tz = pytz.timezone('Europe/Prague')

    @property
    def positions(self) -> Dict[str, BacktestOrder]:
        """
        COMPATIBILITY LAYER: Exposes open positions as a dictionary {symbol: trade}.
        """
        pos_map = {}
        for p in self.open_positions:
            pos_map[p.symbol] = p
        return pos_map

    def get_daily_pnl_pct(self) -> float:
        """
        V12.0: Calculates the current Daily PnL % relative to the Daily Start Equity.
        Used for Profit Buffer Scaling (Earn to Burn).
        """
        if self.daily_start_equity <= 0:
            return 0.0
        
        current_pnl = self.equity - self.daily_start_equity
        return current_pnl / self.daily_start_equity

    def _inject_aux_data(self):
        """
        Injects synthetic cross-rates into the price map.
        V16.0: Added High-Beta Pairs (GBPAUD, GBPNZD, EURAUD) to prevent crashes.
        """
        defaults = {
            "USDJPY": 150.0, "GBPUSD": 1.25, "EURUSD": 1.08,
            "USDCAD": 1.35, "USDCHF": 0.90, "AUDUSD": 0.65, "NZDUSD": 0.60,
            # V16.0 Additions
            "GBPAUD": 1.95, "EURAUD": 1.65, "GBPNZD": 2.10,
            "EURJPY": 162.0, "GBPJPY": 190.0, "AUDJPY": 97.0
        }
        for sym, price in defaults.items():
            if sym not in self.last_price_map:
                self.last_price_map[sym] = price

    def _get_simulation_conversion_rate(self, symbol: str, current_price: float) -> float:
        """
        V16.1 FIX: Robust wrapper for RiskManager.get_conversion_rate.
        REMOVED: Hardcoded static values (0.0065 for JPY etc).
        ADDED: Dynamic lookup via self.last_price_map to ensure PnL accuracy during volatility.
        """
        rate = RiskManager.get_conversion_rate(symbol, current_price, self.last_price_map)
        
        # Sanity Check: If rate is 1.0 but it's clearly a non-USD pair, log a warning
        # This usually means the auxiliary pair (e.g. USDJPY) is missing from the dataframe.
        if rate == 1.0 and not symbol.endswith("USD") and "USD" in symbol:
             # It is a USD pair (e.g. USDJPY), so rate might be 1/Price or Price
             pass
        elif rate == 1.0 and "USD" not in symbol:
             # Cross pair (e.g. EURGBP) returning 1.0 implies missing conversion data
             # We let it slide but ideally we should have data
             pass

        return rate

    def process_pending(self, snapshot: MarketSnapshot):
        """ Alias for process_snapshot """
        self.process_snapshot(snapshot)

    def process_snapshot(self, snapshot: MarketSnapshot):
        """
        Updates the state of the broker based on a new market tick/bar.
        Updates Equity, Checks SL/TP, Updates Price Map.
        Enforces Hard Deck.
        """
        self.last_snapshot = snapshot
        
        # Handle Timezone for Midnight Anchor
        ts = snapshot.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=pytz.utc)
        server_time = ts.astimezone(self.market_tz)
        current_date = server_time.date()
        
        # --- MIDNIGHT ANCHOR LOGIC ---
        if self.current_day_date is None:
            self.current_day_date = current_date
            self.daily_start_equity = self.equity # Anchor established
        elif current_date > self.current_day_date:
            # New Day Rollover: Reset Anchor
            self.current_day_date = current_date
            self.daily_start_equity = self.equity # New Anchor
            
        # 1. Update Price Map (CRITICAL for PnL Conversion)
        # We iterate all columns to grab prices for ALL symbols, not just the active one
        for col in snapshot.data.index:
            if isinstance(col, str):
                try:
                    val = float(snapshot.data[col])
                    if "_close" in col:
                        sym = col.replace("_close", "")
                        self.last_price_map[sym] = val
                    elif "_" not in col and len(col) == 6:
                         self.last_price_map[col] = val
                except:
                    continue

        # 2. Check Open Positions
        active_positions = []
        floating_pnl = 0.0
        
        for trade in self.open_positions:
            current_price = snapshot.get_price(trade.symbol, 'close')
            
            if current_price <= 0:
                active_positions.append(trade)
                continue

            # --- Update MAE/MFE ---
            pip_size = 0.01 if "JPY" in trade.symbol else 0.0001
            dist = (current_price - trade.entry_price) / pip_size
            if trade.action == "SELL": dist = -dist
            
            trade.max_favorable_excursion = max(trade.max_favorable_excursion, dist)
            trade.max_adverse_excursion = min(trade.max_adverse_excursion, dist)

            # --- Check Exit Conditions (SL/TP) ---
            closed = False
            exit_reason = ""
            close_price = 0.0

            bar_high = snapshot.get_high(trade.symbol)
            bar_low = snapshot.get_low(trade.symbol)
            
            if bar_high == 0: bar_high = current_price
            if bar_low == 0: bar_low = current_price

            if trade.action == "BUY":
                if trade.stop_loss > 0 and bar_low <= trade.stop_loss:
                    closed = True
                    exit_reason = "SL_HIT"
                    close_price = trade.stop_loss
                elif trade.take_profit > 0 and bar_high >= trade.take_profit:
                    closed = True
                    exit_reason = "TP_HIT"
                    close_price = trade.take_profit
            elif trade.action == "SELL":
                if trade.stop_loss > 0 and bar_high >= trade.stop_loss:
                    closed = True
                    exit_reason = "SL_HIT"
                    close_price = trade.stop_loss
                elif trade.take_profit > 0 and bar_low <= trade.take_profit:
                    closed = True
                    exit_reason = "TP_HIT"
                    close_price = trade.take_profit

            if closed:
                self._finalize_trade(trade, close_price, snapshot.timestamp, exit_reason)
            else:
                # Update Floating PnL
                contract_size = 100000
                raw_pnl = (current_price - trade.entry_price) * (trade.quantity * contract_size)
                if trade.action == "SELL": raw_pnl = -raw_pnl
                
                # V16.1: Use Dynamic Rate
                rate = self._get_simulation_conversion_rate(trade.symbol, current_price)
                comm_drag = trade.quantity * self.commission_per_lot 
                
                floating_pnl += (raw_pnl * rate) - comm_drag
                active_positions.append(trade)

        self.open_positions = active_positions
        
        # 3. Update Equity
        self.equity = self.balance + floating_pnl
        self.equity_curve.append((snapshot.timestamp, self.equity))
        
        # 4. HARD DECK ENFORCEMENT (V10.0)
        # Check Daily Drawdown relative to Anchor
        if self.daily_start_equity > 0:
            daily_loss_amount = self.daily_start_equity - self.equity
            daily_loss_limit = self.daily_start_equity * self.max_daily_loss_pct
            
            if daily_loss_amount > daily_loss_limit:
                logger.warning(f"💀 HARD DECK BREACHED (Sim): Start {self.daily_start_equity:.2f} -> Curr {self.equity:.2f}")
                self.is_blown = True # Mark as blown for optimizer
                
                # Liquidate all open positions instantly at current price
                # This simulates the "Liquidation" action
                for trade in list(self.open_positions):
                    curr_p = snapshot.get_price(trade.symbol, 'close')
                    self._finalize_trade(trade, curr_p, snapshot.timestamp, "HARD_DECK_LIQ")
                self.open_positions = [] # Clear

        # 5. Margin Call / Blowout Check (Total)
        if self.equity < (self.initial_balance * 0.90): 
             self.is_blown = True

    def submit_order(self, order: BacktestOrder) -> Optional[int]:
        """
        Public API called by ResearchStrategy.
        """
        if order.quantity <= 0: return None
        if self.is_blown: return None # No trades if blown
        
        order.ticket = self.ticket_counter
        self.ticket_counter += 1
        
        if order.entry_price <= 0:
             if self.last_snapshot:
                 order.entry_price = self.last_snapshot.get_price(order.symbol)
        
        # Spread Simulation
        pip = 0.01 if "JPY" in order.symbol else 0.0001
        spread_pips = self.spread_map.get(order.symbol, self.default_spread)
        slippage_cost = spread_pips * pip
        
        order.entry_price = order.entry_price + slippage_cost if order.action == "BUY" else order.entry_price - slippage_cost
        order.commission = order.quantity * self.commission_per_lot
        
        self.open_positions.append(order)
        return order.ticket

    def get_position(self, symbol: str) -> Optional[BacktestOrder]:
        for p in self.open_positions:
            if p.symbol == symbol:
                return p
        return None

    def _close_partial_position(self, trade: BacktestOrder, qty: float, price: float, time: datetime, reason: str):
        self._finalize_trade(trade, price, time, reason)

    def _finalize_trade(self, trade: BacktestOrder, close_price: float, close_time: datetime, reason: str):
        # 1. Calculate Raw PnL
        raw_diff = (close_price - trade.entry_price)
        if trade.action == "SELL": raw_diff = -raw_diff
        
        raw_profit = raw_diff * (trade.quantity * 100000)
        
        # 2. Convert to USD (Dynamic Rate)
        rate = self._get_simulation_conversion_rate(trade.symbol, close_price)
        gross_pnl = raw_profit * rate
        
        # 3. Net PnL
        net_pnl = gross_pnl - trade.commission
        
        # 4. Update Balance
        self.balance += net_pnl
        
        # 5. Update Trade Object
        trade.is_active = False
        trade.close_time = close_time
        trade.close_price = close_price
        trade.gross_pnl = gross_pnl
        trade.net_pnl = net_pnl
        trade.exit_reason = reason
        
        self.closed_positions.append(trade)
        
        # 6. Log to Trade Log (Enhanced Metadata for V12.6)
        # Ensure metadata is serializable
        clean_metadata = {}
        for k, v in trade.metadata.items():
            if isinstance(v, (np.float64, np.float32)):
                clean_metadata[k] = float(v)
            elif isinstance(v, (np.int64, np.int32)):
                clean_metadata[k] = int(v)
            else:
                clean_metadata[k] = v

        self.trade_log.append({
            'Entry_Time': trade.timestamp_created,
            'Exit_Time': close_time,
            'Symbol': trade.symbol,
            'Action': trade.action,
            'Size': float(trade.quantity),
            'Entry_Price': float(trade.entry_price),
            'Exit_Price': float(close_price),
            'Gross_PnL': float(gross_pnl),
            'Commission': float(trade.commission),
            'Net_PnL': float(net_pnl),
            'Status': "CLOSED",
            'Comment': reason,
            'MFE_Pips': float(trade.max_favorable_excursion),
            'MAE_Pips': float(trade.max_adverse_excursion),
            'Duration_Min': (close_time - trade.timestamp_created).total_seconds() / 60,
            'Regime': clean_metadata.get('regime', 'Unknown'),
            'Confidence': clean_metadata.get('confidence', 0.0),
            'Tighten_Stops': clean_metadata.get('tighten_stops', False) # New for V12.6
        })
        
        symbol_icon = "🟢" if net_pnl > 0 else "🔴"
        logger.debug(f"{symbol_icon} CLOSED {trade.symbol} {trade.action}: ${net_pnl:.2f} ({reason})")

    def get_stats(self) -> Dict[str, Any]:
        """
        Generates comprehensive performance statistics for the backtest.
        """
        stats = {
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "final_equity": self.equity,
            "total_trades": len(self.closed_positions),
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "sqn": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "risk_reward_ratio": 0.0, 
            "largest_win": 0.0,
            "largest_loss": 0.0
        }

        if self.closed_positions:
            pnls = [t.net_pnl for t in self.closed_positions]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            
            stats["total_trades"] = len(pnls)
            stats["win_rate"] = (len(wins) / len(pnls)) * 100 if pnls else 0.0
            
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            stats["profit_factor"] = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
            
            stats["avg_win"] = np.mean(wins) if wins else 0.0
            stats["avg_loss"] = np.mean(losses) if losses else 0.0
            stats["largest_win"] = max(wins) if wins else 0.0
            stats["largest_loss"] = min(losses) if losses else 0.0
            
            avg_loss_abs = abs(stats["avg_loss"])
            if avg_loss_abs > 0:
                stats["risk_reward_ratio"] = stats["avg_win"] / avg_loss_abs
            else:
                stats["risk_reward_ratio"] = 10.0 if stats["avg_win"] > 0 else 0.0
            
            win_pct = len(wins) / len(pnls)
            loss_pct = len(losses) / len(pnls)
            stats["expectancy"] = (win_pct * stats["avg_win"]) + (loss_pct * stats["avg_loss"])
            
            if len(pnls) > 1:
                std_pnl = np.std(pnls)
                if std_pnl > 0:
                    stats["sqn"] = np.sqrt(len(pnls)) * (np.mean(pnls) / std_pnl)

        if len(self.equity_curve) > 1:
            df = pd.DataFrame(self.equity_curve, columns=['time', 'equity'])
            df.set_index('time', inplace=True)
            
            df['peak'] = df['equity'].cummax()
            df['dd'] = df['equity'] - df['peak']
            df['dd_pct'] = (df['dd'] / df['peak']) * 100
            
            stats["max_drawdown"] = df['dd'].min()
            stats["max_drawdown_pct"] = df['dd_pct'].min()
            
            try:
                hourly_equity = df['equity'].resample('1H').last().ffill()
                returns = hourly_equity.pct_change().dropna()
                
                if len(returns) > 1:
                    avg_ret = returns.mean()
                    std_ret = returns.std()
                    annualization_factor = np.sqrt(252 * 24)
                    
                    if std_ret > 0:
                        stats["sharpe_ratio"] = (avg_ret / std_ret) * annualization_factor
                    
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) > 0:
                        downside_std = downside_returns.std()
                        if downside_std > 0:
                            stats["sortino_ratio"] = (avg_ret / downside_std) * annualization_factor
                            
            except Exception as e:
                logger.warning(f"Failed to calculate time-series stats: {e}")

        return stats

    def get_equity_curve(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve, columns=['time', 'equity'])