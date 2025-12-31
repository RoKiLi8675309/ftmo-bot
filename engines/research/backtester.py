# =============================================================================
# FILENAME: engines/research/backtester.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/backtester.py
# DEPENDENCIES: shared
# DESCRIPTION: Event-Driven Backtesting Broker. Simulates execution, spread,
# commissions, and PnL tracking for strategy validation.
#
# AUDIT REMEDIATION (2025-12-31):
# 1. METRICS UPDATE: Added native 'risk_reward_ratio' to get_stats().
# 2. CRITICAL FIX: Removed hardcoded commissions ($7/$10). Linked to Config ($5).
# 3. CRITICAL FIX: Removed hardcoded slippage. Linked to Config Spread Map.
# 4. COMPATIBILITY: Added positions property and submit_order API.
# =============================================================================
from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Generator, Any

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
            if key in self.data:
                return float(self.data[key])
            
            # Fallback for generic names if only one symbol exists
            if price_type in self.data:
                return float(self.data[price_type])
                
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
        for col in self.data.index:
            if "_close" in col:
                sym = col.replace("_close", "")
                res[sym] = float(self.data[col])
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
        self.trade_log: List[Dict] = [] # For reporting
        
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

    @property
    def positions(self) -> Dict[str, BacktestOrder]:
        """
        COMPATIBILITY LAYER: Exposes open positions as a dictionary {symbol: trade}.
        This fixes the AttributeError in ResearchStrategy.
        Assumes one trade per symbol (Hedging disabled in this view).
        """
        pos_map = {}
        for p in self.open_positions:
            pos_map[p.symbol] = p
        return pos_map

    def _inject_aux_data(self):
        """
        Injects synthetic cross-rates into the price map to prevent
        RiskManager from logging warnings when running single-symbol backtests.
        """
        defaults = {
            "USDJPY": 150.0,
            "GBPUSD": 1.25,
            "EURUSD": 1.08,
            "USDCAD": 1.35,
            "USDCHF": 0.90,
            "AUDUSD": 0.65,
            "NZDUSD": 0.60
        }
        for sym, price in defaults.items():
            if sym not in self.last_price_map:
                self.last_price_map[sym] = price

    def _get_simulation_conversion_rate(self, symbol: str, current_price: float) -> float:
        """
        Robust wrapper for RiskManager.get_conversion_rate.
        """
        # 1. Try Official Logic
        rate = RiskManager.get_conversion_rate(symbol, current_price, self.last_price_map)
        if rate > 0: return rate
        
        # 2. Simulation Fallback (Should rarely be reached now due to injection)
        s = symbol.upper()
        if "JPY" in s: return 0.0065 # ~150 JPY/USD
        if "GBP" in s: return 1.25
        if "EUR" in s: return 1.08
        if "AUD" in s: return 0.65
        return 1.0

    def process_pending(self, snapshot: MarketSnapshot):
        """ Alias for process_snapshot to match Live Engine nomenclature if needed """
        self.process_snapshot(snapshot)

    def process_snapshot(self, snapshot: MarketSnapshot):
        """
        Updates the state of the broker based on a new market tick/bar.
        Updates Equity, Checks SL/TP, Updates Price Map.
        """
        self.last_snapshot = snapshot
        
        # 1. Update Price Map for Risk Calc
        # Handle both flat structure (Symbol_Close) and multi-index
        for col in snapshot.data.index:
            if "_close" in col:
                sym = col.replace("_close", "")
                self.last_price_map[sym] = float(snapshot.data[col])
            elif "_" not in col and len(col) == 6: # Assumption: Pure symbol name index
                 self.last_price_map[col] = float(snapshot.data[col])

        # 2. Check Open Positions
        active_positions = []
        floating_pnl = 0.0
        
        for trade in self.open_positions:
            current_price = snapshot.get_price(trade.symbol, 'close')
            
            # Skip if no price update
            if current_price <= 0:
                active_positions.append(trade)
                continue

            # --- Update MAE/MFE ---
            # Calculate pips from entry
            pip_size = 0.01 if "JPY" in trade.symbol else 0.0001
            dist = (current_price - trade.entry_price) / pip_size
            if trade.action == "SELL": dist = -dist
            
            trade.max_favorable_excursion = max(trade.max_favorable_excursion, dist)
            trade.max_adverse_excursion = min(trade.max_adverse_excursion, dist)

            # --- Check Exit Conditions (SL/TP) ---
            closed = False
            exit_reason = ""
            close_price = 0.0

            # High/Low checks for more realistic backtesting (Hit SL/TP within bar)
            bar_high = snapshot.get_high(trade.symbol)
            bar_low = snapshot.get_low(trade.symbol)
            
            # If High/Low unavailable, use Close
            if bar_high == 0: bar_high = current_price
            if bar_low == 0: bar_low = current_price

            if trade.action == "BUY":
                # Check SL (Low)
                if trade.stop_loss > 0 and bar_low <= trade.stop_loss:
                    closed = True
                    exit_reason = "SL_HIT"
                    close_price = trade.stop_loss
                # Check TP (High)
                elif trade.take_profit > 0 and bar_high >= trade.take_profit:
                    closed = True
                    exit_reason = "TP_HIT"
                    close_price = trade.take_profit
            elif trade.action == "SELL":
                # Check SL (High)
                if trade.stop_loss > 0 and bar_high >= trade.stop_loss:
                    closed = True
                    exit_reason = "SL_HIT"
                    close_price = trade.stop_loss
                # Check TP (Low)
                elif trade.take_profit > 0 and bar_low <= trade.take_profit:
                    closed = True
                    exit_reason = "TP_HIT"
                    close_price = trade.take_profit

            if closed:
                self._finalize_trade(trade, close_price, snapshot.timestamp, exit_reason)
            else:
                # Update Floating PnL
                # Standard Lot = 100,000 units (Project assumption)
                contract_size = 100000
                raw_pnl = (current_price - trade.entry_price) * (trade.quantity * contract_size)
                if trade.action == "SELL": raw_pnl = -raw_pnl
                
                # Convert to Account Currency (USD)
                rate = self._get_simulation_conversion_rate(trade.symbol, current_price)
                
                # AUDIT FIX: Use Consistent Commission from Config
                # Deduct commission from floating equity to show realistic drawdown
                comm_drag = trade.quantity * self.commission_per_lot 
                
                floating_pnl += (raw_pnl * rate) - comm_drag
                active_positions.append(trade)

        self.open_positions = active_positions
        
        # 3. Update Equity
        self.equity = self.balance + floating_pnl
        self.equity_curve.append((snapshot.timestamp, self.equity))
        
        # 4. Margin Call / Blowout Check
        if self.equity < (self.initial_balance * 0.90): # 10% Max Drawdown Hard Limit
             self.is_blown = True

    def submit_order(self, order: BacktestOrder) -> Optional[int]:
        """
        Public API called by ResearchStrategy.
        """
        if order.quantity <= 0: return None
        
        # Assign Ticket
        order.ticket = self.ticket_counter
        self.ticket_counter += 1
        
        # Simulate Fill (Market Order Logic)
        
        # Ensure Entry Price is set
        if order.entry_price <= 0:
             # Try to get from last snapshot
             if self.last_snapshot:
                 order.entry_price = self.last_snapshot.get_price(order.symbol)
        
        # AUDIT FIX: Dynamic Spread Simulation based on Symbol
        # BUY: Ask = Price + Spread/2
        # SELL: Bid = Price - Spread/2
        # We simulate paying the FULL spread on entry for simplicity (conservative)
        pip = 0.01 if "JPY" in order.symbol else 0.0001
        
        # Fetch spread from config or default
        spread_pips = self.spread_map.get(order.symbol, self.default_spread)
        slippage_cost = spread_pips * pip
        
        order.entry_price = order.entry_price + slippage_cost if order.action == "BUY" else order.entry_price - slippage_cost
        
        # AUDIT FIX: Use Configured Commission
        order.commission = order.quantity * self.commission_per_lot
        
        self.open_positions.append(order)
        return order.ticket

    def get_position(self, symbol: str) -> Optional[BacktestOrder]:
        """Returns the first active position for a symbol (Hedging disabled for this check)."""
        for p in self.open_positions:
            if p.symbol == symbol:
                return p
        return None

    def _close_partial_position(self, trade: BacktestOrder, qty: float, price: float, time: datetime, reason: str):
        """Simulates a partial or full close."""
        self._finalize_trade(trade, price, time, reason)

    def _finalize_trade(self, trade: BacktestOrder, close_price: float, close_time: datetime, reason: str):
        """
        Closes a trade, calculates final PnL, updates Balance.
        """
        # 1. Calculate Raw PnL
        raw_diff = (close_price - trade.entry_price)
        if trade.action == "SELL": raw_diff = -raw_diff
        
        raw_profit = raw_diff * (trade.quantity * 100000)
        
        # 2. Convert to USD
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
        
        # 6. Log to Trade Log (For DataFrame construction)
        self.trade_log.append({
            'Entry_Time': trade.timestamp_created,
            'Exit_Time': close_time,
            'Symbol': trade.symbol,
            'Action': trade.action,
            'Size': trade.quantity,
            'Entry_Price': trade.entry_price,
            'Exit_Price': close_price,
            'Gross_PnL': gross_pnl,
            'Commission': trade.commission,
            'Net_PnL': net_pnl,
            'Status': "CLOSED",
            'Comment': reason,
            'MFE_Pips': trade.max_favorable_excursion,
            'MAE_Pips': trade.max_adverse_excursion,
            'Duration_Min': (close_time - trade.timestamp_created).total_seconds() / 60,
            'Regime': trade.metadata.get('regime', 'Unknown'),
            'Confidence': trade.metadata.get('confidence', 0.0)
        })
        
        # Log it
        symbol_icon = "🟢" if net_pnl > 0 else "🔴"
        logger.debug(f"{symbol_icon} CLOSED {trade.symbol} {trade.action}: ${net_pnl:.2f} ({reason})")

    def get_stats(self) -> Dict[str, Any]:
        """
        Generates comprehensive performance statistics for the backtest.
        Includes Max Drawdown, Sharpe, Sortino, Win Rate, Expectancy, and Risk:Reward.
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
            "risk_reward_ratio": 0.0, # NATIVE SUPPORT ADDED
            "largest_win": 0.0,
            "largest_loss": 0.0
        }

        # 1. Trade-based Stats
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
            
            # Risk:Reward Ratio (Avg Win / Avg Loss)
            avg_loss_abs = abs(stats["avg_loss"])
            if avg_loss_abs > 0:
                stats["risk_reward_ratio"] = stats["avg_win"] / avg_loss_abs
            else:
                stats["risk_reward_ratio"] = 10.0 if stats["avg_win"] > 0 else 0.0
            
            # Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss)
            win_pct = len(wins) / len(pnls)
            loss_pct = len(losses) / len(pnls)
            stats["expectancy"] = (win_pct * stats["avg_win"]) + (loss_pct * stats["avg_loss"])
            
            # SQN = sqrt(N) * (Expectancy / Stdev(PnL))
            if len(pnls) > 1:
                std_pnl = np.std(pnls)
                if std_pnl > 0:
                    stats["sqn"] = np.sqrt(len(pnls)) * (np.mean(pnls) / std_pnl)

        # 2. Time-series Stats (Drawdown, Sharpe)
        if len(self.equity_curve) > 1:
            df = pd.DataFrame(self.equity_curve, columns=['time', 'equity'])
            df.set_index('time', inplace=True)
            
            # Drawdown Calculation
            df['peak'] = df['equity'].cummax()
            df['dd'] = df['equity'] - df['peak']
            df['dd_pct'] = (df['dd'] / df['peak']) * 100
            
            stats["max_drawdown"] = df['dd'].min()
            stats["max_drawdown_pct"] = df['dd_pct'].min()
            
            # Returns Calculation for Sharpe/Sortino
            # Resample to hourly to standardize volatility
            try:
                hourly_equity = df['equity'].resample('1H').last().ffill()
                returns = hourly_equity.pct_change().dropna()
                
                if len(returns) > 1:
                    avg_ret = returns.mean()
                    std_ret = returns.std()
                    
                    # Annualize (Assume 24/5 trading = ~6000 hours/year)
                    # This is an approximation for Forex
                    annualization_factor = np.sqrt(252 * 24)
                    
                    if std_ret > 0:
                        stats["sharpe_ratio"] = (avg_ret / std_ret) * annualization_factor
                    
                    # Sortino (Downside deviation only)
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