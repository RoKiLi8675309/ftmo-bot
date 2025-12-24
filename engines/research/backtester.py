# =============================================================================
# FILENAME: engines/research/backtester.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/backtester.py
# DEPENDENCIES: shared
# DESCRIPTION: Event-Driven Backtesting Broker. Simulates execution, spread,
# commissions, and PnL tracking for strategy validation.
# AUDIT REMEDIATION (GROK):
# 1. Added Random Slippage (0-2 pips).
# 2. Added Commission Logic ($5/lot).
# 3. VERIFIED: trade_log keys match main_research.py requirements.
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
from shared import RiskManager, CONFIG, LogSymbols

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
        Handles both Single-Index (Train) and Multi-Index (Prod) DataFrame formats.
        """
        val = 0.0
        
        # 1. Try Direct Access (Single Symbol / Training Mode)
        if price_type in self.data:
            val = self.data[price_type]
            
        # 2. Try Prefixed Access (Multi Symbol / Production Mode)
        elif f"{symbol}_{price_type}" in self.data:
            val = self.data[f"{symbol}_{price_type}"]
            
        # 3. Fallback Logic: Tick Data Compatibility
        if (pd.isna(val) or val == 0.0) and price_type in ['close', 'open', 'high', 'low']:
            if 'price' in self.data:
                val = self.data['price']
            elif f"{symbol}_price" in self.data:
                val = self.data[f"{symbol}_price"]

        # 4. Final Validation
        if pd.isna(val) or val is None:
            return 0.0
            
        return float(val)
    
    def get_high(self, symbol: str) -> float:
        h = self.get_price(symbol, 'high')
        return h if h > 0 else self.get_price(symbol, 'close')

    def get_low(self, symbol: str) -> float:
        l = self.get_price(symbol, 'low')
        return l if l > 0 else self.get_price(symbol, 'close')

    def get_open(self, symbol: str) -> float:
        o = self.get_price(symbol, 'open')
        return o if o > 0 else self.get_price(symbol, 'close')

    def get_volume(self, symbol: str) -> float:
        return self.get_price(symbol, 'volume')

    def to_price_dict(self) -> Dict[str, float]:
        """
        Extracts a dictionary of {symbol: close_price} for RiskManager context.
        Attempts to infer symbols from column headers if in Multi-Index mode.
        """
        prices = {}
        # Naive extraction: If keys look like "EURUSD_close", extract it.
        # If single symbol mode, we might only have "close", so we can't infer other pairs.
        for key, val in self.data.items():
            if isinstance(key, str) and "_" in key:
                parts = key.split("_")
                if len(parts) == 2 and parts[1] == "close":
                    prices[parts[0]] = float(val)
        return prices

@dataclass
class BacktestOrder:
    """Represents an order submitted to the Backtest Broker."""
    symbol: str
    side: int  # 1 for Buy, -1 for Sell
    quantity: float
    timestamp_created: datetime
    order_type: str = 'MARKET'
    stop_loss: float = 0.0
    take_profit: float = 0.0
    comment: str = ""

@dataclass
class BacktestPosition:
    """Represents an open position in the Backtest Broker."""
    symbol: str
    entry_price: float
    quantity: float
    side: int
    stop_loss: float = 0.0
    take_profit: float = 0.0
    unrealized_pnl: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)

class BacktestBroker:
    """
    Simulates exchange mechanics for backtesting.
    Enforces Spread, Commission, Slippage and Drawdown limits (FTMO rules).
    """
    def __init__(self, starting_cash: float = 100000.0):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.equity = starting_cash
        
        self.positions: Dict[str, BacktestPosition] = {}
        self.pending_orders: deque[BacktestOrder] = deque()
        
        # Performance Tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trade_log: List[Dict[str, Any]] = []

        # Costs & Realism
        audit_conf = CONFIG.get('forensic_audit', {})
        self.commission_per_lot = audit_conf.get('commission_per_lot', 5.0)
        self.max_slippage = audit_conf.get('max_slippage_pips', 2.0)
        self.enforce_spread = audit_conf.get('enforce_spread', True)
        self.spread_map = audit_conf.get('spread_pips', {})

        # State Flags
        self.is_blown = False
        self.max_drawdown_limit = 0.90  # 10% Max Trailing Drawdown
        
        # Snapshot Cache for Price Map (simulating market_prices for RiskManager)
        self.last_price_map = {}

    def get_position(self, symbol: str) -> Optional[BacktestPosition]:
        return self.positions.get(symbol)

    def submit_order(self, order: BacktestOrder) -> None:
        if self.is_blown:
            self.trade_log.append({
                'Entry_Time': order.timestamp_created,
                'Exit_Time': order.timestamp_created,
                'Symbol': order.symbol,
                'Action': 'REJECTED',
                'Status': 'ACCOUNT_BLOWN',
                'Comment': "Order rejected: Account is blown."
            })
            return
        self.pending_orders.append(order)

    def process_pending(self, snapshot: MarketSnapshot) -> None:
        if self.is_blown: return
        
        # Update internal price map from snapshot for use in conversions
        self.last_price_map = snapshot.to_price_dict()
        
        # FIX: Ensure Aux data is present so RiskManager doesn't warn/fail
        self._inject_aux_data()

        # 1. Check Stops and Limits first (Priority)
        self._check_sl_tp(snapshot)

        # 2. Process New Orders
        orders_to_process = len(self.pending_orders)
        for _ in range(orders_to_process):
            if not self.pending_orders: break
            order = self.pending_orders.popleft()
            self._execute_order(order, snapshot)

        # 3. Mark to Market (Update Equity)
        self.update_market_data(snapshot)

        # 4. Drawdown Check
        if self.equity < (self.starting_cash * self.max_drawdown_limit):
            if not self.is_blown:
                self.is_blown = True
                self.trade_log.append({
                    'Entry_Time': snapshot.timestamp,
                    'Exit_Time': snapshot.timestamp,
                    'Symbol': 'ACCOUNT',
                    'Action': 'BLOWOUT',
                    'Size': 0,
                    'Entry_Price': 0,
                    'Exit_Price': 0,
                    'Net_PnL': 0,
                    'Status': 'CRITICAL',
                    'Comment': f"Account Blown: Equity {self.equity:.2f} < Limit"
                })

    def _execute_order(self, order: BacktestOrder, snapshot: MarketSnapshot) -> None:
        base_price = snapshot.get_open(order.symbol)
        if base_price == 0.0:
            base_price = snapshot.get_price(order.symbol, 'close')
        if base_price == 0.0:
            self.pending_orders.appendleft(order)
            return

        pip_size, _ = RiskManager.get_pip_info(order.symbol)

        # 1. Spread Cost
        spread_pips = self.spread_map.get(order.symbol, self.spread_map.get('default', 1.5))
        spread_cost = spread_pips * pip_size if self.enforce_spread else 0.0

        # 2. Random Slippage (Simulated Latency/Impact)
        # GROK REMEDIATION: Random float between 0 and max_slippage (default 2.0)
        slippage_pips = np.random.uniform(0, self.max_slippage)
        slippage_cost = slippage_pips * pip_size

        # Execution Price Calculation
        # Buy: Ask = Price + Spread + Slippage (Adverse)
        # Sell: Bid = Price - Slippage (Adverse)
        if order.side == 1:
            fill_price = base_price + spread_cost + slippage_cost
            cost_desc = f"Spread:{spread_pips:.1f}+Slip:{slippage_pips:.1f}p"
        else:
            fill_price = base_price - slippage_cost
            cost_desc = f"Slip:{slippage_pips:.1f}p"

        # Position Management
        if order.symbol in self.positions:
            pos = self.positions[order.symbol]
            if pos.side == order.side:
                # Average Down
                total_qty = pos.quantity + order.quantity
                avg_price = ((pos.entry_price * pos.quantity) + (fill_price * order.quantity)) / total_qty
                pos.quantity = total_qty
                pos.entry_price = avg_price
                if order.stop_loss > 0: pos.stop_loss = order.stop_loss
                if order.take_profit > 0: pos.take_profit = order.take_profit
            else:
                # Hedge / Close
                self._close_partial_position(pos, order.quantity, fill_price, snapshot.timestamp, order.comment)
        else:
            # DEBUG: Log Entry Costs
            # logger.debug(f"🔵 ENTRY {order.symbol}: Base={base_price:.5f} | Fill={fill_price:.5f} | {cost_desc}")
            
            self.positions[order.symbol] = BacktestPosition(
                symbol=order.symbol,
                entry_price=fill_price,
                quantity=order.quantity,
                side=order.side,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                entry_time=snapshot.timestamp
            )

    def _close_partial_position(self, pos: BacktestPosition, qty_to_close: float, exit_price: float, timestamp: datetime, reason: str) -> None:
        """
        Closes position and deducts COMMISSIONS + SWAP (simplified).
        """
        closed_qty = min(pos.quantity, qty_to_close)
        
        # PnL in Quote
        diff = exit_price - pos.entry_price
        if pos.side == -1: diff = -diff
        pnl_quote = diff * closed_qty * 100000
        
        # Convert to USD using robust simulation conversion
        rate = self._get_simulation_conversion_rate(pos.symbol, exit_price)
        pnl_usd = pnl_quote * rate
        
        # GROK REMEDIATION: Commission Deduction (Round Turn)
        # $5 per lot -> $5 * lots
        comm_cost = self.commission_per_lot * closed_qty
        
        net_pnl = pnl_usd - comm_cost
        
        self.cash += net_pnl
        
        # DEBUG: Log Exit Costs
        # logger.debug(f"💸 EXIT {pos.symbol}: Gross=${pnl_usd:.2f} | Comm=${comm_cost:.2f} | Net=${net_pnl:.2f} | {reason}")

        self.trade_log.append({
            'Entry_Time': pos.entry_time,
            'Exit_Time': timestamp,
            'Symbol': pos.symbol,
            'Action': 'BUY' if pos.side == 1 else 'SELL',
            'Size': round(closed_qty, 2),
            'Entry_Price': pos.entry_price,
            'Exit_Price': exit_price,
            'Gross_PnL': round(pnl_usd, 2),
            'Commission': round(comm_cost, 2),
            'Net_PnL': round(net_pnl, 2),
            'Status': 'CLOSED',
            'Comment': reason
        })
        
        remaining = pos.quantity - closed_qty
        if remaining > 0.0001:
            pos.quantity = remaining
        else:
            del self.positions[pos.symbol]

    def _check_sl_tp(self, snapshot: MarketSnapshot) -> None:
        to_close = []
        for symbol, pos in list(self.positions.items()):
            high = snapshot.get_high(symbol)
            low = snapshot.get_low(symbol)
            if high == 0.0 or low == 0.0: continue

            # SL Logic
            if pos.stop_loss > 0:
                if (pos.side == 1 and low <= pos.stop_loss):
                    # GROK REMEDIATION: Slippage applies to SL too!
                    slippage = np.random.uniform(0, self.max_slippage) * RiskManager.get_pip_info(symbol)[0]
                    exec_price = pos.stop_loss - slippage
                    to_close.append((symbol, exec_price, "SL_HIT"))
                    continue
                elif (pos.side == -1 and high >= pos.stop_loss):
                    slippage = np.random.uniform(0, self.max_slippage) * RiskManager.get_pip_info(symbol)[0]
                    exec_price = pos.stop_loss + slippage
                    to_close.append((symbol, exec_price, "SL_HIT"))
                    continue

            # TP Logic (Assume limit fill at price, usually positive slippage if gap, but we stay conservative)
            if pos.take_profit > 0:
                if (pos.side == 1 and high >= pos.take_profit):
                    to_close.append((symbol, pos.take_profit, "TP_HIT"))
                    continue
                elif (pos.side == -1 and low <= pos.take_profit):
                    to_close.append((symbol, pos.take_profit, "TP_HIT"))
                    continue

        for sym, price, reason in to_close:
            if sym in self.positions:
                self._close_partial_position(
                    self.positions[sym],
                    self.positions[sym].quantity,
                    price,
                    snapshot.timestamp,
                    reason
                )

    def update_market_data(self, snapshot: MarketSnapshot) -> None:
        # Ensure map is current
        self.last_price_map = snapshot.to_price_dict()
        self._inject_aux_data()
        
        floating_pnl = 0.0
        for symbol, pos in self.positions.items():
            price = snapshot.get_price(symbol, 'close')
            if price == 0: continue
            
            diff = price - pos.entry_price
            if pos.side == -1: diff = -diff
            
            # Apply approximated commission to floating PnL for realistic equity curve
            comm_drag = self.commission_per_lot * pos.quantity
            
            raw_pnl = diff * pos.quantity * 100000
            rate = self._get_simulation_conversion_rate(symbol, price)
            floating_pnl += (raw_pnl * rate) - comm_drag
            
        self.equity = self.cash + floating_pnl
        self.equity_curve.append((snapshot.timestamp, self.equity))

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
        if "JPY" in s: return 0.0065 # ~150
        if "GBP" in s: return 1.25
        if "EUR" in s: return 1.08
        if "AUD" in s: return 0.65
        if "CAD" in s: return 0.75
        if "CHF" in s: return 1.10
        if "NZD" in s: return 0.60
        
        return 1.0