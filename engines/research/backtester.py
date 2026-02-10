from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from collections import deque
from datetime import datetime, time, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Generator, Any
import pytz

# Shared Imports
from shared.financial.risk import RiskManager
from shared.core.config import CONFIG
from shared.core.logging_setup import LogSymbols
from shared.data import load_real_data, AdaptiveImbalanceBarGenerator

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
    slippage_penalty: float = 0.0 # Track the cost of reality
    
    def __post_init__(self):
        if not self.action:
            self.action = "BUY" if self.side == 1 else "SELL"

class BacktestBroker:
    """
    Simulates a Broker environment:
    - Order Execution (Immediate Fill with VOLATILITY SLIPPAGE)
    - PnL Tracking (Equity/Balance)
    - Margin Checks (simplified)
    - Daily Hard Deck Enforcement
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
        self.commission_per_lot = CONFIG.get('forensic_audit', {}).get('commission_per_lot', 5.0)
        self.spread_map = CONFIG.get('forensic_audit', {}).get('spread_pips', {})
        self.default_spread = self.spread_map.get('default', 1.6)

        # Pre-populate price map to prevent initial warnings
        self._inject_aux_data()
        
        # --- DAILY ANCHOR TRACKING ---
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
        Calculates the current Daily PnL % relative to the Daily Start Equity.
        Used for Profit Buffer Scaling (Earn to Burn).
        """
        if self.daily_start_equity <= 0:
            return 0.0
        
        current_pnl = self.equity - self.daily_start_equity
        return current_pnl / self.daily_start_equity

    def _inject_aux_data(self):
        """
        Injects synthetic cross-rates into the price map.
        Added High-Beta Pairs (GBPAUD, GBPNZD, EURAUD) to prevent crashes.
        """
        defaults = {
            "USDJPY": 150.0, "GBPUSD": 1.25, "EURUSD": 1.08,
            "USDCAD": 1.35, "USDCHF": 0.90, "AUDUSD": 0.65, "NZDUSD": 0.60,
            "GBPAUD": 1.95, "EURAUD": 1.65, "GBPNZD": 2.10,
            "EURJPY": 162.0, "GBPJPY": 190.0, "AUDJPY": 97.0
        }
        for sym, price in defaults.items():
            if sym not in self.last_price_map:
                self.last_price_map[sym] = price

    def _get_point_value(self, symbol: str) -> float:
        """Determines pipette size (0.00001 or 0.001 for JPY)."""
        if "JPY" in symbol:
            return 0.001
        return 0.00001

    def _get_spread_for_symbol(self, symbol: str) -> float:
        """
        Retrieves the strict spread from config.yaml.
        Falls back to 'default' if specific pair not found.
        """
        # 1. Check direct match (e.g., 'GBPNZD')
        if symbol in self.spread_map:
            return float(self.spread_map[symbol])
        
        # 2. Check Default
        return float(self.spread_map.get('default', 1.6))

    def _get_simulation_conversion_rate(self, symbol: str, current_price: float) -> float:
        """
        Robust wrapper for RiskManager.get_conversion_rate.
        Dynamic lookup via self.last_price_map to ensure PnL accuracy during volatility.
        """
        rate = RiskManager.get_conversion_rate(symbol, current_price, self.last_price_map)
        
        # Sanity Check: If rate is 1.0 but it's clearly a non-USD pair
        if rate == 1.0 and not symbol.endswith("USD") and "USD" in symbol:
             # It is a USD pair (e.g. USDJPY), so rate might be 1/Price or Price
             pass
        elif rate == 1.0 and "USD" not in symbol:
             # Cross pair (e.g. EURGBP) returning 1.0 implies missing conversion data
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
            # Assumption: Snapshot 'close' is the BID price.
            current_bid = snapshot.get_price(trade.symbol, 'close')
            
            if current_bid <= 0:
                active_positions.append(trade)
                continue

            # Calculate Ask based on Configured Spread
            point = self._get_point_value(trade.symbol)
            spread_pips = self._get_spread_for_symbol(trade.symbol)
            spread_cost_price = spread_pips * (point * 10)
            current_ask = current_bid + spread_cost_price

            # Determine Valuation Price (Mark-to-Market)
            if trade.action == "BUY":
                # Longs are valued at Bid (Selling back to market)
                valuation_price = current_bid
            else:
                # Shorts are valued at Ask (Buying back from market)
                valuation_price = current_ask

            # --- Update MAE/MFE ---
            pip_size = point * 10
            
            if trade.action == "BUY":
                dist = (current_bid - trade.entry_price) / pip_size
            else:
                dist = (trade.entry_price - current_ask) / pip_size
            
            trade.max_favorable_excursion = max(trade.max_favorable_excursion, dist)
            trade.max_adverse_excursion = min(trade.max_adverse_excursion, dist)

            # --- Check Exit Conditions (SL/TP) ---
            closed = False
            exit_reason = ""
            close_price = 0.0

            bar_high = snapshot.get_high(trade.symbol) # Bid High
            bar_low = snapshot.get_low(trade.symbol)   # Bid Low
            
            if bar_high == 0: bar_high = current_bid
            if bar_low == 0: bar_low = current_bid

            # Construct Ask High/Low for Short Logic
            bar_high_ask = bar_high + spread_cost_price
            bar_low_ask = bar_low + spread_cost_price

            if trade.action == "BUY":
                # Long: Hit SL if Bid Low <= SL
                if trade.stop_loss > 0 and bar_low <= trade.stop_loss:
                    closed = True
                    exit_reason = "SL_HIT"
                    close_price = trade.stop_loss
                # Long: Hit TP if Bid High >= TP
                elif trade.take_profit > 0 and bar_high >= trade.take_profit:
                    closed = True
                    exit_reason = "TP_HIT"
                    close_price = trade.take_profit
            elif trade.action == "SELL":
                # Short: Hit SL if Ask High >= SL
                if trade.stop_loss > 0 and bar_high_ask >= trade.stop_loss:
                    closed = True
                    exit_reason = "SL_HIT"
                    close_price = trade.stop_loss
                # Short: Hit TP if Ask Low <= TP
                elif trade.take_profit > 0 and bar_low_ask <= trade.take_profit:
                    closed = True
                    exit_reason = "TP_HIT"
                    close_price = trade.take_profit

            if closed:
                self._finalize_trade(trade, close_price, snapshot.timestamp, exit_reason)
            else:
                # Update Floating PnL
                contract_size = self.contract_size if hasattr(self, 'contract_size') else 100000
                
                raw_diff = (valuation_price - trade.entry_price)
                if trade.action == "SELL": raw_diff = -raw_diff
                
                raw_pnl = raw_diff * (trade.quantity * contract_size)
                
                # Use Dynamic Rate
                rate = self._get_simulation_conversion_rate(trade.symbol, valuation_price)
                comm_drag = trade.quantity * self.commission_per_lot 
                
                floating_pnl += (raw_pnl * rate) - comm_drag
                active_positions.append(trade)

        self.open_positions = active_positions
        
        # 3. Update Equity
        self.equity = self.balance + floating_pnl
        self.equity_curve.append((snapshot.timestamp, self.equity))
        
        # 4. HARD DECK ENFORCEMENT
        # Check Daily Drawdown relative to Anchor
        if self.daily_start_equity > 0:
            daily_loss_amount = self.daily_start_equity - self.equity
            daily_loss_limit = self.daily_start_equity * self.max_daily_loss_pct
            
            if daily_loss_amount > daily_loss_limit:
                logger.warning(f"💀 HARD DECK BREACHED (Sim): Start {self.daily_start_equity:.2f} -> Curr {self.equity:.2f}")
                self.is_blown = True # Mark as blown for optimizer
                
                # Liquidate all open positions instantly at current price
                for trade in list(self.open_positions):
                    curr_p = snapshot.get_price(trade.symbol, 'close')
                    if trade.action == "SELL":
                        pt = self._get_point_value(trade.symbol)
                        sp = self._get_spread_for_symbol(trade.symbol) * (pt * 10)
                        curr_p += sp
                        
                    self._finalize_trade(trade, curr_p, snapshot.timestamp, "HARD_DECK_LIQ")
                self.open_positions = [] # Clear

        # 5. Margin Call / Blowout Check (Total)
        if self.equity < (self.initial_balance * 0.90): 
             self.is_blown = True

    def submit_order(self, order: BacktestOrder) -> Optional[int]:
        """
        Public API called by ResearchStrategy.
        Applies STRICT REALITY INJECTION (Volatility Penalty).
        """
        if order.quantity <= 0: return None
        if self.is_blown: return None # No trades if blown
        
        order.ticket = self.ticket_counter
        self.ticket_counter += 1
        
        # Determine Base Price (Bid)
        if order.entry_price <= 0:
             if self.last_snapshot:
                 order.entry_price = self.last_snapshot.get_price(order.symbol)
        
        # Calculate Spread
        point = self._get_point_value(order.symbol)
        spread_pips = self._get_spread_for_symbol(order.symbol)
        spread_cost = spread_pips * (point * 10)
        
        # --- VOLATILITY PENALTY (REALITY INJECTION) ---
        # If RVOL (Relative Volume) is high, we assume SLIPPAGE is inevitable.
        # Live market orders NEVER fill at the exact tick price during momentum ignition.
        rvol = order.metadata.get('rvol', 0.0)
        vol_penalty = 0.0
        
        # Aggressor threshold is roughly 2.0-2.5 RVOL
        if rvol > 2.0:
            # Slippage Factor: For every 1.0 above 2.0 RVOL, increase spread cost by 50%
            # e.g. RVOL 4.0 -> Factor = (4-2)*0.5 = 1.0 -> 100% extra spread penalty
            # e.g. RVOL 10.0 -> Factor = 4.0 -> 400% extra spread penalty (Reality check for spikes)
            slippage_factor = (rvol - 2.0) * 0.5
            vol_penalty = spread_cost * slippage_factor
            
            # Log significant slippage for audit
            if slippage_factor > 1.0:
                logger.info(f"🛡️ REALITY CHECK {order.symbol}: RVOL {rvol:.1f} -> Volatility Slippage {vol_penalty:.5f}")
                
        order.slippage_penalty = vol_penalty

        # EXECUTION REALITY:
        # BUY: Fills at Ask + Penalty
        # SELL: Fills at Bid - Penalty (Worse Price)
        
        if order.action == "BUY":
            # Buy Limit/Market fills at Ask
            order.entry_price = order.entry_price + spread_cost + vol_penalty
        else:
            # Sell Limit/Market fills at Bid - Penalty (To sell, we cross spread, but we are entering at Bid quote)
            # Actually, standard sell fills at Bid. But slippage implies we missed the best Bid.
            order.entry_price = order.entry_price - vol_penalty
            
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
        
        contract_size = self.contract_size if hasattr(self, 'contract_size') else 100000
        raw_profit = raw_diff * (trade.quantity * contract_size)
        
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
        
        # 6. Log to Trade Log (Enhanced Metadata)
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
            'Vol_Slippage': float(trade.slippage_penalty), # Audit metric
            'Duration_Min': (close_time - trade.timestamp_created).total_seconds() / 60,
            'Regime': clean_metadata.get('regime', 'Unknown'),
            'Confidence': clean_metadata.get('confidence', 0.0),
            'Tighten_Stops': clean_metadata.get('tighten_stops', False)
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

def process_data_into_bars(symbol: str, n_ticks: int = 4000000) -> pd.DataFrame:
    """
    Helper to Load Ticks -> Aggregate to Tick Imbalance Bars (TIBs) -> Return Clean DataFrame.
    """
    # 1. Load Massive Amount of Ticks
    raw_ticks = load_real_data(symbol, n_candles=n_ticks, days=730 * 2)
    
    if raw_ticks.empty:
        return pd.DataFrame()

    # 2. AGGREGATION: ADAPTIVE IMBALANCE BARS WITH AUTO-CALIBRATION
    config_threshold = CONFIG['data'].get('volume_bar_threshold', 10) 
    alpha = CONFIG['data'].get('imbalance_alpha', 0.05)
    
    # --- AUTO-CALIBRATION LOOP ---
    first_timestamp = raw_ticks.index.min()
    calibration_cutoff = first_timestamp + timedelta(days=30)
    
    calibration_df = raw_ticks[raw_ticks.index < calibration_cutoff]
    
    if calibration_df.empty:
        calibration_df = raw_ticks
        
    current_threshold = config_threshold
    min_bars_needed = 500
    attempts = 0
    max_attempts = 4
    
    final_threshold = config_threshold
    
    logger.info(f"🔎 Calibrating {symbol} on first {len(calibration_df)} ticks (30 Days)...")

    while attempts < max_attempts:
        gen = AdaptiveImbalanceBarGenerator(
            symbol=symbol,
            initial_threshold=current_threshold,
            alpha=alpha
        )
        
        bar_count = 0
        for row in calibration_df.itertuples():
            price = getattr(row, 'price', getattr(row, 'close', None))
            vol = getattr(row, 'volume', 1.0)
            
            ts = getattr(row, 'Index', getattr(row, 'time', None))
            if isinstance(ts, (datetime, pd.Timestamp)):
                ts_val = ts.timestamp()
            else:
                ts_val = float(ts)
                
            if price is None: continue

            b_vol = 0.0
            s_vol = 0.0
            
            bar = gen.process_tick(price, vol, ts_val, b_vol, s_vol)
            if bar:
                bar_count += 1
        
        if bar_count >= min_bars_needed:
            final_threshold = current_threshold
            if attempts > 0:
                logger.info(f"✅ {symbol}: Auto-Calibrated Threshold to {final_threshold} (Generated {bar_count} bars in 30 days)")
            break
        else:
            attempts += 1
            new_threshold = max(5.0, current_threshold * 0.5) 
            logger.warning(f"⚠️ {symbol}: Insufficient bars ({bar_count}). Retrying with threshold {new_threshold}...")
            current_threshold = new_threshold
            final_threshold = current_threshold

    # --- PRODUCTION GENERATION ---
    gen = AdaptiveImbalanceBarGenerator(
        symbol=symbol,
        initial_threshold=final_threshold,
        alpha=alpha
    )
    
    bars_list = []
    
    for row in raw_ticks.itertuples():
        price = getattr(row, 'price', getattr(row, 'close', None))
        vol = getattr(row, 'volume', 1.0)
        
        ts = getattr(row, 'Index', getattr(row, 'time', None))
        if isinstance(ts, (datetime, pd.Timestamp)):
            ts_val = ts.timestamp()
        else:
            ts_val = float(ts)
            
        b_vol = 0.0
        s_vol = 0.0
        
        if price is None: continue

        bar = gen.process_tick(price, vol, ts_val, b_vol, s_vol)
        
        if bar:
            bars_list.append({
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

    if not bars_list:
        logger.warning(f"❌ {symbol}: Failed to generate bars from full dataset.")
        return pd.DataFrame()

    df_bars = pd.DataFrame(bars_list)
    df_bars['time'] = pd.to_datetime(df_bars['timestamp'], unit='s', utc=True)
    df_bars.set_index('time', inplace=True, drop=False)
    
    return df_bars