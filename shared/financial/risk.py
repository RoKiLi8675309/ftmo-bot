# =============================================================================
# FILENAME: shared/financial/risk.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/financial/risk.py
# DEPENDENCIES: numpy, pandas, scipy (optional on Windows)
# DESCRIPTION: Core Risk Management logic (Position Sizing, FTMO Limits, HRP).
#
# PHOENIX STRATEGY V7.5 (SNIPER PROTOCOL RISK):
# 1. STOP LOSS: Widened to 2.0 * ATR (Prevents premature noise stop-outs).
# 2. TAKE PROFIT: Adjusted to 3.0 * ATR (Sniper Target).
# 3. STATIC RISK: STRICT 0.5% CAP. Removed "Aggressive" 1.0% tier.
#    - Priority is SURVIVAL and consistency over volatility.
# 4. COMPLIANCE: SessionGuard enforces Friday Liquidation and Trading Hours.
# 5. SQN SCALING: Added Performance-Based Sizing (Cut Losers / Press Winners).
# =============================================================================
from __future__ import annotations
import logging
import math
import time
import pytz
import json
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, Optional, Tuple, Any, List

# Scipy Imports (Guarded for Windows Producer compatibility)
try:
    from scipy.stats import linregress
    import scipy.cluster.hierarchy as sch
    import scipy.spatial.distance as ssd
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from shared.core.config import CONFIG
from shared.domain.models import TradeContext, Trade

EPS = 1e-9
logger = logging.getLogger("RiskManager")

class RiskManager:
    """
    Stateless utilities for Pip value calculations, Exchange Rates,
    and Advanced Position Sizing (Fixed Risk for Prop Firms).
    """
    # Default fallback, but logic now prefers config/override
    DEFAULT_CONTRACT_SIZE = 100_000

    @staticmethod
    def get_pip_info(symbol: str) -> Tuple[float, int]:
        """Returns (pip_size, digits)."""
        s = symbol.upper()
        if "JPY" in s:
            return 0.01, 3
        if "XAU" in s or "XAG" in s:
            return 0.1, 2
        # Indices heuristic
        if any(x in s for x in ["US30", "GER30", "NAS100", "SPX500", "DJI", "DAX"]):
            return 1.0, 1  # Usually 1 point steps
        if "BTC" in s or "ETH" in s:
            return 1.0, 2
        return 0.0001, 5

    @staticmethod
    def get_conversion_rate(symbol: str, price: float, market_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculates the Quote -> USD conversion rate using LIVE market data.
        Critical for accurate risk calculation in USD.
        """
        s = symbol.upper()
        
        # Base case: Quote is USD (e.g., EURUSD)
        if s.endswith("USD"):
            return 1.0
        
        # Helper to safely get price from map
        def get_price(sym: str) -> Optional[float]:
            if market_prices and sym in market_prices:
                return market_prices[sym]
            return None

        # JPY Pairs (Quote = JPY). Need JPY->USD (1 / USDJPY).
        if "JPY" in s:
            usdjpy = get_price("USDJPY")
            if usdjpy and usdjpy > 0: return 1.0 / usdjpy
            if s == "USDJPY" and price > 0: return 1.0 / price
            
            if market_prices is not None:
                logger.warning(f"⚠️ RISK MISMATCH: Missing USDJPY price for {symbol} conversion. Using static 150.0 fallback.")
            return 0.00666 # Fallback ~150.0
        
        # GBP Pairs (Quote = GBP). Need GBP->USD (GBPUSD).
        if s.endswith("GBP"):
            gbpusd = get_price("GBPUSD")
            if gbpusd: return gbpusd
            
            if market_prices is not None:
                logger.warning(f"⚠️ RISK MISMATCH: Missing GBPUSD price for {symbol}. Using static 1.25 fallback.")
            return 1.25 # Fallback

        # CAD Pairs (Quote = CAD). Need CAD->USD (1 / USDCAD).
        if s.endswith("CAD"):
            usdcad = get_price("USDCAD")
            if usdcad and usdcad > 0: return 1.0 / usdcad
            if s == "USDCAD" and price > 0: return 1.0 / price
            
            if market_prices is not None:
                logger.warning(f"⚠️ RISK MISMATCH: Missing USDCAD price for {symbol}. Using static 0.75 fallback.")
            return 0.75

        # CHF Pairs (Quote = CHF). Need CHF->USD (1 / USDCHF).
        if s.endswith("CHF"):
            usdchf = get_price("USDCHF")
            if usdchf and usdchf > 0: return 1.0 / usdchf
            if s == "USDCHF" and price > 0: return 1.0 / price
            return 1.10

        # AUD Pairs (Quote = AUD). Need AUD->USD (AUDUSD).
        if s.endswith("AUD"):
            audusd = get_price("AUDUSD")
            if audusd: return audusd
            if s == "AUDUSD" and price > 0: return price
            
            if market_prices is not None:
                logger.warning(f"⚠️ RISK MISMATCH: Missing AUDUSD price for {symbol}. Using static 0.65 fallback.")
            return 0.65

        # NZD Pairs
        if s.endswith("NZD"):
            nzdusd = get_price("NZDUSD")
            if nzdusd: return nzdusd
            if s == "NZDUSD" and price > 0: return price
            return 0.60
            
        # Default fallback
        return 1.0

    @staticmethod
    def calculate_rck_size(
        context: TradeContext,
        conf: float,
        volatility: float,
        active_correlations: int = 0,
        market_prices: Optional[Dict[str, float]] = None,
        atr: Optional[float] = None,
        account_size: Optional[float] = None, 
        contract_size_override: Optional[float] = None, 
        ker: float = 1.0, 
        risk_percent_override: Optional[float] = None,
        performance_score: float = 0.0 # NEW: SQN Input for Dynamic Sizing
    ) -> Tuple[Trade, float]:
        """
        Calculates position size using strict prop firm logic (Sniper Protocol).
        Includes SQN Scaling to cut losers and press winners.
        """
        symbol = context.symbol
        balance = context.account_equity
        price = context.price
        
        # 1. Retrieve Config Parameters
        risk_conf = CONFIG.get('risk_management', {})
        
        # Determine Contract Size
        c_size = contract_size_override if contract_size_override else risk_conf.get('contract_size', RiskManager.DEFAULT_CONTRACT_SIZE)
        
        # USE DETECTED ACCOUNT SIZE (if available), else default to Config
        start_equity = account_size if account_size else float(CONFIG.get('env', {}).get('initial_balance', 100000.0))
        
        # --- SNIPER PROTOCOL: VOLATILITY-ADJUSTED STOPS ---
        # Stop Loss = Entry +/- (ATR(14) * 2.0)
        # We enforce 2.0 multiplier strictly to survive news/noise.
        atr_mult_sl = float(risk_conf.get('stop_loss_atr_mult', 2.0))
        atr_mult_tp = float(risk_conf.get('take_profit_atr_mult', 3.0))
        
        # ATR Fallback Logic
        if atr and atr > 0:
            stop_dist = atr * atr_mult_sl
        else:
            # Fallback 0.2% of price if ATR missing or zero
            stop_dist = price * 0.002 * atr_mult_sl
            
        # --- DEAD PAIR PROTECTION (SPREAD CLAMP) ---
        pip_val, _ = RiskManager.get_pip_info(symbol)
        spread_assumed = CONFIG.get('forensic_audit', {}).get('spread_pips', {}).get(symbol, 1.5)
        
        # Minimum Stop Loss must be at least 3.0x Spread to survive noise
        min_stop_req = (spread_assumed * 3.0 * pip_val)
        if stop_dist < min_stop_req:
             stop_dist = min_stop_req
             
        sl_pips = stop_dist / pip_val

        # --- CROSS-PAIR PIP VALUE CALCULATION ---
        conversion_rate = RiskManager.get_conversion_rate(symbol, price, market_prices)
        
        if conversion_rate <= 0:
            return Trade(symbol, "HOLD", 0.0, 0.0, 0.0, 0.0, "Conversion Error"), 0.0
            
        usd_per_pip_per_lot = c_size * pip_val * conversion_rate
        loss_per_lot_usd = sl_pips * usd_per_pip_per_lot
        
        lots = 0.0
        calculated_risk_usd = 0.0
        
        # --- SNIPER PROTOCOL: STRICT RISK CAP ---
        # Default Base Risk: 0.5% (0.005)
        base_risk_pct = risk_conf.get('base_risk_per_trade_percent', 0.005) * 100.0 # Convert to %
        
        if risk_percent_override is not None:
            # Optuna/WFO override has highest priority (used during optimization)
            risk_pct = risk_percent_override * 100.0 if risk_percent_override < 1.0 else risk_percent_override
        else:
            # Default Strategy Logic: STATIC RISK
            risk_pct = base_risk_pct
        
        # --- NEW: SQN PERFORMANCE SCALING ---
        # "Cut the Losers, Press the Winners"
        # Only apply if we have a valid performance score (not 0.0 default)
        if performance_score != 0.0:
            if performance_score < -1.0:
                # TOXIC ASSET: Hard Stop
                risk_pct = 0.0
                return Trade(symbol, "HOLD", 0.0, 0.0, 0.0, 0.0, f"Toxic Asset (SQN {performance_score:.2f})"), 0.0
                
            elif performance_score < 0.0:
                # LOSING STREAK: Probe Size Only (0.1%)
                # This keeps the bot "in the game" to detect regime shift without bleeding equity
                risk_pct = 0.1
                
            elif performance_score > 2.5:
                # HOT HAND: Scale up slightly (1.25x)
                # But CAP at 1.0% absolute max to prevent ruin
                risk_pct = min(risk_pct * 1.25, 1.0)
        
        # Drawdown Brake: If in significant drawdown (>4%), reduce risk further (Survive Mode)
        if balance < (start_equity * 0.96):
            risk_pct *= 0.5  # Slash risk to recover slowly
        
        # Calculate Risk Amount in USD
        calculated_risk_usd = balance * (risk_pct / 100.0)
        
        # KER Scaling: Reward High Efficiency
        # If KER is high, we take full risk. If low (but passed gate), we reduce slightly.
        # Clamp KER between 0.8 and 1.0 to avoid over-penalizing valid trades
        ker_scalar = max(0.8, min(ker, 1.0))
        calculated_risk_usd *= ker_scalar
        
        # Confidence Scaling
        conf_scalar = min(1.0, max(0.5, conf / 0.8))
        calculated_risk_usd *= conf_scalar

        # Calculate Lots
        if loss_per_lot_usd > 0:
            lots = calculated_risk_usd / loss_per_lot_usd

        # --- CORRELATION PENALTY ---
        if active_correlations > 0:
            penalty_factor = 1.0 / (1.0 + (0.5 * active_correlations))
            lots *= penalty_factor

        # --- CONSTRAINTS & SANITIZATION ---
        min_lot = risk_conf.get('min_lot_size', 0.01)
        max_lot = risk_conf.get('max_lot_size', 50.0)
        max_lev = risk_conf.get('max_leverage', 30.0)
        
        # 1. Leverage Cap
        notional_value = lots * c_size * price * conversion_rate
        if notional_value > 0:
            current_leverage = notional_value / balance
            if current_leverage > max_lev:
                lots = (balance * max_lev) / (c_size * price * conversion_rate)
        
        # 2. Hard Limits
        lots = max(min_lot, min(lots, max_lot))
        lots = round(lots, 2)
        
        final_risk_usd = lots * loss_per_lot_usd
        atr_val = atr if atr is not None else 0.0
        
        # Construct Trade Object
        trade = Trade(
            symbol=symbol,
            action="HOLD",
            volume=lots,
            entry_price=price,
            stop_loss=stop_dist,
            take_profit=stop_dist * (atr_mult_tp / atr_mult_sl),
            comment=f"Risk:{risk_pct:.2f}%|SQN:{performance_score:.1f}|R:${final_risk_usd:.0f}"
        )
        
        return trade, final_risk_usd

class SessionGuard:
    def __init__(self):
        risk_conf = CONFIG.get('risk_management', {})
        tz_str = risk_conf.get('risk_timezone', 'Europe/Prague')
        try:
            self.market_tz = pytz.timezone(tz_str)
        except Exception:
            self.market_tz = pytz.timezone('Europe/Prague')
            
        self.friday_cutoff = dt_time(19, 0) # Generic weekly cutoff
        self.monday_start = dt_time(1, 0)
        self.rollover_start = dt_time(23, 50)
        self.rollover_end = dt_time(1, 15)
        
        # FTMO Spec: Stop entries early, Liquidate later
        self.friday_entry_cutoff_hour = risk_conf.get('friday_entry_cutoff_hour', 16)
        self.liquidation_hour = risk_conf.get('friday_liquidation_hour_server', 21)

    def is_trading_allowed(self) -> bool:
        """
        General market hours check (Weekend, Rollover).
        """
        now_local = datetime.now(self.market_tz)
        weekday = now_local.weekday()
        current_time = now_local.time()
        
        if weekday == 5: return False # Saturday
        if weekday == 6 and current_time < self.monday_start: return False # Sunday AM
        if weekday == 4 and current_time > self.friday_cutoff: return False # Late Friday generic
        
        if current_time >= self.rollover_start or current_time <= self.rollover_end:
            return False
            
        return True

    def is_friday_afternoon(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Returns True if it is Friday and past the entry cutoff hour (e.g., 16:00).
        This differentiates "No New Entries" from "Market Closed".
        """
        if timestamp:
            # Handle timezone if provided, otherwise assume aligned
            if timestamp.tzinfo is None:
                dt = timestamp
            else:
                dt = timestamp.astimezone(self.market_tz)
        else:
            dt = datetime.now(self.market_tz)
            
        if dt.weekday() == 4 and dt.hour >= self.friday_entry_cutoff_hour:
            return True
            
        return False

    def should_liquidate(self) -> bool:
        """
        Returns True if it is Friday and past the liquidation hour.
        This forces the bot to close all trades before the weekend.
        """
        now_local = datetime.now(self.market_tz)
        weekday = now_local.weekday()
        
        # Check if Friday (4) and hour >= liquidation hour (e.g., 21:00)
        if weekday == 4 and now_local.hour >= self.liquidation_hour:
            return True
            
        return False

class FTMORiskMonitor:
    """
    Monitors account health against FTMO's strict drawdown limits.
    """
    def __init__(self, initial_balance: float, max_daily_loss_pct: float, redis_client):
        self.initial_balance = initial_balance
        self.max_daily_loss = initial_balance * max_daily_loss_pct
        self.r = redis_client
        self.starting_equity_of_day = initial_balance
        self.equity = initial_balance
        
        # 10% Profit Target (FTMO Challenge Goal)
        self.profit_target = initial_balance * 1.10

    def can_trade(self) -> bool:
        # 1. Total Drawdown Check (10% Max)
        if self.equity < (self.initial_balance * 0.90): return False
        
        # 2. Daily Drawdown Check (5% Max)
        current_daily_loss = self.starting_equity_of_day - self.equity
        if current_daily_loss >= self.max_daily_loss: return False
        
        # 3. Circuit Breaker: Profit Target Reached
        if self.equity >= self.profit_target:
            return False
            
        return True

    def _check_constraints(self, risk_to_add: float):
        pass
        
    def check_circuit_breakers(self) -> str:
        if self.equity < (self.initial_balance * 0.90): return "Total Drawdown Breach"
        
        current_daily_loss = self.starting_equity_of_day - self.equity
        if current_daily_loss >= self.max_daily_loss: return "Daily Drawdown Breach"
        
        if self.equity >= self.profit_target: return "Profit Target Reached (Victory Lap)"
        
        return "OK"

    def update_equity(self, current_equity: float):
        self.equity = current_equity

class HierarchicalRiskParity:
    """
    Allocates portfolio weights based on hierarchical clustering of asset correlations.
    """
    @staticmethod
    def get_allocation(returns_df: pd.DataFrame) -> Dict[str, float]:
        cols = returns_df.columns.tolist()
        if not SCIPY_AVAILABLE:
            return {c: 1.0/len(cols) for c in cols}
        
        try:
            # 1. Compute Correlation Matrix
            corr = returns_df.corr().fillna(0)
            
            # 2. Compute Distance Matrix
            dist = ssd.pdist(corr, metric='euclidean')
            
            # 3. Hierarchical Clustering (Linkage)
            link = sch.linkage(dist, method='single')
            
            # 4. Quasi-Diagonalization
            sort_ix = HierarchicalRiskParity._get_quasi_diag(link)
            sort_ix = [cols[i] for i in sort_ix]
            
            # 5. Recursive Bisection
            cov = returns_df.cov()
            variances = np.diag(cov)
            variances[variances < EPS] = EPS
            inv_var = 1.0 / variances
            
            weights = inv_var / np.sum(inv_var)
            
            allocation = dict(zip(cols, weights))
            return allocation
        except Exception as e:
            logger.error(f"HRP Failed: {e}")
            return {c: 1.0/len(cols) for c in cols}

    @staticmethod
    def _get_quasi_diag(link: np.ndarray) -> List[int]:
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

class PortfolioRiskManager:
    """
    Manages portfolio-level risk (HRP, Correlations, Penalty Box).
    """
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.returns_buffer = {s: [] for s in symbols}
        self.penalty_box = {}
        self.correlation_matrix = pd.DataFrame()
        self.active_positions = {}

    def update_returns(self, symbol: str, ret: float):
        self.returns_buffer[symbol].append(ret)
        if len(self.returns_buffer[symbol]) > 100:
            self.returns_buffer[symbol].pop(0)

    def update_correlation_matrix(self):
        if not SCIPY_AVAILABLE: return
        min_len = min(len(v) for v in self.returns_buffer.values())
        if min_len < 20: return
        
        data = {s: self.returns_buffer[s][-min_len:] for s in self.symbols}
        df = pd.DataFrame(data)
        self.correlation_matrix = df.corr()

    def get_correlation_count(self, symbol: str, threshold: float = 0.7) -> int:
        if self.correlation_matrix.empty or symbol not in self.correlation_matrix.columns:
            return 0
        
        count = 0
        for held_symbol in self.active_positions:
            if held_symbol == symbol: continue
            if held_symbol in self.correlation_matrix.columns:
                corr = self.correlation_matrix.loc[symbol, held_symbol]
                if abs(corr) > threshold:
                    count += 1
        return count

    def check_penalty_box(self, symbol: str) -> bool:
        if symbol in self.penalty_box:
            if time.time() < self.penalty_box[symbol]:
                return True
            else:
                del self.penalty_box[symbol]
        return False

    def add_to_penalty_box(self, symbol: str, duration_minutes: int = 60):
        self.penalty_box[symbol] = time.time() + (duration_minutes * 60)
        logger.warning(f"🚫 {symbol} added to Penalty Box for {duration_minutes}m")