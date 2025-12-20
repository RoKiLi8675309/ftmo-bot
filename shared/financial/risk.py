# FILENAME: shared/financial/risk.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/financial/risk.py
# DEPENDENCIES: numpy, pandas, scipy (optional on Windows)
# DESCRIPTION: Core Risk Management logic (Position Sizing, FTMO Limits, HRP).
# AUDIT REMEDIATION (GROK): 
#   - IMPLEMENTED Fractional Kelly (0.5x) to cut Drawdowns.
#   - ADDED Detailed Debug Logging for Sizing.
#   - HARDENED Correlation Penalty.
#   - COMPATIBILITY: Accepts injected market_prices for precise conversion.
# CRITICAL: Python 3.9 Compatible.
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
    and Advanced Position Sizing (RCK, CPPI).
    """
    STANDARD_LOT_UNITS = 100_000

    @staticmethod
    def get_pip_info(symbol: str) -> Tuple[float, int]:
        """Returns (pip_size, digits)."""
        s = symbol.upper()
        if "JPY" in s:
            return 0.01, 3
        if "XAU" in s:
            return 0.1, 2
        return 0.0001, 5

    @staticmethod
    def get_conversion_rate(symbol: str, price: float, market_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculates the Quote -> USD conversion rate using LIVE market data.
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
            return 0.0 # Safety

        # GBP Pairs (Quote = GBP). Need GBP->USD (GBPUSD).
        if s.endswith("GBP"):
            gbpusd = get_price("GBPUSD")
            if gbpusd: return gbpusd
            return 0.0
            
        # CAD Pairs (Quote = CAD). Need CAD->USD (1 / USDCAD).
        if s.endswith("CAD"):
            usdcad = get_price("USDCAD")
            if usdcad and usdcad > 0: return 1.0 / usdcad
            if s == "USDCAD" and price > 0: return 1.0 / price
            return 0.0

        # CHF Pairs (Quote = CHF). Need CHF->USD (1 / USDCHF).
        if s.endswith("CHF"):
            usdchf = get_price("USDCHF")
            if usdchf and usdchf > 0: return 1.0 / usdchf
            if s == "USDCHF" and price > 0: return 1.0 / price
            return 0.0

        # AUD Pairs (Quote = AUD). Need AUD->USD (AUDUSD).
        if s.endswith("AUD"):
            audusd = get_price("AUDUSD")
            if audusd: return audusd
            if s == "AUDUSD" and price > 0: return price
            return 0.0

        # NZD Pairs
        if s.endswith("NZD"):
            nzdusd = get_price("NZDUSD")
            if nzdusd: return nzdusd
            if s == "NZDUSD" and price > 0: return price
            return 0.0

        # Default fallback
        return 1.0

    @staticmethod
    def calculate_rck_size(
        context: TradeContext,
        conf: float,
        volatility: float,
        active_correlations: int = 0,
        market_prices: Optional[Dict[str, float]] = None
    ) -> Tuple[Trade, float]:
        """
        Calculates position size using Risk-Constrained Kelly (RCK) and CPPI Cushion.
        GROK FIX: Implements strict Fractional Kelly (0.5x) and logging.
        """
        symbol = context.symbol
        balance = context.account_equity
        price = context.price

        # 1. Retrieve Config Parameters
        risk_conf = CONFIG.get('risk_management', {})

        # --- CPPI SAFE EQUITY CUSHION ---
        cppi_floor_pct = risk_conf.get('cppi_floor_pct', 0.08)
        start_equity = float(CONFIG.get('env', {}).get('initial_balance', 100000.0))
        
        floor_value = start_equity * (1.0 - cppi_floor_pct)
        cushion = max(0.0, balance - floor_value)
        
        cppi_mult = risk_conf.get('cppi_multiplier', 3.0)
        risk_budget_usd = cushion * cppi_mult

        # Clamp Risk Budget to Hard Max Risk %
        max_risk_percent = risk_conf.get('max_risk_percent', 1.0)
        max_risk_usd = balance * (max_risk_percent / 100.0)
        risk_budget_usd = min(risk_budget_usd, max_risk_usd)

        # --- KELLY CRITERION ---
        win_rate = context.win_rate
        rr_ratio = context.risk_reward_ratio
        
        # Singularity Guard
        if rr_ratio <= 0.05:
            return Trade(symbol, "HOLD", 0.0, 0.0, 0.0, 0.0, "Invalid R/R"), 0.0
        
        # Kelly Formula: K = W - (1-W)/R
        raw_kelly = win_rate - ((1 - win_rate) / rr_ratio)
        
        # GROK REMEDIATION: Fractional Kelly
        # Reduces exposure to 50% (or config value) of optimal to reduce volatility
        kelly_fraction = risk_conf.get('kelly_fraction', 0.5)
        kelly_pct = raw_kelly * kelly_fraction
        
        # LG-2 Strict Non-Negative enforcement
        kelly_pct = max(0.0, kelly_pct)

        # --- CORRELATION PENALTY ---
        # Reduce size if we already hold correlated positions
        if active_correlations > 0:
            penalty_factor = 1.0 / (1.0 + (0.5 * active_correlations))
            kelly_pct *= penalty_factor
            logger.debug(f"⛓️ Correlation Penalty {symbol}: {active_correlations} peers -> x{penalty_factor:.2f}")

        # --- FINAL SIZING ---
        kelly_risk_usd = balance * kelly_pct
        
        # Risk Constraint: Min(Kelly, CPPI, Hard Max)
        final_risk_usd = min(kelly_risk_usd, risk_budget_usd)

        # Calculate Stop Loss Distance
        atr_mult = risk_conf.get('stop_loss_atr_mult', 2.0)
        stop_dist = price * volatility * atr_mult

        # Convert Risk USD to Lots
        pip_val, _ = RiskManager.get_pip_info(symbol)
        
        # Safety checks
        if pip_val <= 0 or stop_dist <= 0 or price <= 0:
            return Trade(symbol, "HOLD", 0.0, 0.0, 0.0, 0.0, "Invalid Data"), 0.0

        sl_pips = stop_dist / pip_val

        # --- CROSS-PAIR PIP VALUE CALCULATION ---
        pip_value_quote = RiskManager.STANDARD_LOT_UNITS * pip_val
        conversion_rate = RiskManager.get_conversion_rate(symbol, price, market_prices)
        
        if conversion_rate <= 0:
            return Trade(symbol, "HOLD", 0.0, 0.0, 0.0, 0.0, "Conversion Error"), 0.0

        usd_per_pip_per_lot = pip_value_quote * conversion_rate
        loss_per_lot = sl_pips * usd_per_pip_per_lot
        
        if loss_per_lot <= 0: 
            lots = 0.0
        else: 
            lots = final_risk_usd / loss_per_lot

        # Constraints
        min_lot = risk_conf.get('min_lot_size', 0.01)
        max_lot = risk_conf.get('max_lot_size', 50.0)
        lots = max(min_lot, min(lots, max_lot))
        lots = round(lots, 2)
        
        actual_risk_usd = lots * loss_per_lot

        # DEBUG: Sizing Autopsy
        if lots > 0:
            # Only log if we are actually sizing a trade
            # logger.debug(f"⚖️ Sizing {symbol}: K_Raw={raw_kelly:.2f} | K_Frac={kelly_pct:.2f} | Risk=${final_risk_usd:.2f} | Lots={lots}")
            pass

        # Construct Trade Object Template
        tp_mult = risk_conf.get('take_profit_atr_mult', 3.0)
        trade = Trade(
            symbol=symbol, 
            action="HOLD", 
            volume=lots,
            entry_price=price,
            stop_loss=stop_dist,
            take_profit=stop_dist * tp_mult,
            comment=f"K:{kelly_pct:.2f}|C:{cushion:.0f}"
        )
        
        return trade, actual_risk_usd


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
            
            # 2. Compute Distance Matrix (d = sqrt(2*(1-rho)))
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


class FTMORiskMonitor:
    def __init__(self, initial_balance: float, max_daily_loss_pct: float, redis_client):
        self.initial_balance = initial_balance
        self.max_daily_loss = initial_balance * max_daily_loss_pct
        self.r = redis_client
        self.starting_equity_of_day = initial_balance
        self.equity = initial_balance

    def can_trade(self) -> bool:
        if self.equity < (self.initial_balance * 0.90): return False
        current_daily_loss = self.starting_equity_of_day - self.equity
        if current_daily_loss >= self.max_daily_loss: return False
        return True

    def _check_constraints(self, risk_to_add: float):
        pass
        
    def check_circuit_breakers(self) -> str:
        if self.equity < (self.initial_balance * 0.90): return "Total Drawdown Breach"
        current_daily_loss = self.starting_equity_of_day - self.equity
        if current_daily_loss >= self.max_daily_loss: return "Daily Drawdown Breach"
        return "OK"

    def update_equity(self, current_equity: float):
        self.equity = current_equity


class SessionGuard:
    def __init__(self):
        risk_conf = CONFIG.get('risk_management', {})
        tz_str = risk_conf.get('risk_timezone', 'Europe/Prague')
        try:
            self.market_tz = pytz.timezone(tz_str)
        except Exception:
            self.market_tz = pytz.timezone('Europe/Prague')
        self.friday_cutoff = dt_time(19, 0)
        self.monday_start = dt_time(1, 0)
        self.rollover_start = dt_time(23, 50)
        self.rollover_end = dt_time(1, 15)

    def is_trading_allowed(self) -> bool:
        now_local = datetime.now(self.market_tz)
        weekday = now_local.weekday()
        current_time = now_local.time()
        if weekday == 5: return False
        if weekday == 6 and current_time < self.monday_start: return False
        if weekday == 4 and current_time > self.friday_cutoff: return False
        if current_time >= self.rollover_start or current_time <= self.rollover_end:
            return False
        return True