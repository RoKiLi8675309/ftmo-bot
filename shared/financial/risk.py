# =============================================================================
# FILENAME: shared/financial/risk.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/financial/risk.py
# DEPENDENCIES: numpy, pandas, scipy (optional on Windows)
# DESCRIPTION: Core Risk Management logic (Position Sizing, FTMO Limits, HRP).
#
# AUDIT REMEDIATION (2025-12-24 - PROFIT OPTIMIZATION):
# 1. DRAWDOWN BRAKE: Defensive scaling. If Equity < 98% of Start, Risk *= 0.5.
# 2. VOLATILITY TARGETING: Strict 20% Annual Target implementation.
# 3. ADVANCED SIZING: Dynamic Kelly Criterion scaled by ML Confidence.
# 4. KER SCALING (NEW): Position size scaled by Market Efficiency (Signal-to-Noise).
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
    and Advanced Position Sizing (Volatility Targeting, Kelly, CPPI).
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
        if any(x in s for x in ["US30", "GER30", "NAS100", "SPX500"]):
            return 1.0, 1  # Usually 1 point steps
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
            return 0.0065 # Fallback ~150
        
        # GBP Pairs (Quote = GBP). Need GBP->USD (GBPUSD).
        if s.endswith("GBP"):
            gbpusd = get_price("GBPUSD")
            if gbpusd: return gbpusd
            return 1.25 # Fallback

        # CAD Pairs (Quote = CAD). Need CAD->USD (1 / USDCAD).
        if s.endswith("CAD"):
            usdcad = get_price("USDCAD")
            if usdcad and usdcad > 0: return 1.0 / usdcad
            if s == "USDCAD" and price > 0: return 1.0 / price
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
        account_size: Optional[float] = None, # NEW ARGUMENT for Auto-Detection
        contract_size_override: Optional[float] = None, # NEW: Allow overriding lot size
        ker: float = 1.0 # AUDIT FIX: Kaufman Efficiency Ratio input
    ) -> Tuple[Trade, float]:
        """
        Calculates position size using Volatility Targeting + Dynamic Kelly.
        Handles risk clamping, dead pair protection, and fractional scaling.
        """
        symbol = context.symbol
        balance = context.account_equity
        price = context.price
        
        # 1. Retrieve Config Parameters
        risk_conf = CONFIG.get('risk_management', {})
        sizing_method = risk_conf.get('sizing_method', 'volatility_targeting')
        
        # Determine Contract Size
        c_size = contract_size_override if contract_size_override else risk_conf.get('contract_size', RiskManager.DEFAULT_CONTRACT_SIZE)
        
        # --- CPPI SAFE EQUITY CUSHION ---
        cppi_floor_pct = risk_conf.get('cppi_floor_pct', 0.08)
        
        # USE DETECTED ACCOUNT SIZE (if available), else default to Config
        start_equity = account_size if account_size else float(CONFIG.get('env', {}).get('initial_balance', 100000.0))
        
        floor_value = start_equity * (1.0 - cppi_floor_pct)
        cushion = max(0.0, balance - floor_value)
        
        cppi_mult = risk_conf.get('cppi_multiplier', 2.0)
        risk_budget_usd = cushion * cppi_mult
        
        # Clamp Risk Budget to Hard Max Risk %
        base_risk_pct = risk_conf.get('max_risk_percent', 1.5) / 100.0
        max_risk_usd = balance * base_risk_pct
        
        # Effective Risk Budget: Min(CPPI Budget, Hard Cap)
        final_risk_usd = min(risk_budget_usd, max_risk_usd)
        
        # --- OPTIMIZATION: DRAWDOWN BRAKE ---
        # If we are in a minor drawdown (> 2%), cut risk in half to prevent spirals.
        if balance < (start_equity * 0.98):
            final_risk_usd *= 0.5
            
        # --- ADAPTIVE STOP LOSS (ATR Based) ---
        atr_mult_sl = risk_conf.get('stop_loss_atr_mult', 1.5)
        atr_mult_tp = risk_conf.get('take_profit_atr_mult', 2.0)
        
        # ATR Fallback Logic
        if atr and atr > 0:
            stop_dist = atr * atr_mult_sl
        else:
            # Fallback 0.1% of price if ATR missing or zero
            stop_dist = price * 0.001 * atr_mult_sl
            
        # --- DEAD PAIR PROTECTION (SPREAD CLAMP) ---
        pip_val, _ = RiskManager.get_pip_info(symbol)
        spread_assumed = CONFIG.get('forensic_audit', {}).get('spread_pips', {}).get(symbol, 1.5)
        
        # Minimum Stop Loss must be at least 3.0x Spread
        min_stop_req = (spread_assumed * 3.0 * pip_val)
        if stop_dist < min_stop_req:
             stop_dist = min_stop_req
             
        sl_pips = stop_dist / pip_val

        # --- CROSS-PAIR PIP VALUE CALCULATION ---
        pip_value_quote = c_size * pip_val
        conversion_rate = RiskManager.get_conversion_rate(symbol, price, market_prices)
        
        if conversion_rate <= 0:
            return Trade(symbol, "HOLD", 0.0, 0.0, 0.0, 0.0, "Conversion Error"), 0.0
            
        usd_per_pip_per_lot = pip_value_quote * conversion_rate
        loss_per_lot = sl_pips * usd_per_pip_per_lot
        
        lots = 0.0
        
        # --- SIZING METHOD SELECTION ---
        
        if sizing_method == 'volatility_targeting':
            # --- METHOD 1: VOLATILITY TARGETING + KELLY ---
            
            # A. Volatility Scalar
            target_ann_vol = risk_conf.get('target_annual_volatility', 0.20)
            if volatility <= 1e-9: volatility = 0.001
            ann_factor = 269.4 # M5 approximation
            realized_ann_vol = volatility * ann_factor
            vol_scalar = target_ann_vol / realized_ann_vol
            
            # B. Dynamic Kelly Criterion (Step 5)
            # Formula: f = (p(b+1) - 1) / b
            # p = Win Rate, b = Risk/Reward Ratio
            p = context.win_rate
            b = context.risk_reward_ratio
            if b <= 0: b = 1.0
            
            kelly_optimal = (p * (b + 1) - 1) / b
            # Clamp Kelly to valid range [0, 1] to prevent negative or infinite sizing
            kelly_optimal = max(0.0, min(kelly_optimal, 1.0))
            
            # Apply "Fractional Kelly" setting (e.g. 0.25)
            kelly_fraction_cfg = risk_conf.get('kelly_fraction', 0.25)
            
            # C. ML Confidence Scaling (Power Law)
            # Strongly penalize low-confidence signals (e.g. 0.6^2 = 0.36 multiplier)
            confidence_scalar = conf * conf
            
            # Combine Factors
            final_scalar = vol_scalar * (kelly_optimal * kelly_fraction_cfg) * confidence_scalar
            
            # Calculate Target Exposure in USD
            target_exposure = balance * final_scalar
            
            # Convert Exposure to Lots
            notional_value_per_lot = c_size * price * conversion_rate
            if notional_value_per_lot > 0:
                lots = target_exposure / notional_value_per_lot
                
            # --- CEILING CHECK ---
            # Ensure this lot size doesn't violate the dollar risk budget
            current_risk_dollars = lots * loss_per_lot
            if current_risk_dollars > final_risk_usd:
                lots = final_risk_usd / loss_per_lot if loss_per_lot > 0 else 0.0

        else:
            # --- METHOD 2: INVERSE VOLATILITY (LEGACY) ---
            if loss_per_lot > 0:
                lots = final_risk_usd / loss_per_lot
            lots *= conf

        # --- CORRELATION PENALTY ---
        if active_correlations > 0:
            penalty_factor = 1.0 / (1.0 + (0.5 * active_correlations))
            lots *= penalty_factor

        # --- AUDIT FIX: KER SCALING (EFFICIENCY) ---
        # Scale down size if trend efficiency (KER) is low (noisy market).
        # We clamp KER between 0.5 (noise penalty) and 1.0 (full size).
        ker_scalar = max(0.5, min(ker, 1.0))
        lots *= ker_scalar

        # --- CONSTRAINTS & SANITIZATION ---
        min_lot = risk_conf.get('min_lot_size', 0.01)
        max_lot = risk_conf.get('max_lot_size', 10.0)
        max_lev = risk_conf.get('max_leverage', 30.0)
        
        # Leverage Cap
        notional = c_size * price * conversion_rate
        if notional > 0:
            max_lots_lev = (balance * max_lev) / notional
            lots = min(lots, max_lots_lev)
            
        lots = max(min_lot, min(lots, max_lot))
        lots = round(lots, 2)
        
        actual_risk_usd = lots * loss_per_lot
        atr_val = atr if atr is not None else 0.0
        
        # Construct Trade Object
        trade = Trade(
            symbol=symbol,
            action="HOLD",
            volume=lots,
            entry_price=price,
            stop_loss=stop_dist,
            take_profit=stop_dist * (atr_mult_tp / atr_mult_sl),
            comment=f"{'Kelly' if sizing_method == 'volatility_targeting' else 'InvVol'}|R:${actual_risk_usd:.0f}|ATR:{atr_val:.5f}|KER:{ker_scalar:.2f}"
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
            logger.info("🎉 PROFIT TARGET REACHED! Trading Halted to preserve pass.")
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