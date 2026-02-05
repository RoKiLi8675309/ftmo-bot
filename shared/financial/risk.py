# =============================================================================
# FILENAME: shared/financial/risk.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/financial/risk.py
# DEPENDENCIES: numpy, pandas, scipy (optional on Windows)
# DESCRIPTION: Core Risk Management logic (Position Sizing, FTMO Limits, HRP).
#
# PHOENIX V16.4 UPDATE (SURVIVAL PROTOCOL):
# 1. REMOVED ALPHA BOOST: Deleted hardcoded risk multipliers for specific pairs.
# 2. RISK CAPS: Strict 0.5% hard cap on "Hot Hand" scaling.
# 3. MARGIN CLAMP: Preserved logic to prevent "Not Enough Money" errors.
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
            return 0.0
        
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
            
        # Default fallback for unknown pairs
        return 0.0

    @staticmethod
    def calculate_required_margin(symbol: str, lots: float, price: float, contract_size: float, conversion_rate: float) -> float:
        """
        V16.11 FIX: Enforce correct leverage tiers for FTMO/Prop Firms.
        Formula: (Lots * Contract_Size * Price * Conversion_Rate) / Leverage
        """
        # 1. Identify Asset Class Leverage
        risk_conf = CONFIG.get('risk_management', {})
        lev_map = risk_conf.get('leverage', {})
        
        # FTMO Standard Leverage (Swing Account is 1:30, Normal is 1:100)
        # We assume 1:30 to be SAFE unless specified
        leverage = float(lev_map.get('default', 30.0))
        
        s = symbol.upper()
        
        # Forex Minors/Crosses often typically 1:100 or 1:30
        if "JPY" in s and not "USD" in s and not "GBP" in s and not "EUR" in s:
            leverage = float(lev_map.get('minor', 30.0)) 
        
        # Metals (Gold/Silver)
        if "XAU" in s or "XAG" in s:
            leverage = float(lev_map.get('gold', 20.0))
            
        # Indices (US30, NAS100) - Usually 1:50 or 1:20
        if any(x in s for x in ["US30", "GER30", "NAS100", "SPX500"]):
            leverage = float(lev_map.get('indices', 20.0))
            
        # Crypto (BTC/ETH) - VERY LOW LEVERAGE (1:2 or 1:5)
        if "BTC" in s or "ETH" in s:
            leverage = float(lev_map.get('crypto', 2.0))
            
        if leverage <= 0: leverage = 1.0 # Safety fallback (1:1)
        
        # 2. Calculate Notional Value in Account Currency (USD)
        notional_value = lots * contract_size * price * conversion_rate
        
        # 3. Required Margin
        margin = notional_value / leverage
        return margin

    @staticmethod
    def _calculate_max_margin_volume(symbol: str, free_margin: float, contract_size: float, price: float, conversion_rate: float) -> float:
        """
        V16.11: MARGIN CLAMP.
        Calculates the maximum lot size allowed by available free margin.
        Formula: MaxVol = (FreeMargin * Leverage) / (ContractSize * Price * ConvRate)
        """
        if free_margin <= 0: return 0.0

        risk_conf = CONFIG.get('risk_management', {})
        lev_map = risk_conf.get('leverage', {})

        # Determine Leverage (Consistent with calculate_required_margin)
        lev = float(lev_map.get('default', 30.0))
        s = symbol.upper()
        
        if "JPY" in s and not "USD" in s and not "GBP" in s and not "EUR" in s:
            lev = float(lev_map.get('minor', 30.0))
        if any(pair in s for pair in ["GBPAUD", "AUDJPY", "EURAUD", "GBPNZD", "GBPJPY"]):
            lev = float(lev_map.get('minor', 30.0))
        if "XAU" in s or "XAG" in s:
            lev = float(lev_map.get('gold', 20.0))
        elif "BTC" in s or "ETH" in s:
            lev = float(lev_map.get('crypto', 2.0))
        elif any(x in s for x in ["US30", "GER30", "NAS100", "SPX500"]):
            lev = float(lev_map.get('indices', 20.0))

        if lev <= 0: return 0.0

        # Calculate Margin per 1 Lot
        # Margin = (1 * Contract * Price * Conv) / Lev
        one_lot_margin = (contract_size * price * conversion_rate) / lev

        if one_lot_margin <= 0: return 0.0

        # Max Volume = Free Margin / One Lot Margin
        # Apply 95% Safety Factor to prevent immediate margin call on spread
        max_vol = (free_margin * 0.95) / one_lot_margin

        return max_vol

    @staticmethod
    def calculate_rck_size(
        context: TradeContext, 
        conf: float, 
        volatility: float,
        active_correlations: int,
        market_prices: Optional[Dict[str, float]] = None,
        atr: Optional[float] = None,
        ker: float = 0.0,
        account_size: Optional[float] = None, 
        contract_size_override: Optional[float] = None, 
        risk_percent_override: Optional[float] = None,
        performance_score: float = 0.0, # SQN or Sharpe
        daily_pnl_pct: float = 0.0,
        current_open_risk_pct: float = 0.0,
        free_margin: float = 999999.0 
    ) -> Tuple[Trade, float]:
        """
        ADVANCED POSITION SIZING KERNEL (V16.4 SURVIVAL PROTOCOL).
        Integrates RCK (Risk-Confidence-Kelly) Optimization with Margin Guards.
        REMOVED: Alpha Squad Boost (Reckless).
        ADDED: Strict caps on scaling.
        """
        symbol = context.symbol
        balance = context.account_equity
        price = context.price
        
        # Safety: Zero Balance Check
        if balance <= 0:
             return Trade(symbol, "HOLD", 0.0, 0.0, 0.0, 0.0, "Zero Balance"), 0.0

        # 1. Retrieve Config Parameters
        risk_conf = CONFIG.get('risk_management', {})
        
        # Determine Contract Size
        c_size = contract_size_override if contract_size_override else risk_conf.get('contract_size', RiskManager.DEFAULT_CONTRACT_SIZE)
        
        # USE DETECTED ACCOUNT SIZE (if available), else default to Config
        start_equity = account_size if account_size else float(CONFIG.get('env', {}).get('initial_balance', 100000.0))
        
        # --- SNIPER PROTOCOL: VOLATILITY-ADJUSTED STOPS ---
        atr_mult_sl = float(risk_conf.get('stop_loss_atr_mult', 1.5)) 
        atr_mult_tp = float(risk_conf.get('take_profit_atr_mult', 3.0)) # V16.0: 3R Target for Scalping
        
        # ATR Fallback Logic
        if atr and atr > 0:
            stop_dist = atr * atr_mult_sl
        else:
            stop_dist = price * 0.002 * atr_mult_sl
            
        # --- DEAD PAIR PROTECTION (SPREAD CLAMP) ---
        pip_val_raw, _ = RiskManager.get_pip_info(symbol)
        spread_assumed = CONFIG.get('forensic_audit', {}).get('spread_pips', {}).get(symbol, 1.5)
        
        # Minimum Stop Loss must be at least 3.0x Spread to survive noise
        min_stop_req = (spread_assumed * 3.0 * pip_val_raw)
        if stop_dist < min_stop_req:
             stop_dist = min_stop_req
             
        sl_pips = stop_dist / pip_val_raw

        # --- CROSS-PAIR PIP VALUE CALCULATION ---
        conversion_rate = RiskManager.get_conversion_rate(symbol, price, market_prices)
        
        # FAIL-SAFE: If conversion fails, DO NOT TRADE.
        if conversion_rate <= 0:
            return Trade(symbol, "HOLD", 0.0, 0.0, 0.0, 0.0, "Conversion Error (No Rate)"), 0.0
            
        usd_per_pip_per_lot = c_size * pip_val_raw * conversion_rate
        loss_per_lot_usd = sl_pips * usd_per_pip_per_lot
        
        lots = 0.0
        calculated_risk_usd = 0.0
        
        # --- V16.4: SURVIVAL RISK PARAMETERS ---
        default_base_risk = risk_conf.get('base_risk_per_trade_percent', 0.0025) # 0.25%
        buffer_threshold = risk_conf.get('profit_buffer_threshold', 0.02)
        scaled_risk_val = risk_conf.get('scaled_risk_percent', 0.005) # 0.5% Cap
        
        scaling_comment = ""
        
        if risk_percent_override is not None:
            # Optuna/WFO override has highest priority
            risk_pct = risk_percent_override * 100.0 if risk_percent_override < 1.0 else risk_percent_override
        else:
            # Default Strategy Logic with Buffer Scaling
            if daily_pnl_pct >= buffer_threshold:
                risk_pct = scaled_risk_val * 100.0
                scaling_comment = "|Buf:ON"
            else:
                risk_pct = default_base_risk * 100.0
        
        # --- SQN PERFORMANCE SCALING (V16.4: CAPPED) ---
        if performance_score != 0.0:
            if performance_score < -2.0:
                # TOXIC ASSET RECOVERY PROTOCOL
                risk_pct = 0.1 # Probe size
                scaling_comment += "|Toxic:Probe"
                
            elif performance_score < 0.0:
                # LOSING STREAK: Probe Size Only (0.25%)
                risk_pct = 0.25
                scaling_comment += "|SQN:Low"
                
            elif performance_score > 2.5:
                # HOT HAND: Scale up slightly, but HARD CAP at 0.5%
                risk_pct = min(risk_pct * 1.1, 0.5) 
                scaling_comment += "|SQN:High"
        
        # --- REMOVED ALPHA SQUAD BOOST ---
        # No arbitrary multipliers based on symbol name. Code must survive on merit.

        # --- ASYMPTOTIC DECAY ---
        daily_limit_pct = risk_conf.get('max_daily_loss_pct', 0.040) 
        
        if daily_pnl_pct < 0:
            current_loss_pct = abs(daily_pnl_pct)
            remaining_buffer = daily_limit_pct - current_loss_pct
            
            if remaining_buffer < 0.02: 
                decay_risk_pct = (remaining_buffer / 2.0) * 100.0
                if decay_risk_pct < risk_pct:
                    risk_pct = max(0.0, decay_risk_pct)
                    scaling_comment += "|Decay:ON"

        # --- TOTAL PORTFOLIO RISK CAP ---
        max_total_risk_pct = float(risk_conf.get('max_risk_percent', 1.5)) # Reduced cap
        potential_total_risk = current_open_risk_pct + (risk_pct / 100.0) * 100.0 
        
        if potential_total_risk > max_total_risk_pct:
            # Clamp risk to remaining capacity
            allowed_risk = max_total_risk_pct - current_open_risk_pct
            if allowed_risk < 0.1: # If less than 0.1% risk allowed, skip trade
                return Trade(symbol, "HOLD", 0.0, 0.0, 0.0, 0.0, f"Max Risk Cap Hit ({current_open_risk_pct:.2f}%)"), 0.0
            
            # Apply Clamp
            risk_pct = allowed_risk
            scaling_comment += f"|Cap:{risk_pct:.2f}%"

        # Calculate Risk Amount in USD
        calculated_risk_usd = balance * (risk_pct / 100.0)
        
        # KER Scaling (Efficiency)
        ker_scalar = max(0.8, min(ker, 1.0))
        calculated_risk_usd *= ker_scalar
        
        # Calculate Lots
        if loss_per_lot_usd > 0:
            lots = calculated_risk_usd / loss_per_lot_usd

        # --- CORRELATION PENALTY ---
        if active_correlations > 0:
            penalty_factor = 1.0 / (1.0 + (0.5 * active_correlations))
            lots *= penalty_factor

        # --- MARGIN CLAMP (V16.11) ---
        # Calculate max volume allowed by Free Margin
        max_margin_lots = RiskManager._calculate_max_margin_volume(symbol, free_margin, c_size, price, conversion_rate)

        if lots > max_margin_lots:
            logger.warning(f"⚠️ MARGIN CLAMP: {symbol} Lots {lots:.2f} -> {max_margin_lots:.2f} (Free Margin ${free_margin:.0f})")
            lots = max_margin_lots
            scaling_comment += "|LevGuard"

        # --- FINAL HARD LIMITS ---
        min_lot = risk_conf.get('min_lot_size', 0.01)
        max_lot = risk_conf.get('max_lot_size', 50.0)
        
        lots = max(min_lot, min(lots, max_lot))
        lots = round(lots, 2)
        
        final_risk_usd = lots * loss_per_lot_usd
        
        # Construct Trade Object
        trade = Trade(
            symbol=symbol,
            action="HOLD",
            volume=lots,
            entry_price=price,
            stop_loss=stop_dist,
            take_profit=stop_dist * (atr_mult_tp / atr_mult_sl),
            comment=f"Risk:{risk_pct:.2f}%{scaling_comment}|SQN:{performance_score:.1f}|R:${final_risk_usd:.0f}"
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
            
        self.friday_cutoff = dt_time(19, 0)
        self.monday_start = dt_time(1, 0)
        self.rollover_start = dt_time(23, 50)
        self.rollover_end = dt_time(1, 15)
        
        self.friday_entry_cutoff_hour = risk_conf.get('friday_entry_cutoff_hour', 16)
        self.liquidation_hour = risk_conf.get('friday_liquidation_hour_server', 21)

    def is_trading_allowed(self) -> bool:
        """General market hours check."""
        now_local = datetime.now(self.market_tz)
        weekday = now_local.weekday()
        current_time = now_local.time()
        
        if weekday == 5: return False # Saturday
        if weekday == 6 and current_time < self.monday_start: return False # Sunday AM
        if weekday == 4 and current_time > self.friday_cutoff: return False # Late Friday
        
        if current_time >= self.rollover_start or current_time <= self.rollover_end:
            return False
            
        return True

    def is_friday_afternoon(self, timestamp: Optional[datetime] = None) -> bool:
        if timestamp:
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
        now_local = datetime.now(self.market_tz)
        weekday = now_local.weekday()
        
        if weekday > 4: return True # Weekend
        if weekday == 4 and now_local.hour >= self.liquidation_hour: return True
            
        return False

class FTMORiskMonitor:
    def __init__(self, initial_balance: float, max_daily_loss_pct: float, redis_client):
        self.initial_balance = initial_balance
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_daily_loss = initial_balance * max_daily_loss_pct
        self.r = redis_client
        self.starting_equity_of_day = initial_balance
        self.equity = initial_balance
        self.profit_target = initial_balance * 1.10
        
        # V15.0: Ratcheting State
        self.ratchet_floor = 0.0
        self.ratchet_step_pct = CONFIG.get('risk_management', {}).get('equity_ratchet', {}).get('step_percent', 0.05)

    def can_trade(self) -> bool:
        if self.equity <= 0: return False

        # 1. Total Drawdown Check (10% Max)
        total_dd_limit = self.initial_balance * 0.90
        
        # V15.0: Ratchet Logic Override
        # If we have locked in a floor, that becomes the new "hard deck"
        effective_floor = max(total_dd_limit, self.ratchet_floor)
        
        if self.equity < effective_floor: 
            return False
        
        # 2. Daily Drawdown Check (Default 5% Max)
        current_daily_loss = self.starting_equity_of_day - self.equity
        current_daily_limit = self.initial_balance * self.max_daily_loss_pct
        
        if current_daily_loss >= current_daily_limit: 
            return False
        
        # 3. Profit Target
        if self.equity >= self.profit_target:
            return False
            
        return True

    def update_equity(self, current_equity: float):
        self.equity = current_equity
        
        # V15.0: Check Ratchet
        # If equity grows 5% above initial, lock the floor at Init + 2.5% (Trailing)
        # Or simplistic: Every 5% gain locks the previous 5% level
        gain_pct = (self.equity - self.initial_balance) / self.initial_balance
        
        if gain_pct > 0:
             steps = int(gain_pct / self.ratchet_step_pct)
             if steps > 0:
                 new_floor = self.initial_balance * (1.0 + ((steps - 1) * self.ratchet_step_pct))
                 if new_floor > self.ratchet_floor:
                     self.ratchet_floor = new_floor
                     # Log implicitly via caller or separate mechanism if needed

    def check_circuit_breakers(self) -> str:
        if self.equity <= 0:
            return "Equity Uninitialized (0.0). Waiting for Producer Sync."

        total_limit = self.initial_balance * 0.90
        # Check Ratchet
        if self.ratchet_floor > total_limit:
             if self.equity < self.ratchet_floor:
                 return f"Ratchet Breach: Equity {self.equity:.2f} < Locked Floor {self.ratchet_floor:.2f}"
        
        if self.equity < total_limit:
            return f"Total Drawdown Breach: Equity {self.equity:.2f} < Limit {total_limit:.2f} (10%)"
        
        current_daily_loss = self.starting_equity_of_day - self.equity
        daily_limit = self.initial_balance * self.max_daily_loss_pct
        
        if current_daily_loss >= daily_limit:
            return f"Daily Drawdown Breach: Loss -${current_daily_loss:.2f} >= Limit ${daily_limit:.2f} (Anchor: {self.starting_equity_of_day:.2f})"
        
        if self.equity >= self.profit_target:
            return "Profit Target Reached (Victory Lap)"
        
        return "OK"

class HierarchicalRiskParity:
    @staticmethod
    def get_allocation(returns_df: pd.DataFrame) -> Dict[str, float]:
        cols = returns_df.columns.tolist()
        if not SCIPY_AVAILABLE:
            return {c: 1.0/len(cols) for c in cols}
        
        try:
            corr = returns_df.corr().fillna(0)
            dist = ssd.pdist(corr, metric='euclidean')
            link = sch.linkage(dist, method='single')
            
            sort_ix = HierarchicalRiskParity._get_quasi_diag(link)
            sort_ix = [cols[i] for i in sort_ix]
            
            cov = returns_df.cov()
            variances = np.diag(cov)
            variances[variances < EPS] = EPS
            inv_var = 1.0 / variances
            weights = inv_var / np.sum(inv_var)
            
            return dict(zip(cols, weights))
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
        # In a real impl, active_positions would be updated by the engine
        # Here we assume it is managed externally or via Redis sync
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