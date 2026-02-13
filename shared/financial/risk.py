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

# Scipy Imports (Guarded)
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
    DEFAULT_CONTRACT_SIZE = 100_000

    @staticmethod
    def get_contract_size(symbol: str) -> float:
        """
        Determines the standard contract size based on asset class.
        Crucial for correct Margin and Risk calculations.
        """
        s = symbol.upper()
        
        # 1. Metals
        if "XAU" in s or "XAG" in s: return 100.0
        # 2. Crypto
        if "BTC" in s or "ETH" in s or "LTC" in s or "XRP" in s: return 1.0
        # 3. Indices
        if any(idx in s for idx in ["US30", "GER30", "GER40", "NAS100", "SPX500", "US500", "DJI", "DAX", "UK100", "JP225"]):
            return 1.0
        # 4. Oil
        if "WTI" in s or "BRENT" in s or "OIL" in s: return 100.0
        # 5. Forex (Standard)
        return 100_000.0

    @staticmethod
    def get_pip_info(symbol: str) -> Tuple[float, int]:
        """Returns (pip_size, digits)."""
        s = symbol.upper()
        if "JPY" in s: return 0.01, 3
        if "XAU" in s or "XAG" in s: return 0.1, 2
        if any(x in s for x in ["US30", "GER30", "NAS100", "SPX500", "DJI", "DAX"]): return 1.0, 1
        if "BTC" in s or "ETH" in s: return 1.0, 2
        return 0.0001, 5

    @staticmethod
    def get_conversion_rate(symbol: str, price: float, market_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculates the Quote -> USD conversion rate using LIVE market data.
        """
        s = symbol.upper()
        if s.endswith("USD"): return 1.0
        
        def get_price(sym: str) -> Optional[float]:
            if market_prices and sym in market_prices: return market_prices[sym]
            return None

        # JPY Pairs (Quote = JPY) -> Need 1 / USDJPY
        if "JPY" in s:
            usdjpy = get_price("USDJPY")
            if usdjpy and usdjpy > 0: return 1.0 / usdjpy
            if s == "USDJPY" and price > 0: return 1.0 / price
            return 0.0
        
        # GBP Pairs (Quote = GBP) -> Need GBPUSD
        if s.endswith("GBP"):
            gbpusd = get_price("GBPUSD")
            if gbpusd: return gbpusd
            return 0.0

        # CAD Pairs (Quote = CAD) -> Need 1 / USDCAD
        if s.endswith("CAD"):
            usdcad = get_price("USDCAD")
            if usdcad and usdcad > 0: return 1.0 / usdcad
            if s == "USDCAD" and price > 0: return 1.0 / price
            return 0.0

        # CHF Pairs (Quote = CHF) -> Need 1 / USDCHF
        if s.endswith("CHF"):
            usdchf = get_price("USDCHF")
            if usdchf and usdchf > 0: return 1.0 / usdchf
            if s == "USDCHF" and price > 0: return 1.0 / price
            return 0.0

        # AUD Pairs (Quote = AUD) -> Need AUDUSD
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
            
        return 0.0

    @staticmethod
    def calculate_required_margin(symbol: str, lots: float, price: float, contract_size: float, conversion_rate: float) -> float:
        """
        Calculates required margin based on configured leverage tiers.
        """
        risk_conf = CONFIG.get('risk_management', {})
        lev_map = risk_conf.get('leverage', {})
        
        # Determine Leverage
        leverage = float(lev_map.get('default', 30.0))
        s = symbol.upper()
        
        if "JPY" in s and not "USD" in s and not "GBP" in s and not "EUR" in s:
            leverage = float(lev_map.get('minor', 30.0))
        if "XAU" in s or "XAG" in s:
            leverage = float(lev_map.get('gold', 20.0))
        if any(x in s for x in ["US30", "GER30", "NAS100", "SPX500"]):
            leverage = float(lev_map.get('indices', 20.0))
        if "BTC" in s or "ETH" in s:
            leverage = float(lev_map.get('crypto', 2.0))
            
        if leverage <= 0: leverage = 1.0
        
        notional_value = lots * contract_size * price * conversion_rate
        margin = notional_value / leverage
        return margin

    @staticmethod
    def _calculate_max_margin_volume(symbol: str, free_margin: float, contract_size: float, price: float, conversion_rate: float) -> float:
        """
        Calculates maximum volume allowed by free margin (Leverage Guard).
        """
        if free_margin <= 0: return 0.0

        risk_conf = CONFIG.get('risk_management', {})
        lev_map = risk_conf.get('leverage', {})
        
        lev = float(lev_map.get('default', 30.0))
        s = symbol.upper()
        
        if "JPY" in s and not "USD" in s and not "GBP" in s and not "EUR" in s:
            lev = float(lev_map.get('minor', 30.0))
        if "XAU" in s or "XAG" in s:
            lev = float(lev_map.get('gold', 20.0))
        elif "BTC" in s or "ETH" in s:
            lev = float(lev_map.get('crypto', 2.0))
        elif any(x in s for x in ["US30", "GER30", "NAS100", "SPX500"]):
            lev = float(lev_map.get('indices', 20.0))

        if lev <= 0: return 0.0

        one_lot_margin = (contract_size * price * conversion_rate) / lev
        if one_lot_margin <= 0: return 0.0

        # 95% Safety Factor
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
        performance_score: float = 0.0,
        daily_pnl_pct: float = 0.0,
        current_open_risk_pct: float = 0.0,
        free_margin: float = 999999.0 
    ) -> Tuple[Trade, float]:
        """
        V17.0 UPDATE: Supports 'fixed_lots' to force micro-sizing (Survival Protocol).
        Prevents 4.0 lot blowups by bypassing equity-based calculations.
        """
        symbol = context.symbol
        balance = context.account_equity
        price = context.price
        
        if balance <= 0:
             return Trade(symbol, "HOLD", 0.0, 0.0, 0.0, 0.0, "Zero Balance"), 0.0

        # 1. Config & Contract Size
        risk_conf = CONFIG.get('risk_management', {})
        if contract_size_override:
            c_size = contract_size_override
        else:
            c_size = RiskManager.get_contract_size(symbol)
        
        # 2. Stop Loss Geometry (Geometry First, Sizing Second)
        atr_mult_sl = float(risk_conf.get('stop_loss_atr_mult', 1.5)) 
        atr_mult_tp = float(risk_conf.get('take_profit_atr_mult', 3.0))
        
        if atr and atr > 0:
            stop_dist = atr * atr_mult_sl
        else:
            stop_dist = price * 0.002 * atr_mult_sl
            
        # Hard Floor Enforcement
        pip_val_raw, _ = RiskManager.get_pip_info(symbol)
        if pip_val_raw <= 0: pip_val_raw = 0.0001
        
        # V17.0: Absolute Hard Floor (25 pips default)
        config_min_pips = float(risk_conf.get('min_stop_loss_pips', 25.0))
        min_stop_hard_floor = config_min_pips * pip_val_raw
        
        # Apply Floor
        stop_dist = max(stop_dist, min_stop_hard_floor)
        sl_pips = stop_dist / pip_val_raw

        # 3. Conversion Rate
        conversion_rate = RiskManager.get_conversion_rate(symbol, price, market_prices)
        if conversion_rate <= 0:
            return Trade(symbol, "HOLD", 0.0, 0.0, 0.0, 0.0, "No Conversion Rate"), 0.0

        # Calculate Risk Per Lot (Required for both Fixed and Dynamic logic)
        usd_per_pip_per_lot = c_size * pip_val_raw * conversion_rate
        loss_per_lot_usd = sl_pips * usd_per_pip_per_lot
        
        if loss_per_lot_usd <= 1e-9:
            return Trade(symbol, "HOLD", 0.0, 0.0, 0.0, 0.0, "Zero Risk Error"), 0.0

        # =========================================================
        # V17.0 FIX: FIXED LOT BYPASS
        # If method is 'fixed_lots', we ignore all equity math.
        # =========================================================
        sizing_method = risk_conf.get('sizing_method', 'risk_percentage')
        
        if sizing_method == 'fixed_lots':
            fixed_qty = float(risk_conf.get('fixed_lot_size', 0.01))
            
            # Verify Margin is sufficient for this fixed size
            max_margin_lots = RiskManager._calculate_max_margin_volume(symbol, free_margin, c_size, price, conversion_rate)
            
            if fixed_qty > max_margin_lots:
                # If we can't afford 0.01, we hold.
                return Trade(symbol, "HOLD", 0.0, 0.0, 0.0, 0.0, f"Insufficient Margin for {fixed_qty}"), 0.0
            
            # Calculate implied risk for stats
            implied_risk_usd = fixed_qty * loss_per_lot_usd
            
            trade = Trade(
                symbol=symbol,
                action="HOLD", # Action set by caller
                volume=fixed_qty,
                entry_price=price,
                stop_loss=stop_dist,
                take_profit=stop_dist * (atr_mult_tp / atr_mult_sl),
                comment=f"Fixed:{fixed_qty}|SL:{sl_pips:.0f}|R:${implied_risk_usd:.2f}"
            )
            return trade, implied_risk_usd

        # =========================================================
        # LEGACY PERCENTAGE LOGIC (Fallback)
        # =========================================================
        
        lots = 0.0
        calculated_risk_usd = 0.0
        
        default_base_risk = risk_conf.get('base_risk_per_trade_percent', 0.0025)
        buffer_threshold = risk_conf.get('profit_buffer_threshold', 0.02)
        scaled_risk_val = risk_conf.get('scaled_risk_percent', 0.005)
        
        scaling_comment = ""
        
        if risk_percent_override is not None:
            risk_pct = risk_percent_override * 100.0 if risk_percent_override < 1.0 else risk_percent_override
        else:
            if daily_pnl_pct >= buffer_threshold:
                risk_pct = scaled_risk_val * 100.0
                scaling_comment = "|Buf:ON"
            else:
                risk_pct = default_base_risk * 100.0
        
        # SQN Scaling
        if performance_score != 0.0:
            if performance_score < -2.0:
                risk_pct = 0.1 # Toxic
                scaling_comment += "|Toxic"
            elif performance_score < 0.0:
                risk_pct = 0.25 # Cold
                scaling_comment += "|Cold"
            elif performance_score > 2.5:
                risk_pct = min(risk_pct * 1.1, 0.5) # Hot (Capped)
                scaling_comment += "|Hot"

        # Calculate Lots
        calculated_risk_usd = balance * (risk_pct / 100.0)
        
        # KER Efficiency Scaling
        ker_scalar = max(0.8, min(ker, 1.0))
        calculated_risk_usd *= ker_scalar
        
        lots = calculated_risk_usd / loss_per_lot_usd

        # Correlation Penalty
        if active_correlations > 0:
            penalty_factor = 1.0 / (1.0 + (0.5 * active_correlations))
            lots *= penalty_factor

        # Margin Clamp
        max_margin_lots = RiskManager._calculate_max_margin_volume(symbol, free_margin, c_size, price, conversion_rate)
        if lots > max_margin_lots:
            lots = max_margin_lots
            scaling_comment += "|LevGuard"

        # Final Limits
        min_lot = risk_conf.get('min_lot_size', 0.01)
        max_lot = risk_conf.get('max_lot_size', 50.0)
        lots = max(min_lot, min(lots, max_lot))
        lots = round(lots, 2)
        
        final_risk_usd = lots * loss_per_lot_usd
        
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
        self.friday_liquidation_hour = risk_conf.get('friday_liquidation_hour_server', 21)

        session_conf = risk_conf.get('session_control', {})
        self.session_enabled = session_conf.get('enabled', False)
        self.start_hour = session_conf.get('start_hour_server', 10)
        self.liq_hour = session_conf.get('liquidate_hour_server', 21)

    def is_trading_allowed(self) -> bool:
        now_local = datetime.now(self.market_tz)
        weekday = now_local.weekday()
        current_time = now_local.time()
        
        if weekday == 5: return False 
        if weekday == 6 and current_time < self.monday_start: return False 
        if weekday == 4 and current_time > self.friday_cutoff: return False 
        
        if current_time >= self.rollover_start or current_time <= self.rollover_end:
            return False
            
        if self.session_enabled:
            if now_local.hour < self.start_hour: return False
            if now_local.hour >= self.liq_hour: return False

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
        
        if weekday > 4: return True 
        if weekday == 4 and now_local.hour >= self.friday_liquidation_hour: return True
        if self.session_enabled:
            if now_local.hour >= self.liq_hour: return True
            
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
        self.ratchet_floor = 0.0
        self.ratchet_step_pct = CONFIG.get('risk_management', {}).get('equity_ratchet', {}).get('step_percent', 0.05)

    def can_trade(self) -> bool:
        if self.equity <= 0: return False
        
        total_dd_limit = self.initial_balance * 0.90
        effective_floor = max(total_dd_limit, self.ratchet_floor)
        
        if self.equity < effective_floor: return False
        
        current_daily_loss = self.starting_equity_of_day - self.equity
        current_daily_limit = self.initial_balance * self.max_daily_loss_pct
        
        if current_daily_loss >= current_daily_limit: return False
        if self.equity >= self.profit_target: return False
            
        return True

    def update_equity(self, current_equity: float):
        self.equity = current_equity
        gain_pct = (self.equity - self.initial_balance) / self.initial_balance
        if gain_pct > 0:
             steps = int(gain_pct / self.ratchet_step_pct)
             if steps > 0:
                 new_floor = self.initial_balance * (1.0 + ((steps - 1) * self.ratchet_step_pct))
                 if new_floor > self.ratchet_floor:
                     self.ratchet_floor = new_floor

    def check_circuit_breakers(self) -> str:
        if self.equity <= 0: return "Equity Uninitialized"
        total_limit = self.initial_balance * 0.90
        
        if self.ratchet_floor > total_limit:
             if self.equity < self.ratchet_floor: return f"Ratchet Breach"
        
        if self.equity < total_limit: return f"Total Drawdown Breach"
        
        current_daily_loss = self.starting_equity_of_day - self.equity
        daily_limit = self.initial_balance * self.max_daily_loss_pct
        
        if current_daily_loss >= daily_limit: return f"Daily Drawdown Breach"
        if self.equity >= self.profit_target: return "Profit Target Reached"
        
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
        if self.correlation_matrix.empty or symbol not in self.correlation_matrix.columns: return 0
        count = 0
        for held_symbol in self.active_positions:
            if held_symbol == symbol: continue
            if held_symbol in self.correlation_matrix.columns:
                corr = self.correlation_matrix.loc[symbol, held_symbol]
                if abs(corr) > threshold: count += 1
        return count

    def check_penalty_box(self, symbol: str) -> bool:
        if symbol in self.penalty_box:
            if time.time() < self.penalty_box[symbol]: return True
            else: del self.penalty_box[symbol]
        return False

    def add_to_penalty_box(self, symbol: str, duration_minutes: int = 60):
        self.penalty_box[symbol] = time.time() + (duration_minutes * 60)
        logger.warning(f"🚫 {symbol} added to Penalty Box for {duration_minutes}m")