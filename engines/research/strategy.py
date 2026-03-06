from __future__ import annotations
import logging
import sys
import numpy as np
import math
import pytz
from collections import deque, defaultdict, Counter
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timedelta

# Third-Party Imports
try:
    from river import drift, linear_model, optim, forest, metrics
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Shared Imports
from shared import (
    CONFIG,
    OnlineFeatureEngineer,
    RiskManager,
    TradeContext,
    LogSymbols,
    Trade
)

# Local Imports
from engines.research.backtester import MarketSnapshot, BacktestBroker, BacktestOrder

logger = logging.getLogger("ResearchStrategy")

# =============================================================================
# V20.5 MATHEMATICAL PARITY CLASSES (THE ML UNCHOKING)
# Embedded locally to guarantee the WFO Backtester uses the exact same 
# Probability Calibration and TP/SL Geometry as the Live Predictor.
# =============================================================================

class ProbabilityCalibrator:
    """
    V20.5 BOOTSTRAP PROTOCOL:
    Platt Scaling Calibrator adapted for swing trading confidence.
    Guarantees smooth, continuous probability outputs (0.01 to 0.99).
    Now requires Statistical Maturity (50 wins & 50 losses) before choking trades.
    """
    def __init__(self, window: int = 1000):
        self.calibrator = None
        if ML_AVAILABLE:
            self.calibrator = linear_model.LogisticRegression(
                optimizer=optim.SGD(0.05),
                loss=optim.losses.Log(),
                l2=0.1
            )
        self.samples_seen = 0
        self.pos_count = 0
        self.neg_count = 0

    def update(self, prob: float, label: int):
        if ML_AVAILABLE and math.isfinite(prob):
            self.calibrator.learn_one({'raw_prob': prob}, label)
            self.samples_seen += 1
            if label == 1:
                self.pos_count += 1
            else:
                self.neg_count += 1

    def calibrate(self, raw_prob: float) -> float:
        if not ML_AVAILABLE: return raw_prob
        
        # V20.5 FIX: Guarantee WFO threshold bypass during data gathering
        # Prevents the "Probability Collapse" where the model blocks all trades
        if self.pos_count < 50 or self.neg_count < 50:
            return max(0.60, raw_prob) 
            
        try:
            calibrated_dict = self.calibrator.predict_proba_one({'raw_prob': raw_prob})
            calibrated = calibrated_dict.get(1, raw_prob)
            # V20.5 FIX: Never penalize base model by more than 5% to prevent collapse
            return max(raw_prob - 0.05, min(calibrated, 0.99))
        except Exception:
            return max(0.60, raw_prob)

class MetaLabeler:
    """
    Secondary model that learns when the primary model is likely to be right or wrong.
    """
    def __init__(self):
        self.model = None
        self.buffer = deque(maxlen=1000)
        self.pos_count = 0
        self.neg_count = 0
        
        if ML_AVAILABLE:
            self.model = forest.ARFClassifier(
                n_models=10, 
                seed=42, 
                metric=metrics.F1()
            )

    def update(self, features: Dict[str, float], primary_action: int, outcome_pnl: float):
        if not ML_AVAILABLE or primary_action == 0:
            return
        
        y_meta = 1 if outcome_pnl > 0 else 0
        
        if y_meta == 1:
            self.pos_count += 1
        else:
            self.neg_count += 1
            
        try:
            meta_feats = self._enrich(features, primary_action)
            clean_features = self._sanitize(meta_feats)
            self.model.learn_one(clean_features, y_meta)
            self.buffer.append((clean_features, y_meta))
        except Exception: pass

    def predict(self, features: Dict[str, float], primary_action: int, threshold: float = 0.50) -> bool:
        if not ML_AVAILABLE or primary_action == 0:
            return False
        
        # V20.5 FIX: Ensure Statistical Maturity.
        if self.pos_count < 50 or self.neg_count < 50: 
            return True 
        
        try:
            meta_feats = self._enrich(features, primary_action)
            clean_features = self._sanitize(meta_feats)
            probs = self.model.predict_proba_one(clean_features)
            return probs.get(1, 0.0) >= 0.40
        except Exception:
            return True

    def _enrich(self, features: Dict[str, float], action: int) -> Dict[str, float]:
        clean = features.copy()
        clean['primary_action'] = float(action)
        clean['vol_x_action'] = clean.get('volatility', 0.0) * action
        clean['hurst_x_action'] = (clean.get('hurst', 0.5) - 0.5) * action
        clean['vpin_x_action'] = clean.get('vpin', 0.5) * action
        clean['ker_x_action'] = clean.get('ker', 0.5) * action
        clean['aggressor_x_action'] = (clean.get('aggressor_ratio', 0.5) - 0.5) * action
        return clean

    def _sanitize(self, features: Dict[str, float]) -> Dict[str, float]:
        return {k: float(v) for k, v in features.items() if v is not None and math.isfinite(v)}

class AdaptiveTripleBarrier:
    """
    SPREAD TRAP CURE & CHOP LABELING.
    Simulates actual broker execution constraints perfectly aligned with RiskManager.
    """
    def __init__(self, horizon_ticks: int = 144, risk_mult: float = 1.5, reward_mult: float = 3.0, 
                 drift_threshold: float = 0.75, horizon_type: str = 'TIME', horizon_value: float = 0.0):
        self.buffer = deque()
        self.time_limit = horizon_ticks
        self.risk_mult = risk_mult
        self.reward_mult = reward_mult
        self.drift_threshold = drift_threshold
        self.horizon_type = horizon_type.upper()
        self.horizon_threshold = horizon_value
        if self.horizon_type == 'TIME' and self.horizon_threshold == 0:
            self.horizon_threshold = horizon_ticks

    def add_trade_opportunity(self, features: Dict[str, float], entry_price: float, current_atr: float, 
                              timestamp: float, parkinson_vol: float = 0.0, 
                              min_stop_dist: float = 0.0, spread_in_price: float = 0.0, 
                              comm_in_price: float = 0.0, min_profit_dist: float = 0.0,
                              proposed_action: int = 0, pred_proba: float = 0.5,
                              override_reward_mult: float = None):
        
        if current_atr <= 0: current_atr = entry_price * 0.0001
        
        volatility = features.get('volatility', 0.0)
        adaptive_scalar = 1.0 + (volatility * 100.0)
        
        p_vol = parkinson_vol if parkinson_vol > 0 else features.get('parkinson_vol', 0.0)
        vol_boost = 0.5 if p_vol > 0.002 else 0.0
        
        actual_reward_mult = override_reward_mult if override_reward_mult else self.reward_mult
        
        effective_risk_mult = (self.risk_mult + vol_boost) * adaptive_scalar
        effective_reward_mult = (actual_reward_mult + vol_boost) * adaptive_scalar
        
        raw_risk_dist = effective_risk_mult * current_atr
        actual_risk_dist = max(raw_risk_dist, min_stop_dist) 
        
        actual_reward_dist = actual_risk_dist * (effective_reward_mult / effective_risk_mult)
        
        buy_tp = entry_price + actual_reward_dist
        buy_sl = entry_price - actual_risk_dist
        
        sell_tp = entry_price - actual_reward_dist - spread_in_price
        sell_sl = entry_price + actual_risk_dist - spread_in_price

        self.buffer.append({
            'features': features.copy(),
            'entry': entry_price,
            'buy_tp': buy_tp,
            'buy_sl': buy_sl,
            'sell_tp': sell_tp,
            'sell_sl': sell_sl,
            'atr': current_atr,
            'start_time': timestamp, 
            'age': 0,
            'cum_vol': 0.0,
            'cum_volatility': 0.0,
            'spread_pct': spread_in_price / entry_price if entry_price > 0 else 0.0,
            'comm_pct': comm_in_price / entry_price if entry_price > 0 else 0.0,
            'proposed_action': proposed_action,
            'pred_proba': pred_proba
        })

    def resolve_labels(self, current_high: float, current_low: float, current_close: float = None, 
                       current_volume: float = 0.0, current_log_ret: float = 0.0) -> List[Tuple[Dict[str, float], int, float, int, float, float]]:
        resolved = []
        active = deque()
        if current_close is None: current_close = (current_high + current_low) / 2.0

        while self.buffer:
            trade = self.buffer.popleft()
            trade['age'] += 1
            trade['cum_vol'] += current_volume
            trade['cum_volatility'] += (current_log_ret ** 2)

            is_expired = False
            if self.horizon_type == 'TIME':
                if trade['age'] >= self.time_limit: is_expired = True
            elif self.horizon_type == 'VOLUME':
                thresh = self.horizon_threshold if self.horizon_threshold > 0 else current_volume * 100
                if trade['cum_vol'] >= thresh: is_expired = True
            elif self.horizon_type == 'VOLATILITY':
                thresh = self.horizon_threshold if self.horizon_threshold > 0 else 0.0005
                if trade['cum_volatility'] >= thresh: is_expired = True

            buy_status = 0
            sell_status = 0

            if current_low <= trade['buy_sl']: buy_status = -1
            elif current_high >= trade['buy_tp']: buy_status = 1
            
            if current_high >= trade['sell_sl']: sell_status = -1
            elif current_low <= trade['sell_tp']: sell_status = 1

            if buy_status != 0 or sell_status != 0 or is_expired:
                buy_ret = 0.0
                sell_ret = 0.0
                spread_pct = trade['spread_pct']
                comm_pct = trade['comm_pct']

                if buy_status == 1: 
                    buy_ret = (trade['buy_tp'] - trade['entry']) / trade['entry'] - spread_pct - comm_pct
                elif buy_status == -1: 
                    buy_ret = (trade['buy_sl'] - trade['entry']) / trade['entry'] - spread_pct - comm_pct
                elif is_expired: 
                    buy_ret = (current_close - trade['entry']) / trade['entry'] - spread_pct - comm_pct

                if sell_status == 1: 
                    sell_ret = (trade['entry'] - trade['sell_tp']) / trade['entry'] - spread_pct - comm_pct
                elif sell_status == -1: 
                    sell_ret = (trade['entry'] - trade['sell_sl']) / trade['entry'] - spread_pct - comm_pct
                elif is_expired: 
                    sell_ret = (trade['entry'] - current_close) / trade['entry'] - spread_pct - comm_pct

                # V20.5 FIX: REVERT BASE MODEL TO BINARY
                if buy_ret > sell_ret:
                    optimal_label = 1
                    optimal_ret = buy_ret
                elif sell_ret > buy_ret:
                    optimal_label = -1
                    optimal_ret = sell_ret
                else:
                    optimal_label = 1 if current_close >= trade['entry'] else -1
                    optimal_ret = buy_ret
                    
                proposed_action = trade.get('proposed_action', 0)
                pred_proba = trade.get('pred_proba', 0.5)
                proposed_ret = 0.0
                
                if proposed_action == 1:
                    proposed_ret = buy_ret
                elif proposed_action == -1:
                    proposed_ret = sell_ret

                resolved.append((
                    trade['features'], 
                    optimal_label, 
                    optimal_ret,
                    proposed_action,
                    pred_proba,
                    proposed_ret
                ))
            else:
                active.append(trade)

        self.buffer = active
        return resolved

# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class ResearchStrategy:
    """
    Represents an independent trading agent for a single symbol.
    Strictly implements the Profit Maximization Protocol.
    """
    def __init__(self, model: Any, symbol: str, params: dict[str, Any]):
        self.model = model
        self.symbol = symbol
        self.params = params
        self.debug_mode = False 
        
        # 1. Feature Engineer
        self.fe = OnlineFeatureEngineer(
            window_size=params.get('window_size', 50)
        )
        
        # --- RISK CONFIGURATION ---
        self.risk_conf = params.get('risk_management', CONFIG.get('risk_management', {}))
        self.sl_atr_mult = float(self.risk_conf.get('stop_loss_atr_mult', 1.5))
        self.max_currency_exposure = int(self.risk_conf.get('max_currency_exposure', 4))
        self.cooldown_minutes = int(self.risk_conf.get('loss_cooldown_minutes', 15)) 
        
        # 2. Adaptive Triple Barrier Labeler
        tbm_conf = params.get('tbm', {})
        self.optimized_reward_mult = float(tbm_conf.get('barrier_width', 3.0)) 
        
        self.labeler = AdaptiveTripleBarrier(
            horizon_ticks=int(tbm_conf.get('horizon_minutes', 720)), 
            risk_mult=self.sl_atr_mult, 
            reward_mult=self.optimized_reward_mult,
            drift_threshold=float(tbm_conf.get('drift_threshold', 1.5))
        )
        
        # 3. Meta Labeler & Calibrators
        self.meta_labeler = MetaLabeler()
        self.calibrator_buy = ProbabilityCalibrator()
        self.calibrator_sell = ProbabilityCalibrator()
        self.meta_label_events = 0 
        
        # 4. Warm-up State
        self.burn_in_limit = params.get('burn_in_periods', 30)
        self.burn_in_counter = 0
        self.burn_in_complete = False
        
        # State
        self.last_features = None
        self.last_price = 0.0
        self.last_price_map = {}
        self.bars_processed = 0
        self.consecutive_losses = 0 
        
        self.active_signals = deque()
        self.daily_performance = {'pnl': 0.0, 'losses': 0}
        
        self.last_trade_bar = 0
        self.last_trade_direction = 0
        
        # --- GOLDEN TRIO BUFFERS ---
        self.window_size_trio = 100
        self.closes_buffer = deque(maxlen=self.window_size_trio)
        self.volume_buffer = deque(maxlen=self.window_size_trio)
        
        self.h4_buffer = deque(maxlen=200) 
        self.d1_buffer = deque(maxlen=200) 
        self.last_h4_idx = -1
        self.last_d1_idx = -1
        self.current_d1_ema = 0.0 
        
        self.sma_window = deque(maxlen=200)
        self.returns_window = deque(maxlen=20)
        
        # --- FORENSIC RECORDER ---
        self.decision_log = deque(maxlen=1000)
        self.trade_events = []
        self.rejection_stats = defaultdict(int) 
        self.feature_importance_counter = Counter() 
        
        # ============================================================
        # STRATEGY CONFIGURATION (V20.5 UNCHOKED PROTOCOL)
        # ============================================================
        phx_conf = CONFIG.get('phoenix_strategy', {})
        
        self.bb_window = 20
        self.bb_std = float(phx_conf.get('bb_std_dev', 1.5)) 
        self.bb_buffer = deque(maxlen=self.bb_window)
        
        self.ker_floor = 0.0003 
        self.hurst_breakout = float(phx_conf.get('hurst_breakout_threshold', 0.55))
        self.hurst_veto = 0.65 
        
        self.rvol_trigger = float(phx_conf.get('rvol_volatility_trigger', 0.8)) 
        
        self.regime_enforcement = "DISABLED" 
        self.asset_regime_map = phx_conf.get('asset_regime_map', {})
        self.max_rvol_thresh = 35.0 
        self.adx_threshold = float(CONFIG.get('features', {}).get('adx', {}).get('threshold', 25.0))
        
        # --- SESSION CONTROL ---
        session_conf = self.risk_conf.get('session_control', {})
        self.session_enabled = session_conf.get('enabled', False)
        self.start_hour = session_conf.get('start_hour_server', 10)
        self.liq_hour = session_conf.get('liquidate_hour_server', 21)
        
        self.friday_entry_cutoff = self.risk_conf.get('friday_entry_cutoff_hour', 16)
        self.friday_close_hour = self.risk_conf.get('friday_liquidation_hour_server', 21)
        
        tz_str = self.risk_conf.get('risk_timezone', 'Europe/Prague')
        try:
            self.server_tz = pytz.timezone(tz_str)
        except Exception:
            self.server_tz = pytz.timezone('Europe/Prague')
        
        if ML_AVAILABLE:
            self.ker_drift_detector = drift.ADWIN(delta=0.01)
        self.dynamic_ker_offset = 0.0

        self.daily_max_losses = 20 
        self.daily_max_loss_pct = 0.045

    def _calculate_golden_trio(self) -> Tuple[float, float, float]:
        closes = self.closes_buffer
        vols = self.volume_buffer
        hurst = 0.5
        ker = 0.5
        rvol = 1.0
        
        if len(closes) < 30:
            return hurst, ker, rvol

        prices = np.array(closes)
        
        try:
            if np.var(prices) < 1e-9:
                hurst = 0.5
            else:
                lags = range(2, 20)
                tau = []
                for lag in lags:
                    diff = np.subtract(prices[lag:], prices[:-lag])
                    std = np.std(diff)
                    tau.append(std if std > 1e-9 else 1e-9)
                
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst = max(0.0, min(1.0, poly[0]))
        except:
            hurst = 0.5

        try:
            diffs = np.diff(prices)
            net_change = abs(prices[-1] - prices[0])
            sum_changes = np.sum(np.abs(diffs))
            ker = net_change / sum_changes if sum_changes > 1e-9 else 0.0
        except:
            ker = 0.0

        if len(vols) > 10:
            curr_vol = vols[-1]
            avg_vol = np.mean(list(vols)[:-1]) 
            rvol = curr_vol / avg_vol if avg_vol > 1e-9 else 1.0
        
        return hurst, ker, rvol

    def _check_volatility_condition(self, current_price: float) -> bool:
        if len(self.returns_window) < 10: return True 
        vol = np.std(list(self.returns_window))
        return vol >= 0.00005

    def _get_preferred_regime(self, symbol: str) -> str:
        if symbol in self.asset_regime_map: return self.asset_regime_map[symbol]
        regimes_found = [regime for key, regime in self.asset_regime_map.items() if key in symbol]
        if "TREND_BREAKOUT" in regimes_found: return "TREND_BREAKOUT"
        if "MEAN_REVERSION" in regimes_found: return "MEAN_REVERSION"
        return "NEUTRAL"

    def _check_currency_exposure(self, broker: BacktestBroker, symbol: str) -> bool:
        base_ccy = symbol[:3]
        quote_ccy = symbol[3:]
        base_count = 0
        quote_count = 0
        
        for pos in broker.open_positions:
            pos_base = pos.symbol[:3]
            pos_quote = pos.symbol[3:]
            if base_ccy == pos_base or base_ccy == pos_quote: base_count += 1
            if quote_ccy == pos_base or quote_ccy == pos_quote: quote_count += 1
        
        if base_count >= self.max_currency_exposure:
            self.rejection_stats[f"Max Exposure ({base_ccy})"] += 1
            return False
            
        if quote_count >= self.max_currency_exposure:
            self.rejection_stats[f"Max Exposure ({quote_ccy})"] += 1
            return False
            
        return True

    def _check_revenge_guard(self, broker: BacktestBroker, current_time: datetime) -> bool:
        my_closed_trades = [t for t in broker.closed_positions if t.symbol == self.symbol and t.close_time is not None]
        if not my_closed_trades: return False
            
        last_trade = my_closed_trades[-1]
        close_time = last_trade.close_time
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=current_time.tzinfo)
        else:
            close_time = close_time.astimezone(current_time.tzinfo)
            
        cooldown_expiry = close_time + timedelta(minutes=self.cooldown_minutes)
        return current_time < cooldown_expiry

    def on_data(self, snapshot: MarketSnapshot, broker: BacktestBroker):
        """
        Main Event Loop for the Strategy (Backtest Mode).
        """
        price = snapshot.get_price(self.symbol, 'close')
        high = snapshot.get_high(self.symbol)
        low = snapshot.get_low(self.symbol)
        volume = snapshot.get_price(self.symbol, 'volume')
        
        if price <= 0: return

        self.last_price_map = snapshot.to_price_dict()
        if self.symbol not in self.last_price_map:
            self.last_price_map[self.symbol] = price

        self.closes_buffer.append(price)
        self.volume_buffer.append(volume)
        self.sma_window.append(price)
        
        if len(self.sma_window) >= 2:
            prev_p = self.sma_window[-2]
            if prev_p > 0:
                self.returns_window.append(math.log(price / prev_p))
        
        self.bb_buffer.append(price)

        self._inject_auxiliary_data()

        dt_ts = datetime.now()
        try:
            if hasattr(snapshot.timestamp, 'timestamp'):
                timestamp = snapshot.timestamp.timestamp()
                dt_ts = snapshot.timestamp
            else:
                timestamp = float(snapshot.timestamp)
                dt_ts = datetime.fromtimestamp(timestamp)
        except Exception:
            timestamp = 0.0

        if dt_ts.tzinfo is None:
            dt_ts = dt_ts.replace(tzinfo=pytz.utc)
        server_time = dt_ts.astimezone(self.server_tz)

        context_data = self._simulate_mtf_context(price, server_time)
        self.current_d1_ema = context_data.get('d1', {}).get('ema200', 0.0)

        if self._check_symbol_circuit_breaker(broker, server_time):
            self.rejection_stats["Symbol Circuit Breaker (Loss Limit)"] += 1
            return
            
        if self._check_revenge_guard(broker, server_time):
            self.rejection_stats["Mandatory Cooldown (Revenge Guard)"] += 1
            return

        self._update_streak_status(broker)
        self._manage_trailing_stops(broker, price, dt_ts)
        self._manage_time_stops(broker, dt_ts)

        is_weekend_hold = (server_time.weekday() > 4) 
        is_friday_close = (server_time.weekday() == 4 and server_time.hour >= self.friday_close_hour)
        is_daily_session_close = False
        if self.session_enabled and server_time.hour >= self.liq_hour:
            is_daily_session_close = True

        if is_weekend_hold or is_friday_close or is_daily_session_close:
            for pos in list(broker.open_positions):
                if pos.symbol == self.symbol:
                     broker._close_partial_position(
                        pos, pos.quantity, price, dt_ts, "Session/Friday Liquidation"
                    )
            return 

        buy_vol = snapshot.get_price(self.symbol, 'buy_vol')
        sell_vol = snapshot.get_price(self.symbol, 'sell_vol')
        if buy_vol == 0 and sell_vol == 0:
            if self.last_price > 0:
                if price > self.last_price:
                    buy_vol = volume; sell_vol = 0.0
                elif price < self.last_price:
                    buy_vol = 0.0; sell_vol = volume
                else:
                    buy_vol = volume / 2.0; sell_vol = volume / 2.0
                    
        prev_price = self.sma_window[-2] if len(self.sma_window) >= 2 else price
        self.last_price = price

        # ============================================================
        # A. FEATURE ENGINEERING
        # ============================================================
        features = self.fe.update(
            price=price, timestamp=timestamp, volume=volume,
            high=high, low=low, buy_vol=buy_vol, sell_vol=sell_vol,
            time_feats={}, context_data=context_data
        )
        if features is None: return

        hurst, ker_val, rvol_val = self._calculate_golden_trio()
        features['hurst'] = hurst
        features['ker'] = ker_val
        features['rvol'] = rvol_val
        
        self.last_features = features

        if ML_AVAILABLE:
            self.ker_drift_detector.update(ker_val)
            if self.ker_drift_detector.drift_detected:
                self.dynamic_ker_offset = max(-0.10, self.dynamic_ker_offset - 0.05)
            else:
                self.dynamic_ker_offset = min(0.0, self.dynamic_ker_offset + 0.001)

        # ============================================================
        # B. DELAYED TRAINING
        # ============================================================
        log_ret = features.get('log_ret', 0.0)
        
        resolved_labels = self.labeler.resolve_labels(
            high, low, current_close=price, current_volume=volume, current_log_ret=log_ret
        )
        
        if resolved_labels:
            for (stored_feats, optimal_label, optimal_ret, past_action, past_prob, past_ret) in resolved_labels:
                
                w_pos = float(self.params.get('positive_class_weight', 1.0))
                w_neg = float(self.params.get('negative_class_weight', 1.0))
                base_weight = w_pos if optimal_label != 0 else w_neg
                
                ret_scalar = math.log1p(abs(optimal_ret) * 100.0)
                ret_scalar = max(0.5, min(ret_scalar, 5.0))
                
                hist_ker = stored_feats.get('ker', 0.5)
                ker_weight = hist_ker * 2.0 
                
                final_weight = base_weight * ret_scalar * ker_weight
                
                if optimal_label == 0:
                    final_weight *= 0.05 
                
                clean_stored = {k: float(v) for k, v in stored_feats.items() if math.isfinite(v)}
                
                if optimal_label != 0 and ML_AVAILABLE:
                    self.model.learn_one(clean_stored, optimal_label, sample_weight=final_weight)
                
                if past_action != 0:
                    self.meta_labeler.update(clean_stored, primary_action=past_action, outcome_pnl=past_ret)
                    if past_action == 1:
                        self.calibrator_buy.update(past_prob, 1 if past_ret > 0 else 0)
                    elif past_action == -1:
                        self.calibrator_sell.update(past_prob, 1 if past_ret > 0 else 0)

        # ============================================================
        # C. SMART SIGNAL MATRIX (V20.5 EXPERT HEURISTICS)
        # ============================================================
        regime_label = "Neutral"
        proposed_action = 0 
        
        is_trending = hurst >= self.hurst_breakout
        
        adx_val = features.get('adx', 0.0)
        rsi_norm = features.get('rsi_norm', 0.5)
        rsi_val = rsi_norm * 100.0

        if len(self.bb_buffer) >= self.bb_window:
            bb_mu = np.mean(self.bb_buffer)
            bb_std = np.std(self.bb_buffer)
            if bb_std < 1e-9: bb_std = 1e-9 
            
            upper_bb = bb_mu + (self.bb_std * bb_std)
            lower_bb = bb_mu - (self.bb_std * bb_std)
            
            if is_trending:
                regime_label = "TREND_BREAKOUT"
                if adx_val >= self.adx_threshold: # V20.5 Momentum
                    if price >= upper_bb: 
                        proposed_action = 1 
                    elif price <= lower_bb: 
                        proposed_action = -1 
            else:
                regime_label = "MEAN_REVERSION"
                # V20.5 Exhaustion
                if price <= lower_bb and rsi_val < 30.0: 
                    proposed_action = 1 
                elif price >= upper_bb and rsi_val > 70.0: 
                    proposed_action = -1  

        if hurst >= self.hurst_veto and regime_label == "MEAN_REVERSION" and proposed_action != 0:
            proposed_action = 0
            self.rejection_stats["Veto: Hyper Trend (Hurst)"] += 1

        clean_features = {k: float(v) for k, v in features.items() if math.isfinite(v)}
        try:
            if ML_AVAILABLE:
                pred_proba = self.model.predict_proba_one(clean_features)
                prob_success = pred_proba.get(proposed_action, 0.0)
            else:
                prob_success = 0.5
        except:
            prob_success = 0.0

        # ============================================================
        # D. ADD TRADE OPPORTUNITY
        # ============================================================
        current_atr = features.get('atr', 0.001)
        parkinson = features.get('parkinson_vol', 0.0)
        
        pip_val, _ = RiskManager.get_pip_info(self.symbol)
        if pip_val <= 0: pip_val = 0.0001
        
        spread_pips = float(CONFIG.get('forensic_audit', {}).get('spread_pips', {}).get(self.symbol, 1.6))
        spread_in_price = spread_pips * pip_val
        comm_in_price = 0.5 * pip_val 
        
        min_stop_pips = float(self.risk_conf.get('min_stop_loss_pips', 15.0))
        min_stop_dist = min_stop_pips * pip_val
        min_profit_dist = float(self.params.get('tbm', {}).get('min_profit_pips', 30.0)) * pip_val 

        # V20.5: Forced 1:2 Minimum Risk Reward
        current_reward_target = max(self.optimized_reward_mult, 2.0)
        if regime_label == "TREND_BREAKOUT":
            current_reward_target = max(current_reward_target, 3.0)

        self.labeler.add_trade_opportunity(
            features=features, entry_price=price, current_atr=current_atr, 
            timestamp=timestamp, parkinson_vol=parkinson,
            min_stop_dist=min_stop_dist, spread_in_price=spread_in_price,
            comm_in_price=comm_in_price, min_profit_dist=min_profit_dist,
            proposed_action=proposed_action, pred_proba=prob_success,
            override_reward_mult=current_reward_target
        )

        # ============================================================
        # E. SURVIVAL GATES
        # ============================================================
        if self.burn_in_counter < self.burn_in_limit:
            self.burn_in_counter += 1
            if self.burn_in_counter == self.burn_in_limit: self.burn_in_complete = True
            return

        self.bars_processed += 1
        existing_positions = [p for p in broker.open_positions if p.symbol == self.symbol]
        
        if self.session_enabled and server_time.hour < self.start_hour: return 
        if server_time.weekday() == 4 and server_time.hour >= self.friday_entry_cutoff: return 

        if proposed_action == 0:
            self.rejection_stats["No Trigger"] += 1
            return
            
        effective_ker_thresh = max(0.0001, self.ker_floor + self.dynamic_ker_offset)
        if ker_val < effective_ker_thresh:
            self.rejection_stats[f"Low Efficiency (KER)"] += 1
            if self.rejection_stats[f"Low Efficiency (KER)"] % 50 == 0:
                if self.debug_mode: logger.info(f"🛡️ {self.symbol} GATE: Low Efficiency (KER {ker_val:.4f} < {effective_ker_thresh:.4f})")
            return

        # ============================================================
        # F. ML CONFIRMATION & EXECUTION
        # ============================================================
        try:
            if proposed_action == 1: confidence = self.calibrator_buy.calibrate(prob_success)
            elif proposed_action == -1: confidence = self.calibrator_sell.calibrate(prob_success)
            else: confidence = 0.0
                
            min_conf = float(self.params.get('min_calibrated_probability', 0.60)) 
            if confidence < min_conf:
                self.rejection_stats[f"Low ML Confidence"] += 1
                if self.rejection_stats[f"Low ML Confidence"] % 10 == 0:
                    if self.debug_mode: logger.info(f"🤖 {self.symbol} GATE: Low ML Confidence ({confidence:.2f} < {min_conf:.2f})")
                return
            
            meta_thresh = float(self.params.get('meta_labeling_threshold', 0.50))
            is_profitable = self.meta_labeler.predict(clean_features, proposed_action, threshold=meta_thresh)
            if proposed_action != 0: self.meta_label_events += 1

            is_pyramid = False
            
            if existing_positions:
                pyramid_config = self.risk_conf.get('pyramiding', {})
                if not pyramid_config.get('enabled', False): return 
                
                last_pos = existing_positions[-1]
                if (last_pos.side == 1 and proposed_action != 1) or (last_pos.side == -1 and proposed_action != -1):
                    return 

                if len(existing_positions) > pyramid_config.get('max_adds', 1): return
                    
                risk_dist = last_pos.metadata.get('initial_risk_dist', 1.0)
                dist = (price - last_pos.entry_price) if last_pos.side == 1 else (last_pos.entry_price - price)
                current_r = dist / risk_dist if risk_dist > 0 else 0
                
                if current_r < pyramid_config.get('add_on_profit_r', 1.0): 
                    if self.debug_mode: logger.info(f"🧱 {self.symbol} GATE: Waiting for Open Trade Profit (> {pyramid_config.get('add_on_profit_r', 1.0)}R)")
                    return 
                is_pyramid = True

            if is_profitable:
                action_str = "BUY" if proposed_action == 1 else "SELL"
                
                imp_feats = [regime_label]
                if rvol_val > 2.0: imp_feats.append('High_Fuel')
                if hurst > 0.6: imp_feats.append('High_Hurst')
                for f in imp_feats: self.feature_importance_counter[f] += 1
                
                tighten_stops = (rvol_val > self.rvol_trigger)

                self._execute_entry(confidence, price, features, broker, dt_ts, proposed_action, regime_label, tighten_stops, is_pyramid, current_reward_target)
                
                self.last_trade_bar = self.bars_processed
                self.last_trade_direction = proposed_action
                
                if self.debug_mode: logger.info(f"🔥 {self.symbol} ML TRIGGER ACTIVATED! Confidence: {confidence:.2f}")
                
            else:
                self.rejection_stats['Meta-Labeler Reject'] += 1
                if self.rejection_stats['Meta-Labeler Reject'] % 10 == 0:
                    if self.debug_mode: logger.info(f"🧠 {self.symbol} GATE: Meta-Labeler Veto (Predicts Chop/Loss)")

        except Exception as e:
            if self.debug_mode: logger.error(f"Strategy Error: {e}")
            pass

    def _execute_entry(self, confidence, price, features, broker, dt_timestamp, action_int, regime, tighten_stops, is_pyramid, current_reward_target):
        action = "BUY" if action_int == 1 else "SELL"
        exposure_count = 0
        quote_currency = self.symbol[-3:] 
        for pos in broker.open_positions:
            if quote_currency in pos.symbol: exposure_count += 1
        
        ctx = TradeContext(
            symbol=self.symbol,
            price=price,
            stop_loss_price=0.0,
            account_equity=broker.equity,
            account_currency="USD",
            win_rate=0.45, 
            risk_reward_ratio=current_reward_target 
        )

        current_atr = features.get('atr', 0.001)
        current_ker = features.get('ker', 1.0)
        volatility = features.get('volatility', 0.001)
        parkinson = features.get('parkinson_vol', 0.0)
        
        risk_override = self.risk_conf.get('pyramiding', {}).get('risk_per_add_percent', 0.00) if is_pyramid else self.params.get('risk_per_trade_percent')
        sqn_score = self._calculate_symbol_sqn(broker)
        effective_sqn = -0.99 if -3.0 < sqn_score < -1.0 else sqn_score

        daily_pnl_val = broker.equity - broker.daily_start_equity
        daily_pnl_pct = daily_pnl_val / broker.daily_start_equity if broker.daily_start_equity > 0 else 0.0

        current_open_risk_usd = 0.0
        used_margin = 0.0
        contract_size = 100000 
        
        for pos in broker.open_positions:
            price_dist = abs(pos.entry_price - pos.stop_loss)
            rate = RiskManager.get_conversion_rate(pos.symbol, pos.entry_price, self.last_price_map)
            current_open_risk_usd += price_dist * pos.quantity * contract_size * rate
            used_margin += RiskManager.calculate_required_margin(pos.symbol, pos.quantity, pos.entry_price, contract_size, rate)
            
        current_open_risk_pct = (current_open_risk_usd / broker.equity) * 100.0
        estimated_free_margin = max(0.0, broker.equity - used_margin)

        trade_intent, risk_usd = RiskManager.calculate_rck_size(
            context=ctx,
            conf=confidence,
            volatility=volatility,
            active_correlations=exposure_count,
            market_prices=self.last_price_map,
            atr=current_atr, 
            ker=current_ker,
            account_size=broker.equity,
            risk_percent_override=risk_override,
            performance_score=effective_sqn,
            daily_pnl_pct=daily_pnl_pct,
            current_open_risk_pct=current_open_risk_pct,
            free_margin=estimated_free_margin,
            parkinson_vol=parkinson 
        )

        if trade_intent.volume <= 0:
            self.rejection_stats[f"Risk Zero"] += 1
            return

        trade_intent.action = action
        qty = trade_intent.volume
        
        stop_dist = trade_intent.stop_loss
        tp_dist = trade_intent.take_profit

        sl_price = price - stop_dist if action == "BUY" else price + stop_dist
        tp_price = price + tp_dist if action == "BUY" else price - tp_dist
        side = 1 if action == "BUY" else -1

        order = BacktestOrder(
            symbol=self.symbol,
            side=side,
            quantity=qty,
            entry_price=price, 
            timestamp_created=dt_timestamp, 
            stop_loss=sl_price,
            take_profit=tp_price,
            comment=f"{trade_intent.comment}|{regime}",
            metadata={
                'regime': regime,
                'confidence': float(confidence),
                'rvol': features.get('rvol', 0),
                'parkinson': parkinson,
                'ker': current_ker,
                'atr': current_atr,
                'optimized_rr': current_reward_target,
                'initial_risk_dist': stop_dist, 
                'entry_price_snap': price,
                'tighten_stops': tighten_stops,
                'is_pyramid': is_pyramid
            }
        )
        broker.submit_order(order)
        
        self.trade_events.append({
            'time': dt_timestamp.timestamp(), 
            'action': action,
            'price': price,
            'conf': confidence,
            'rvol': features.get('rvol', 0),
            'ker': current_ker,
            'atr': current_atr,
            'regime': regime,
            'pyramid': is_pyramid
        })

    def _calculate_symbol_sqn(self, broker: BacktestBroker) -> float:
        trades = [t.net_pnl for t in broker.closed_positions if t.symbol == self.symbol]
        if len(trades) < 5: return 0.0
        window = trades[-50:]
        avg_pnl = np.mean(window)
        std_pnl = np.std(window)
        if std_pnl < 1e-9: return 0.0
        return math.sqrt(len(window)) * (avg_pnl / std_pnl)

    def _manage_time_stops(self, broker: BacktestBroker, current_time: datetime):
        tbm_conf = self.params.get('tbm', {})
        horizon_minutes = int(tbm_conf.get('horizon_minutes', 720)) 
        hard_stop_seconds = horizon_minutes * 60
        
        to_close = []
        for pos in broker.open_positions:
            if pos.symbol != self.symbol: continue
            
            pos_time = pos.timestamp_created.replace(tzinfo=pytz.utc) if pos.timestamp_created.tzinfo is None else pos.timestamp_created
            curr_time_aware = current_time.replace(tzinfo=pytz.utc) if current_time.tzinfo is None else current_time
            
            if (curr_time_aware - pos_time).total_seconds() > hard_stop_seconds:
                to_close.append((pos, f"Time Stop ({int(horizon_minutes/60)}h)"))
            
        for pos, reason in to_close:
            broker._close_partial_position(pos, pos.quantity, self.last_price, current_time, reason)

    def _manage_trailing_stops(self, broker: BacktestBroker, current_price: float, timestamp: datetime):
        trail_conf = self.risk_conf.get('trailing_stop', {})
        activation_r = float(trail_conf.get('activation_r', 1.0))
        trail_dist_r = float(trail_conf.get('trail_dist_r', 0.5))

        for pos in broker.open_positions:
            if pos.symbol != self.symbol: continue
            risk_dist = pos.metadata.get('initial_risk_dist', 0.0)
            if risk_dist <= 0: continue
            
            new_sl = None
            reason = ""
            
            if pos.side == 1: 
                r_multiple = (current_price - pos.entry_price) / risk_dist
                if r_multiple >= activation_r:
                    target_sl = max(current_price - (risk_dist * trail_dist_r), pos.entry_price + (risk_dist * 0.1))
                    if target_sl > pos.stop_loss:
                        new_sl, reason = target_sl, f"Trail ({r_multiple:.1f}R)"
                        
            else: 
                r_multiple = (pos.entry_price - current_price) / risk_dist
                if r_multiple >= activation_r:
                    target_sl = min(current_price + (risk_dist * trail_dist_r), pos.entry_price - (risk_dist * 0.1))
                    if target_sl < pos.stop_loss:
                        new_sl, reason = target_sl, f"Trail ({r_multiple:.1f}R)"
            
            if new_sl is not None:
                pos.stop_loss = new_sl
                if "Trail" not in pos.comment: pos.comment += f"|{reason}"

    def _check_symbol_circuit_breaker(self, broker: BacktestBroker, server_time: datetime) -> bool:
        today_losses = 0
        today_pnl = 0.0
        current_day_start = server_time.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        
        for trade in broker.closed_positions:
            if trade.symbol != self.symbol: continue
            
            if trade.close_time:
                close_time_aware = trade.close_time if trade.close_time.tzinfo else trade.close_time.replace(tzinfo=pytz.utc)
                exit_ts = close_time_aware.timestamp()
            else:
                exit_ts = 0.0
            
            if exit_ts >= current_day_start:
                today_pnl += trade.net_pnl
                if trade.net_pnl < 0: today_losses += 1
        
        if today_losses >= self.daily_max_losses: return True
        if today_pnl < -(broker.initial_balance * self.daily_max_loss_pct): return True
        return False

    def _update_streak_status(self, broker: BacktestBroker):
        if not broker.closed_positions:
            self.consecutive_losses = 0
            return
        streak = 0
        for trade in reversed(broker.closed_positions):
            if trade.net_pnl < 0: streak += 1
            else: break
        self.consecutive_losses = streak

    def _simulate_mtf_context(self, price: float, dt: datetime) -> Dict[str, Any]:
        h4_idx = (dt.day * 6) + (dt.hour // 4)
        if h4_idx != self.last_h4_idx:
            self.h4_buffer.append(price)
            self.last_h4_idx = h4_idx
            
        d1_idx = dt.toordinal()
        if d1_idx != self.last_d1_idx:
            self.d1_buffer.append(price)
            self.last_d1_idx = d1_idx
            
        ctx = {'d1': {}, 'h4': {}}
        if len(self.d1_buffer) > 0:
            arr = np.array(self.d1_buffer)
            if len(arr) < 200: ema = np.mean(arr)
            else:
                alpha = 2 / 201
                ema = arr[0]
                for x in arr[1:]: ema = (alpha * x) + ((1 - alpha) * ema)
            ctx['d1']['ema200'] = ema
        else: ctx['d1']['ema200'] = 0.0
            
        if len(self.h4_buffer) > 14:
            arr = np.array(self.h4_buffer)
            changes = np.diff(arr)
            gains, losses = changes[changes > 0], -changes[changes < 0]
            avg_gain = np.mean(gains[-14:]) if len(gains) > 0 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) > 0 else 0
            ctx['h4']['rsi'] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + (avg_gain / avg_loss)))
        else: ctx['h4']['rsi'] = 50.0
        return ctx

    def _inject_auxiliary_data(self):
        defaults = {
            "USDJPY": 150.0, "GBPUSD": 1.25, "EURUSD": 1.08,
            "USDCAD": 1.35, "USDCHF": 0.90, "AUDUSD": 0.65, "NZDUSD": 0.60,
            "GBPJPY": 190.0, "EURJPY": 160.0, "AUDJPY": 95.0,
            "GBPAUD": 1.95, "EURAUD": 1.65, "GBPNZD": 2.05
        }
        for sym, price in defaults.items():
            if sym not in self.last_price_map: self.last_price_map[sym] = price

    def generate_autopsy(self) -> str:
        if not self.trade_events:
            reject_str = ", ".join([f"{k}: {v}" for k, v in sorted(self.rejection_stats.items(), key=lambda item: item[1], reverse=True)[:5]])
            status = "Waiting for Warm-Up" if not self.burn_in_complete else "No Trigger Conditions Met"
            return f"AUTOPSY: No trades. Status: {status}. Top Rejections: {{{reject_str}}}. Bars processed: {self.bars_processed}"
        
        avg_conf = np.mean([t['conf'] for t in self.trade_events])
        avg_rvol = np.mean([t['rvol'] for t in self.trade_events]) 
        reject_str = ", ".join([f"{k}: {v}" for k, v in sorted(self.rejection_stats.items(), key=lambda item: item[1], reverse=True)[:5]])
        
        report = (
            f"\n --- 💀 PHOENIX AUTOPSY ({self.symbol}) ---\n"
            f" Trades: {len(self.trade_events)}\n"
            f" Avg Conf: {avg_conf:.2f}\n"
            f" Avg RVol: {avg_rvol:.2f}\n"
            f" Top Drivers: {str(self.feature_importance_counter.most_common(5))}\n"
            f" Rejections: {{{reject_str}}}\n"
            f" ----------------------------------------\n"
        )
        return report