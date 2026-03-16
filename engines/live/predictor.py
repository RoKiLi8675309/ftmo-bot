import logging
import pickle
import os
import json
import time
import math
import uuid
import pytz 
from datetime import datetime, date
from collections import defaultdict, deque, Counter
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np 
import pandas as pd

# Third-Party ML Imports (Use Conda for installation)
try:
    from river import forest, compose, preprocessing, metrics, drift, linear_model, optim
    ML_AVAILABLE = True
except ImportError:
    print("CRITICAL: 'river' library not found. Install with: conda install -c conda-forge river")
    ML_AVAILABLE = False
    import sys
    sys.exit(1)

# Shared Imports
from shared import (
    CONFIG,
    LogSymbols,
    OnlineFeatureEngineer,
    VolumeBar,
    RiskManager,
    load_real_data,            
    AdaptiveImbalanceBarGenerator, 
    get_redis_connection       
)

logger = logging.getLogger("Predictor")

class Signal:
    def __init__(self, symbol: str, action: str, confidence: float, meta_data: Dict[str, Any], signal_id: str = None):
        self.symbol = symbol
        self.action = action  
        self.confidence = confidence
        self.meta_data = meta_data
        self.id = signal_id or str(uuid.uuid4())

# =============================================================================
# V20.18 MATHEMATICAL PARITY CLASSES (DUAL-MODEL ASYMMETRY PROTOCOL)
# =============================================================================

class ProbabilityCalibrator:
    """
    Platt Scaling Calibrator adapted for swing trading confidence.
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
        
        # Keep the bot exploring but using raw probabilities until mature.
        if self.pos_count < 10 or self.neg_count < 10:
            return raw_prob 
            
        try:
            calibrated_dict = self.calibrator.predict_proba_one({'raw_prob': raw_prob})
            calibrated = calibrated_dict.get(1, raw_prob)
            return max(0.01, min(calibrated, 0.99))
        except Exception:
            return raw_prob

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
        
        # This is where NOISE is filtered. 1 = Profit, 0 = Loss.
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
            
        if self.pos_count < 10 or self.neg_count < 10: 
            return True 
            
        try:
            meta_feats = self._enrich(features, primary_action)
            clean_features = self._sanitize(meta_feats)
            probs = self.model.predict_proba_one(clean_features)
            prob_profit = probs.get(1, 0.0)
            return prob_profit >= threshold
        except Exception as e:
            logger.debug(f"MetaLabeler Predict Error: {e}")
            return True

    def _enrich(self, features: Dict[str, float], action: int) -> Dict[str, float]:
        clean = features.copy()
        clean['primary_action'] = float(action)
        clean['vol_x_action'] = clean.get('volatility', 0.0) * action
        clean['hurst_x_action'] = (clean.get('hurst', 0.5) - 0.5) * action
        clean['vpin_x_action'] = clean.get('vpin', 0.5) * action
        clean['ker_x_action'] = clean.get('ker', 0.5) * action
        return clean

    def _sanitize(self, features: Dict[str, float]) -> Dict[str, float]:
        clean = {}
        for k, v in features.items():
            if v is not None and math.isfinite(v):
                clean[k] = float(v)
            else:
                clean[k] = 0.0
        return clean

class AdaptiveTripleBarrier:
    """
    SPREAD TRAP CURE & DUAL ASYMMETRY LABELING.
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
                              override_reward_mult: float = None, signal_id: str = None):
        
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
        
        buy_tp = entry_price + actual_reward_dist + spread_in_price
        buy_sl = entry_price - actual_risk_dist + spread_in_price
        
        sell_tp = entry_price - actual_reward_dist - spread_in_price
        sell_sl = entry_price + actual_risk_dist - spread_in_price

        min_profit_pct = (min_profit_dist + spread_in_price + comm_in_price) / entry_price if entry_price > 0 else 0.0
        
        feats_copy = features.copy()
        feats_copy['min_profit_pct'] = min_profit_pct

        self.buffer.append({
            'signal_id': signal_id, 
            'is_executed': False, 
            'features': feats_copy,
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
                       current_volume: float = 0.0, current_log_ret: float = 0.0,
                       current_timestamp: float = 0.0) -> List[Tuple[Dict[str, float], int, int, float, float, int, float, float, bool]]:
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
                if current_timestamp > 0 and trade.get('start_time', 0) > 0:
                    duration_sec = current_timestamp - trade['start_time']
                    if duration_sec >= (self.time_limit * 60): is_expired = True
                else:
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

                buy_label = 1 if buy_ret > 0 else 0
                sell_label = 1 if sell_ret > 0 else 0
                    
                proposed_action = trade.get('proposed_action', 0)
                pred_proba = trade.get('pred_proba', 0.5)
                proposed_ret = 0.0
                
                if proposed_action == 1:
                    proposed_ret = buy_ret
                elif proposed_action == -1:
                    proposed_ret = sell_ret

                is_executed = trade.get('is_executed', False) 

                resolved.append((
                    trade['features'], 
                    buy_label,
                    sell_label,
                    buy_ret,
                    sell_ret,
                    proposed_action,
                    pred_proba,
                    proposed_ret,
                    is_executed
                ))
            else:
                active.append(trade)

        self.buffer = active
        return resolved

# =============================================================================
# STRATEGY ENGINE (LIVE PREDICTOR)
# =============================================================================

class MultiAssetPredictor:
    """
    Live prediction engine applying V20.18 Dual-Model EV Math.
    """
    def __init__(self, symbols: List[str], threshold_map: Optional[Dict[str, float]] = None):
        self.symbols = symbols
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.threshold_map = threshold_map if threshold_map else {}
        self.feature_engineers = {s: OnlineFeatureEngineer(window_size=CONFIG['features']['window_size']) for s in symbols}
        
        tbm_conf = CONFIG['online_learning']['tbm']
        risk_conf = CONFIG.get('risk_management', {})
        risk_mult_conf = float(risk_conf.get('stop_loss_atr_mult', 1.5)) 
        
        self.max_currency_exposure = int(risk_conf.get('max_currency_exposure', 4))
        
        session_conf = risk_conf.get('session_control', {})
        self.session_enabled = session_conf.get('enabled', False)
        self.start_hour = session_conf.get('start_hour_server', 0)
        self.liq_hour = session_conf.get('liquidate_hour_server', 23)
        self.friday_entry_cutoff = risk_conf.get('friday_entry_cutoff_hour', 16)
        
        self.labelers = {}
        self.optimized_params = {} 
        
        self.window_size_trio = 100
        self.closes_buffer = {s: deque(maxlen=self.window_size_trio) for s in symbols}
        self.volume_buffer = {s: deque(maxlen=self.window_size_trio) for s in symbols}
        
        self.bb_window = 20
        self.bb_std = 1.5 
        self.bb_buffers = {s: deque(maxlen=self.bb_window) for s in symbols}
        
        self.sniper_closes = {s: deque(maxlen=200) for s in symbols} 
        self.sniper_rsi = {s: deque(maxlen=15) for s in symbols}    

        self.sma_window = {s: deque(maxlen=200) for s in symbols}
        self.returns_window = {s: deque(maxlen=20) for s in symbols}

        self.last_trade_bar = {s: 0 for s in symbols}
        self.last_trade_direction = {s: 0 for s in symbols}
        
        self.last_features = {s: None for s in symbols}
        self.prev_features = {s: None for s in symbols}

        phx_conf = CONFIG.get('phoenix_strategy', {})
        self.regime_enforcement = "DISABLED" 
        self.asset_regime_map = phx_conf.get('asset_regime_map', {})
        
        self.ker_floor = 0.0003 
        self.hurst_breakout = float(phx_conf.get('hurst_breakout_threshold', 0.52)) 
        self.hurst_veto = 0.65 
        self.rvol_trigger = 0.8 
        self.max_rvol_thresh = 35.0 
        self.adx_threshold = 0.0 

        for s in symbols:
            s_risk = risk_mult_conf
            s_reward = tbm_conf.get('barrier_width', 3.0) 
            s_horizon = tbm_conf.get('horizon_minutes', 720) 
            s_horizon_type = tbm_conf.get('horizon_type', 'TIME')
            s_horizon_val = float(tbm_conf.get('horizon_threshold', 0.0))
            
            params_path = self.models_dir / f"best_params_{s}.json"
            if params_path.exists():
                try:
                    with open(params_path, 'r') as f:
                        bp = json.load(f)
                        self.optimized_params[s] = bp 
                        
                        if 'barrier_width' in bp: s_reward = max(float(bp['barrier_width']), 2.0)
                        if 'horizon_minutes' in bp: s_horizon = int(bp['horizon_minutes'])
                        
                        if 'risk_per_trade_percent' in bp:
                            if s not in self.optimized_params: self.optimized_params[s] = {}
                            self.optimized_params[s]['risk_per_trade_percent'] = float(bp['risk_per_trade_percent'])
                            
                except Exception as e:
                    logger.warning(f"Failed to load optimized params for {s}: {e}")

            self.labelers[s] = AdaptiveTripleBarrier(
                horizon_ticks=s_horizon, risk_mult=s_risk, reward_mult=s_reward,
                drift_threshold=tbm_conf.get('drift_threshold', 1.5),
                horizon_type=s_horizon_type, horizon_value=s_horizon_val
            )

        self.models = {s: {'buy': None, 'sell': None} for s in symbols}
        self.meta_labelers = {s: MetaLabeler() for s in symbols}
        self.calibrators = {s: {'buy': ProbabilityCalibrator(), 'sell': ProbabilityCalibrator()} for s in symbols}
        
        self.burn_in_counters = {s: 0 for s in symbols}
        self.burn_in_limit = 5
        self.warmup_gap_bridge = {s: False for s in symbols}
        
        self.rejection_stats = {s: defaultdict(int) for s in symbols}
        self.feature_stats = {s: defaultdict(float) for s in symbols}
        self.bar_counters = {s: 0 for s in symbols}
        self.feature_importance_counter = {s: Counter() for s in symbols}
        
        self.consecutive_losses = {s: 0 for s in symbols}
        
        self.last_save_time = time.time()
        self.save_interval = 300 

        self.spread_map = CONFIG.get('forensic_audit', {}).get('spread_pips', {})
        self.l2_missing_warned = {s: False for s in symbols}
        self.last_close_prices = {s: 0.0 for s in symbols}
        
        try:
            self.server_tz = pytz.timezone(risk_conf.get('risk_timezone', 'Europe/Prague'))
        except Exception:
            self.server_tz = pytz.timezone('Europe/Prague')

        self.ker_drift_detectors = {s: drift.ADWIN(delta=0.01) for s in symbols}
        self.dynamic_ker_offsets = {s: 0.0 for s in symbols}
        
        self.redis_client = get_redis_connection(
            host=CONFIG['redis']['host'],
            port=CONFIG['redis']['port'],
            db=0,
            decode_responses=True
        )

        self._inject_auxiliary_data()
        self._init_models()
        self._load_state()
        self._perform_warmup()

    def confirm_execution(self, symbol: str, signal_id: str):
        if symbol in self.labelers:
            for trade in self.labelers[symbol].buffer:
                if trade.get('signal_id') == signal_id:
                    trade['is_executed'] = True
                    logger.debug(f"Confirming Execution for Signal {signal_id} ({symbol})")
                    break

    def _init_models(self):
        conf = CONFIG['online_learning']
        metric_map = {"LogLoss": metrics.LogLoss(), "F1": metrics.F1(), "Accuracy": metrics.Accuracy(), "ROCAUC": metrics.ROCAUC()}
        selected_metric = metric_map.get(conf.get('metric', 'LogLoss'), metrics.LogLoss())
        
        for sym in self.symbols:
            def create_pipeline():
                base_clf = forest.ARFClassifier(
                    n_models=conf.get('n_models', 50),
                    grace_period=conf['grace_period'],
                    delta=conf['delta'],
                    split_criterion='gini',
                    leaf_prediction='mc',
                    max_features=conf.get('max_features', 'sqrt'),
                    lambda_value=conf.get('lambda_value', 10),
                    metric=selected_metric,
                    warning_detector=drift.ADWIN(delta=conf.get('warning_delta', 0.001)),
                    drift_detector=drift.ADWIN(delta=conf['delta'])
                )
                return compose.Pipeline(preprocessing.StandardScaler(), base_clf)

            self.models[sym] = {
                'buy': create_pipeline(),
                'sell': create_pipeline()
            }
            self.meta_labelers[sym] = MetaLabeler()
            self.calibrators[sym] = {'buy': ProbabilityCalibrator(), 'sell': ProbabilityCalibrator()}

    def _perform_warmup(self):
        logger.info(f"{LogSymbols.TRAINING} Starting Model Pre-Training (Warm-Up)...")
        
        for sym in self.symbols:
            try:
                df = load_real_data(sym, n_candles=500000, days=30)
                if df is None or df.empty: 
                    # V20.18 FIX: explicitly warn if DB is empty to inform user of cold start.
                    logger.warning(f"⚠️ {sym} Warm-Up Data Empty! Model will start completely cold and learn dynamically.")
                    continue
                
                calibrated_thresh = self.threshold_map.get(sym)
                config_thresh = CONFIG['data'].get('volume_bar_threshold', 10.0)
                thresh = calibrated_thresh if calibrated_thresh else config_thresh

                alpha = CONFIG['data'].get('imbalance_alpha', 0.05)
                gen = AdaptiveImbalanceBarGenerator(sym, initial_threshold=thresh, alpha=alpha)
                
                bars_trained = 0
                for row in df.itertuples():
                    
                    try:
                        row_dict = row._asdict()
                    except AttributeError:
                        row_dict = row.__dict__
                        
                    price = None
                    vol = 1.0
                    ts_val = 0.0
                    
                    for k, v in row_dict.items():
                        if v is None: continue
                        k_str = str(k).lower()
                        
                        if 'close' in k_str or 'price' in k_str:
                            try: price = float(v)
                            except: pass
                        elif 'volume' in k_str or 'tick_volume' in k_str:
                            try: vol = float(v)
                            except: pass
                        elif k_str in ['index', 'time']:
                            if isinstance(v, (datetime, pd.Timestamp)):
                                ts_val = v.timestamp()
                            else:
                                try: ts_val = float(v)
                                except: pass
                                
                    if price is None or math.isnan(price): 
                        continue
                    if math.isnan(vol): 
                        vol = 1.0
                        
                    bar = gen.process_tick(price, vol, ts_val, 0.0, 0.0)
                    
                    if bar:
                        self._train_on_bar(sym, bar)
                        bars_trained += 1
                
                self.warmup_gap_bridge[sym] = True
                self.last_close_prices[sym] = 0.0 
                logger.info(f"✅ {sym} Warm-Up Complete: Trained on {bars_trained} bars.")
                
            except Exception as e:
                logger.error(f"❌ Warm-Up Failed for {sym}: {e}", exc_info=True)

    def _train_on_bar(self, symbol: str, bar: Any):
        
        bar_close = bar.close if hasattr(bar, 'close') else bar.get('close', 0.0)
        bar_volume = bar.volume if hasattr(bar, 'volume') else bar.get('volume', 1.0)
        bar_high = bar.high if hasattr(bar, 'high') else bar.get('high', bar_close)
        bar_low = bar.low if hasattr(bar, 'low') else bar.get('low', bar_close)
        
        if hasattr(bar, 'timestamp'):
            if hasattr(bar.timestamp, 'timestamp'):
                bar_ts = bar.timestamp.timestamp()
            else:
                bar_ts = float(bar.timestamp)
        else:
            bar_ts = float(bar.get('timestamp', 0.0))
            
        b_vol = getattr(bar, 'buy_vol', bar_volume/2) if hasattr(bar, 'buy_vol') else bar.get('buy_vol', bar_volume/2)
        s_vol = getattr(bar, 'sell_vol', bar_volume/2) if hasattr(bar, 'sell_vol') else bar.get('sell_vol', bar_volume/2)
        
        fe = self.feature_engineers[symbol]
        labeler = self.labelers[symbol]
        model_buy = self.models[symbol]['buy']
        model_sell = self.models[symbol]['sell']
        
        self.closes_buffer[symbol].append(bar_close)
        self.volume_buffer[symbol].append(bar_volume)
        self.bb_buffers[symbol].append(bar_close)
        self.sma_window[symbol].append(bar_close)
        
        if len(self.sma_window[symbol]) >= 2:
            prev_p = self.sma_window[symbol][-2]
            if prev_p > 0:
                self.returns_window[symbol].append(math.log(bar_close / prev_p))
        
        self.prev_features[symbol] = self.last_features[symbol]

        features = fe.update(
            price=bar_close, timestamp=bar_ts, volume=bar_volume,
            high=bar_high, low=bar_low, buy_vol=b_vol, sell_vol=s_vol,
            time_feats={'sin_hour':0, 'cos_hour':0} 
        )
        
        if features is None: return

        hurst, ker_val, rvol_val = self._calculate_golden_trio(symbol)
        features['hurst'] = hurst
        features['ker'] = ker_val
        features['rvol'] = rvol_val
        
        self.last_features[symbol] = features
        
        log_ret = features.get('log_ret', 0.0)

        resolved_labels = labeler.resolve_labels(
            bar_high, bar_low, current_close=bar_close,
            current_volume=bar_volume, current_log_ret=log_ret,
            current_timestamp=bar_ts
        )
        
        if resolved_labels:
            for (stored_feats, buy_label, sell_label, buy_ret, sell_ret, past_action, past_prob, past_ret, is_executed) in resolved_labels:
                clean_stored = {k: float(v) for k, v in stored_feats.items() if math.isfinite(v)}
                
                w_pos = float(CONFIG['online_learning'].get('positive_class_weight', 1.0))
                w_neg = float(CONFIG['online_learning'].get('negative_class_weight', 1.0))
                
                buy_base_weight = w_pos if buy_label == 1 else w_neg
                buy_ret_scalar = math.log1p(abs(buy_ret) * 100.0)
                buy_ret_scalar = max(0.5, min(buy_ret_scalar, 5.0))
                hist_ker = stored_feats.get('ker', 0.5)
                buy_final_weight = buy_base_weight * buy_ret_scalar * (hist_ker * 2.0)
                
                sell_base_weight = w_pos if sell_label == 1 else w_neg
                sell_ret_scalar = math.log1p(abs(sell_ret) * 100.0)
                sell_ret_scalar = max(0.5, min(sell_ret_scalar, 5.0))
                sell_final_weight = sell_base_weight * sell_ret_scalar * (hist_ker * 2.0)
                
                if ML_AVAILABLE:
                    model_buy.learn_one(clean_stored, buy_label, sample_weight=buy_final_weight)
                    model_sell.learn_one(clean_stored, sell_label, sample_weight=sell_final_weight)
                
                if past_action != 0:
                    self.meta_labelers[symbol].update(clean_stored, primary_action=past_action, outcome_pnl=past_ret)
                    if past_action == 1:
                        self.calibrators[symbol]['buy'].update(past_prob, 1 if past_ret > 0 else 0)
                    elif past_action == -1:
                        self.calibrators[symbol]['sell'].update(past_prob, 1 if past_ret > 0 else 0)

    def _calculate_golden_trio(self, symbol: str) -> Tuple[float, float, float]:
        closes = self.closes_buffer[symbol]
        vols = self.volume_buffer[symbol]
        hurst = 0.5
        ker = 0.5
        rvol = 1.0
        
        if len(closes) < 30: return hurst, ker, rvol
        prices = np.array(closes)
        
        try:
            if np.var(prices) < 1e-9: hurst = 0.5
            else:
                lags = range(2, 20)
                tau = []
                for lag in lags:
                    diff = np.subtract(prices[lag:], prices[:-lag])
                    std = np.std(diff)
                    tau.append(std if std > 1e-9 else 1e-9)
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst = max(0.0, min(1.0, poly[0]))
        except: pass

        try:
            diffs = np.diff(prices)
            net_change = abs(prices[-1] - prices[0])
            sum_changes = np.sum(np.abs(diffs))
            ker = net_change / sum_changes if sum_changes > 1e-9 else 0.0
        except: pass

        if len(vols) > 10:
            avg_vol = np.mean(list(vols)[:-1]) 
            rvol = vols[-1] / avg_vol if avg_vol > 1e-9 else 1.0
        
        return hurst, ker, rvol

    def process_bar(self, symbol: str, bar: VolumeBar, context_data: Dict[str, Any] = None) -> Optional[Signal]:
        if symbol not in self.symbols: return None
        
        price = bar.close if hasattr(bar, 'close') else bar.get('close', 0.0)
        bar_volume = bar.volume if hasattr(bar, 'volume') else bar.get('volume', 1.0)
        bar_high = bar.high if hasattr(bar, 'high') else bar.get('high', price)
        bar_low = bar.low if hasattr(bar, 'low') else bar.get('low', price)
        
        if hasattr(bar, 'timestamp'):
            if hasattr(bar.timestamp, 'timestamp'):
                bar_ts = bar.timestamp.timestamp()
            else:
                bar_ts = float(bar.timestamp)
        else:
            bar_ts = float(bar.get('timestamp', 0.0))
            
        buy_vol = getattr(bar, 'buy_vol', bar_volume/2) if hasattr(bar, 'buy_vol') else bar.get('buy_vol', bar_volume/2)
        sell_vol = getattr(bar, 'sell_vol', bar_volume/2) if hasattr(bar, 'sell_vol') else bar.get('sell_vol', bar_volume/2)
        
        if time.time() - self.last_save_time > self.save_interval:
            self.save_state()
            self.last_save_time = time.time()

        fe = self.feature_engineers[symbol]
        labeler = self.labelers[symbol]
        model_buy = self.models[symbol]['buy']
        model_sell = self.models[symbol]['sell']
        meta_labeler = self.meta_labelers[symbol]
        stats = self.rejection_stats[symbol]
        
        self.bar_counters[symbol] += 1
        
        is_bridging_gap = False
        if self.warmup_gap_bridge[symbol]:
            is_bridging_gap = True
            logger.info(f"🌉 BRIDGING GAP {symbol}: Syncing Live Price {price:.5f}")
            self.warmup_gap_bridge[symbol] = False
            self.last_close_prices[symbol] = price
        
        prev_close = self.last_close_prices.get(symbol, price)
        self.last_close_prices[symbol] = price

        self.closes_buffer[symbol].append(price)
        self.volume_buffer[symbol].append(bar_volume)
        self.sma_window[symbol].append(price)
        
        if len(self.sma_window[symbol]) >= 2:
            prev_p = self.sma_window[symbol][-2]
            if prev_p > 0:
                self.returns_window[symbol].append(math.log(price / prev_p))

        self.bb_buffers[symbol].append(price)

        if buy_vol == 0 and sell_vol == 0:
            if not self.l2_missing_warned[symbol]:
                logger.warning(f"⚠️ {symbol}: Zero Flow Detected. Using Tick Rule.")
                self.l2_missing_warned[symbol] = True
            effective_vol = bar_volume if bar_volume > 0 else 1.0
            if price > prev_close: buy_vol = effective_vol; sell_vol = 0.0
            elif price < prev_close: buy_vol = 0.0; sell_vol = effective_vol
            else: buy_vol = effective_vol / 2.0; sell_vol = effective_vol / 2.0

        self.prev_features[symbol] = self.last_features[symbol]

        features = fe.update(
            price=price, timestamp=bar_ts, volume=bar_volume,
            high=bar_high, low=bar_low, buy_vol=buy_vol, sell_vol=sell_vol,
            time_feats={}, context_data=context_data
        )
        
        if features is None: return None
        
        hurst, ker_val, rvol_val = self._calculate_golden_trio(symbol)
        features['hurst'] = hurst
        features['ker'] = ker_val
        features['rvol'] = rvol_val
        
        self.last_features[symbol] = features
        
        parkinson = features.get('parkinson_vol', 0.0)
        log_ret = features.get('log_ret', 0.0)

        self.ker_drift_detectors[symbol].update(ker_val)
        if self.ker_drift_detectors[symbol].drift_detected:
            self.dynamic_ker_offsets[symbol] = max(-0.10, self.dynamic_ker_offsets[symbol] - 0.05)
        else:
            self.dynamic_ker_offsets[symbol] = min(0.0, self.dynamic_ker_offsets[symbol] + 0.001)

        try:
            resolved_labels = labeler.resolve_labels(
                bar_high, bar_low, current_close=price, 
                current_volume=bar_volume, current_log_ret=log_ret,
                current_timestamp=bar_ts
            )
            
            if resolved_labels:
                for (stored_feats, buy_label, sell_label, buy_ret, sell_ret, past_action, past_prob, past_ret, is_executed) in resolved_labels:
                    
                    if is_executed:
                        if past_ret > 0: self.consecutive_losses[symbol] = 0
                        elif past_ret < 0: self.consecutive_losses[symbol] += 1

                    w_pos = float(CONFIG['online_learning'].get('positive_class_weight', 1.0))
                    w_neg = float(CONFIG['online_learning'].get('negative_class_weight', 1.0))
                    
                    buy_base_weight = w_pos if buy_label == 1 else w_neg
                    buy_ret_scalar = math.log1p(abs(buy_ret) * 100.0)
                    buy_ret_scalar = max(0.5, min(buy_ret_scalar, 5.0))
                    hist_ker = stored_feats.get('ker', 0.5)
                    buy_final_weight = buy_base_weight * buy_ret_scalar * (hist_ker * 2.0)
                    
                    sell_base_weight = w_pos if sell_label == 1 else w_neg
                    sell_ret_scalar = math.log1p(abs(sell_ret) * 100.0)
                    sell_ret_scalar = max(0.5, min(sell_ret_scalar, 5.0))
                    sell_final_weight = sell_base_weight * sell_ret_scalar * (hist_ker * 2.0)
                    
                    clean_stored = {k: float(v) for k, v in stored_feats.items() if math.isfinite(v)}
                    
                    if ML_AVAILABLE:
                        model_buy.learn_one(clean_stored, buy_label, sample_weight=buy_final_weight)
                        model_sell.learn_one(clean_stored, sell_label, sample_weight=sell_final_weight)
                    
                    if past_action != 0:
                        meta_labeler.update(clean_stored, primary_action=past_action, outcome_pnl=past_ret)
                        if past_action == 1:
                            self.calibrators[symbol]['buy'].update(past_prob, 1 if past_ret > 0 else 0)
                        elif past_action == -1:
                            self.calibrators[symbol]['sell'].update(past_prob, 1 if past_ret > 0 else 0)
                            
        except Exception as e:
            logger.error(f"Training Loop Crash: {e}", exc_info=True)

        current_atr = features.get('atr', 0.001)
        pip_val, _ = RiskManager.get_pip_info(symbol)
        if pip_val <= 0: pip_val = 0.0001
        
        spread_pips = float(CONFIG.get('forensic_audit', {}).get('spread_pips', {}).get(symbol, 1.6))
        spread_in_price = spread_pips * pip_val
        comm_in_price = 0.5 * pip_val
        
        min_stop_pips = float(CONFIG.get('risk_management', {}).get('min_stop_loss_pips', 20.0))
        min_stop_dist = min_stop_pips * pip_val
        min_profit_pips = float(CONFIG.get('online_learning', {}).get('tbm', {}).get('min_profit_pips', 40.0))
        min_profit_dist = min_profit_pips * pip_val

        is_trending = hurst >= self.hurst_breakout
        proposed_action = 0 
        
        regime_label = "TREND_BREAKOUT" if is_trending else "MEAN_REVERSION"

        clean_features = {k: float(v) for k, v in features.items() if math.isfinite(v)}
        
        # --- V20.18 FIX: DUAL MODEL EV PROTOCOL (With Cold Start Prevention) ---
        prob_buy = 0.0
        prob_sell = 0.0
        
        try:
            if ML_AVAILABLE:
                pred_proba_buy = model_buy.predict_proba_one(clean_features)
                # If the model is completely untrained, it returns an empty dict {}. 
                # We default to 0.5 (neutral prior) to prevent a perpetual cold-start lockup.
                prob_buy = pred_proba_buy.get(1, 0.5 if not pred_proba_buy else 0.0)
                
                pred_proba_sell = model_sell.predict_proba_one(clean_features)
                prob_sell = pred_proba_sell.get(1, 0.5 if not pred_proba_sell else 0.0)
            else:
                prob_buy, prob_sell = 0.5, 0.5
        except:
            prob_buy, prob_sell = 0.5, 0.5

        current_reward_target = max(self.labelers[symbol].reward_mult, 2.0)
        if is_trending:
            current_reward_target = max(current_reward_target, 3.0)

        break_even_wr = 1.0 / (1.0 + current_reward_target)
        
        streak = self.consecutive_losses.get(symbol, 0)
        tilt_penalty = 0.03 * streak
        
        edge_thresh = self.optimized_params.get(symbol, {}).get('model_edge_threshold', 0.01)
        required_prob = break_even_wr + edge_thresh + tilt_penalty
        
        cal_prob_buy = self.calibrators[symbol]['buy'].calibrate(prob_buy)
        cal_prob_sell = self.calibrators[symbol]['sell'].calibrate(prob_sell)
        
        buy_edge = cal_prob_buy - required_prob
        sell_edge = cal_prob_sell - required_prob

        # Select action purely based on highest positive edge
        if buy_edge > 0 and buy_edge >= sell_edge:
            proposed_action = 1
            prob_success = prob_buy 
            confidence = cal_prob_buy
        elif sell_edge > 0 and sell_edge > buy_edge:
            proposed_action = -1
            prob_success = prob_sell 
            confidence = cal_prob_sell
        else:
            proposed_action = 0
            prob_success = 0.0
            confidence = max(cal_prob_buy, cal_prob_sell)

        signal_id = str(uuid.uuid4())

        self.labelers[symbol].add_trade_opportunity(
            features=features, entry_price=price, current_atr=current_atr, 
            timestamp=bar_ts, parkinson_vol=parkinson,
            min_stop_dist=min_stop_dist, spread_in_price=spread_in_price,
            comm_in_price=comm_in_price, min_profit_dist=min_profit_dist, 
            proposed_action=proposed_action, pred_proba=prob_success,
            override_reward_mult=current_reward_target, signal_id=signal_id
        )

        if self.burn_in_counters[symbol] < 5:
            self.burn_in_counters[symbol] += 1
            return Signal(symbol, "WARMUP", 0.0, {})

        if is_bridging_gap:
            return Signal(symbol, "HOLD", 0.0, {"reason": "Bridging Gap"})

        # V20.18 FIX: Added occasional diagnostic logging so you can see the math working 
        # when the bot holds due to negative expected value.
        if proposed_action == 0:
            stats["Negative EV (Math Hold)"] += 1
            if stats["Negative EV (Math Hold)"] % 50 == 0:
                logger.info(f"📐 {symbol} GATE: Negative EV | BuyEdge: {buy_edge:.2f} | SellEdge: {sell_edge:.2f}")
            return Signal(symbol, "HOLD", 0.0, {"reason": "Negative EV"})

        effective_ker_thresh = max(0.0001, self.ker_floor + self.dynamic_ker_offsets[symbol])
        if ker_val < effective_ker_thresh:
            stats["Low KER"] += 1
            if stats["Low KER"] % 50 == 0:
                logger.info(f"🛡️ {symbol} GATE: Low Efficiency (KER {ker_val:.4f} < {effective_ker_thresh:.4f})")
            return Signal(symbol, "HOLD", 0.0, {"reason": "Low KER"})

        try:
            base_meta_thresh = self.optimized_params.get(symbol, {}).get('meta_labeling_threshold', 0.50)
            meta_thresh = max(base_meta_thresh + (0.02 * streak), 0.40)
            
            is_profitable = meta_labeler.predict(clean_features, proposed_action, threshold=meta_thresh)
            
            open_positions = context_data.get('positions', {}) if context_data else {}
            
            is_pyramid = False
            if symbol in open_positions:
                pyramid_config = CONFIG.get('risk_management', {}).get('pyramiding', {})
                if not pyramid_config.get('enabled', False):
                    return Signal(symbol, "HOLD", confidence, {"reason": "Pyramiding Disabled (Trade Open)"})
                
                pos = open_positions[symbol]
                entry_price = float(pos.get('entry_price', 0.0))
                sl_price = float(pos.get('sl', 0.0))
                pos_type = str(pos.get('type', '')).upper()
                
                if (pos_type == "BUY" and proposed_action != 1) or (pos_type == "SELL" and proposed_action != -1):
                    return Signal(symbol, "HOLD", confidence, {"reason": "Pyramiding Direction Clash"})

                risk_dist = 0.0
                try:
                    comment = str(pos.get('comment', ''))
                    short_id = None
                    if "Auto_" in comment:
                        parts = comment.split('_')
                        if len(parts) >= 2: short_id = parts[1][:8]
                    elif len(comment) >= 8:
                        short_id = comment[:8]
                        
                    if short_id:
                        stored_risk = self.redis_client.hget("bot:initial_risk", short_id) if hasattr(self, 'redis_client') else None
                        if stored_risk: risk_dist = float(stored_risk)
                except Exception as e:
                    logger.warning(f"Failed to fetch initial risk from Redis for {symbol}: {e}")
                
                if risk_dist <= 1e-5:
                    risk_dist = abs(entry_price - sl_price)
                    if risk_dist < 1e-5: return Signal(symbol, "HOLD", confidence, {"reason": "Pyramiding Risk Dist 0"})

                dist = (price - entry_price) if pos_type == "BUY" else (entry_price - price)
                current_r = dist / risk_dist if risk_dist > 0 else 0
                
                if current_r < pyramid_config.get('add_on_profit_r', 1.0):
                    logger.info(f"🧱 {symbol} GATE: Waiting for Open Trade Profit (> {pyramid_config.get('add_on_profit_r', 1.0)}R)")
                    return Signal(symbol, "HOLD", confidence, {"reason": f"Pyramiding Limit"})
                
                is_pyramid = True

            if is_profitable:
                action_str = "BUY" if proposed_action == 1 else "SELL"
                
                imp_feats = [regime_label]
                if rvol_val > 2.0: imp_feats.append('High_Fuel')
                if hurst > 0.6: imp_feats.append('High_Hurst')
                for f in imp_feats: self.feature_importance_counter[symbol][f] += 1

                opt_risk = self.optimized_params.get(symbol, {}).get('risk_per_trade_percent')
                tighten_stops = (rvol_val > self.rvol_trigger)

                self.last_trade_bar[symbol] = self.bar_counters[symbol]
                self.last_trade_direction[symbol] = proposed_action

                logger.info(f"🔥 {symbol} ML TRIGGER ACTIVATED! Confidence: {confidence:.2f}")

                return Signal(symbol, action_str, confidence, {
                    "meta_ok": True,
                    "volatility": features.get('volatility', 0.001),
                    "atr": current_atr,
                    "ker": ker_val,
                    "parkinson_vol": parkinson,
                    "rvol": rvol_val,
                    "hurst": hurst,
                    "amihud": features.get('amihud', 0.0),
                    "regime": regime_label,
                    "drivers": imp_feats,
                    "optimized_rr": current_reward_target,
                    "risk_percent_override": opt_risk,
                    "pyramid": is_pyramid, 
                    "tighten_stops": tighten_stops 
                }, signal_id=signal_id)
            else:
                stats['Meta-Labeler Reject'] += 1
                if stats['Meta-Labeler Reject'] % 10 == 0:
                    logger.info(f"🧠 {symbol} GATE: Meta-Labeler Veto (Predicts Chop/Loss)")
                return Signal(symbol, "HOLD", confidence, {"reason": f"Meta Rejected"})

        except Exception as e:
            logger.error(f"Strategy ML Eval Error: {e}", exc_info=True)
            return Signal(symbol, "HOLD", 0.0, {"reason": "Exception during ML Check"})

    def _inject_auxiliary_data(self):
        defaults = {
            "USDJPY": 150.0, "GBPUSD": 1.25, "EURUSD": 1.08,
            "USDCAD": 1.35, "USDCHF": 0.90, "AUDUSD": 0.65, "NZDUSD": 0.60,
            "GBPJPY": 190.0, "EURJPY": 160.0, "AUDJPY": 95.0,
            "GBPAUD": 1.95, "EURAUD": 1.65, "GBPNZD": 2.05
        }
        for sym, price in defaults.items():
            if sym not in self.last_close_prices or self.last_close_prices[sym] == 0:
                self.last_close_prices[sym] = price

    def save_state(self):
        for sym in self.symbols:
            def safe_dump(obj, filename):
                try:
                    with open(self.models_dir / filename, "wb") as f:
                        pickle.dump(obj, f)
                except Exception as e:
                    logger.debug(f"Skipped saving {filename}: {e}")

            safe_dump(self.models[sym]['buy'], f"river_pipeline_buy_{sym}.pkl")
            safe_dump(self.models[sym]['sell'], f"river_pipeline_sell_{sym}.pkl")
            
            safe_dump(self.meta_labelers[sym], f"meta_model_{sym}.pkl")
            safe_dump(self.calibrators[sym], f"calibrators_{sym}.pkl")
            safe_dump(self.feature_engineers[sym], f"feature_engineer_{sym}.pkl")
            safe_dump(self.labelers[sym], f"labeler_{sym}.pkl")
            
            try:
                buffer_state = {
                    'closes': self.closes_buffer[sym],
                    'volumes': self.volume_buffer[sym],
                    'bb': self.bb_buffers[sym],
                    'sniper_closes': self.sniper_closes[sym],
                    'sniper_rsi': self.sniper_rsi[sym],
                    'sma_window': self.sma_window[sym],
                    'returns': self.returns_window[sym],
                    'last_features': self.last_features[sym],
                    'prev_features': self.prev_features[sym],
                    'last_trade_bar': self.last_trade_bar[sym],
                    'last_trade_direction': self.last_trade_direction[sym]
                }
                with open(self.models_dir / f"buffers_{sym}.pkl", "wb") as f:
                    pickle.dump(buffer_state, f)
            except Exception as e:
                logger.debug(f"Failed to save buffers for {sym}: {e}")
                
        logger.info(f"{LogSymbols.DATABASE} Models & State Auto-Saved.")

    def _load_state(self):
        loaded_count = 0
        for sym in self.symbols:
            buy_model_path = self.models_dir / f"river_pipeline_buy_{sym}.pkl"
            sell_model_path = self.models_dir / f"river_pipeline_sell_{sym}.pkl"
            meta_path = self.models_dir / f"meta_model_{sym}.pkl"
            fe_path = self.models_dir / f"feature_engineer_{sym}.pkl"
            buf_path = self.models_dir / f"buffers_{sym}.pkl"
            cal_path = self.models_dir / f"calibrators_{sym}.pkl"
            labeler_path = self.models_dir / f"labeler_{sym}.pkl"
            
            if buy_model_path.exists():
                try:
                    with open(buy_model_path, "rb") as f: self.models[sym]['buy'] = pickle.load(f)
                    loaded_count += 1
                except Exception: pass
                
            if sell_model_path.exists():
                try:
                    with open(sell_model_path, "rb") as f: self.models[sym]['sell'] = pickle.load(f)
                    loaded_count += 1
                except Exception: pass
            
            if meta_path.exists():
                try:
                    with open(meta_path, "rb") as f: self.meta_labelers[sym] = pickle.load(f)
                except Exception: pass

            if cal_path.exists():
                try:
                    with open(cal_path, "rb") as f: self.calibrators[sym] = pickle.load(f)
                except Exception: pass
            
            if fe_path.exists():
                try:
                    with open(fe_path, "rb") as f: self.feature_engineers[sym] = pickle.load(f)
                except Exception: pass

            if labeler_path.exists():
                try:
                    with open(labeler_path, "rb") as f: self.labelers[sym] = pickle.load(f)
                except Exception: pass

            if buf_path.exists():
                try:
                    with open(buf_path, "rb") as f:
                        buf_state = pickle.load(f)
                        self.closes_buffer[sym].extend(buf_state.get('closes', []))
                        self.volume_buffer[sym].extend(buf_state.get('volumes', []))
                        self.bb_buffers[sym].extend(buf_state.get('bb', []))
                        self.sniper_closes[sym].extend(buf_state.get('sniper_closes', []))
                        self.sniper_rsi[sym].extend(buf_state.get('sniper_rsi', []))
                        self.sma_window[sym].extend(buf_state.get('sma_window', []))
                        self.returns_window[sym].extend(buf_state.get('returns', []))
                        self.last_features[sym] = buf_state.get('last_features')
                        self.prev_features[sym] = buf_state.get('prev_features')
                        self.last_trade_bar[sym] = buf_state.get('last_trade_bar', 0)
                        self.last_trade_direction[sym] = buf_state.get('last_trade_direction', 0)
                except Exception as e:
                    logger.warning(f"Failed to load buffers for {sym}: {e}")
            
        if loaded_count > 0:
            logger.info(f"{LogSymbols.SUCCESS} Loaded {loaded_count} existing models (Dual-Model Format).")