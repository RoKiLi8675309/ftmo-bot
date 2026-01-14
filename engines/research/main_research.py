# =============================================================================
# FILENAME: engines/research/main_research.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/main_research.py
# DEPENDENCIES: shared, engines.research.backtester, engines.research.strategy, pyyaml
# DESCRIPTION: CLI Entry point for Research, Training, and Backtesting.
# 
# PHOENIX V14.0 UPDATE (AGGRESSOR OPTIMIZATION):
# 1. RISK SPACE: Expanded search to include 2.0% risk (House Money Tier).
# 2. SIGNIFICANCE: Enforces min_trades from config (100) to filter luck.
# 3. ALIGNMENT: Ensures optimization objective matches Aggressor goals.
# =============================================================================
import os
import sys

# --- CRITICAL STABILITY FIX: FORCE SINGLE THREADING FOR MATH LIBS ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import logging
import pickle
import json
import time
import math
import uuid
import gc
import optuna
import numpy as np
import pandas as pd
import psutil
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from joblib import Parallel, delayed
from river import compose, preprocessing, forest, metrics, ensemble, drift

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after path fix
try:
    from engines.research.backtester import BacktestBroker, MarketSnapshot
    from engines.research.strategy import ResearchStrategy
    from shared import CONFIG, setup_logging, load_real_data, LogSymbols, AdaptiveImbalanceBarGenerator, RiskManager
except ImportError as e:
    print(f"CRITICAL: Failed to import dependencies. Ensure you are running from the project root or 'shared' is accessible.\nError: {e}")
    sys.exit(1)

setup_logging("Research")
log = logging.getLogger("Research")

# --- 1. TELEMETRY & REPORTING UTILS ---

class EmojiCallback:
    """
    Injects FTMO-style Emojis and RICH TELEMETRY into Optuna Trial reporting.
    """
    def __call__(self, study, trial):
        val = trial.value
        attrs = trial.user_attrs
        
        # Extract Symbol from Study Name (Convention: "study_SYMBOL")
        symbol = study.study_name.replace("study_", "")
        
        # 1. Determine Status Icon & Rank
        risk_pct = attrs.get('risk_pct', 0.0) # Retrieved from attributes
        trades = attrs.get('trades', 0)
        pnl = attrs.get('pnl', 0.0)
        dd = attrs.get('max_dd_pct', 0.0) * 100
        calmar = attrs.get('calmar', 0.0)
        
        status = "FAIL"
        icon = "🔻"
        
        if attrs.get('blown', False):
            icon = "💀" # Blown Account (>8% DD or Ruin)
            status = "BLOWN"
        elif dd > 8.0: # HARD LIMIT: 8.0%
            icon = "⚠️" # High Risk
            status = "RISKY"
        elif attrs.get('pruned', False):
            icon = "✂️" # Pruned (Low Trades)
            status = "PRUNE"
        elif pnl > 1000 and calmar > 2.0:
            icon = "🚀" # High Quality & High Profit
            status = "ALPHA"
        elif pnl > 0:
            icon = "✅" # Profitable
            status = "PASS"

        # 2. Extract Metrics
        wr = attrs.get('win_rate', 0.0) * 100
        pf = attrs.get('profit_factor', 0.0)
        
        # 3. Format Output (Column Aligned)
        # Structure: [ICON] STATUS | SYMBOL | ID | RISK | SCORE | PnL | DD | WR | TRADES
        msg = (
            f"{icon} {status:<6} | {symbol:<6} | Trial {trial.number:<3} | "
            f"R: {risk_pct:>4.3f}% | "
            f"🏆 {val:>8.1f} | "
            f"💰 ${pnl:>9,.0f} | " # Compacted PnL for space
            f"📉 {dd:>5.2f}% | "
            f"🎯 {wr:>5.1f}% | "
            f"⚡ {pf:>4.2f} | "
            f"#️⃣ {trades:<4}"
        )
        
        # 4. Log to File & Console
        log.info(msg.strip())
        
        # 5. Detailed Analysis (Victory Lap OR Autopsy)
        if 'autopsy' in attrs:
            # Victory Lap for significant profit
            if pnl > 1000.0 and not attrs.get('blown', False):
                report_type = "🏁 VICTORY LAP"
                report_msg = f"\n{report_type} (Trial {trial.number}): {attrs['autopsy'].strip()}\n" + ("-" * 80)
                log.info(report_msg)
            
            # Failure Autopsy for ANY loss or zero trades
            elif pnl < 0 or trades == 0 or attrs.get('blown', False):
                report_type = "🔎 FAILURE AUTOPSY"
                report_msg = f"\n{report_type} (Trial {trial.number}): {attrs['autopsy'].strip()}\n" + ("-" * 80)
                log.info(report_msg)

def process_data_into_bars(symbol: str, n_ticks: int = 4000000) -> pd.DataFrame:
    """
    Helper to Load Ticks -> Aggregate to Tick Imbalance Bars (TIBs) -> Return Clean DataFrame.
    """
    # 1. Load Massive Amount of Ticks (To get sufficient Bars)
    raw_ticks = load_real_data(symbol, n_candles=n_ticks, days=730 * 2)
    
    if raw_ticks.empty:
        return pd.DataFrame()

    # 2. V10.1 AGGREGATION: ADAPTIVE IMBALANCE BARS WITH AUTO-CALIBRATION
    # Fetch params from config (now defaults to 10 in V12.5)
    config_threshold = CONFIG['data'].get('volume_bar_threshold', 10) 
    alpha = CONFIG['data'].get('imbalance_alpha', 0.05)
    
    # --- AUTO-CALIBRATION LOOP ---
    current_threshold = config_threshold
    bars_list = []
    min_bars_needed = 500
    attempts = 0
    max_attempts = 4
    
    while attempts < max_attempts:
        gen = AdaptiveImbalanceBarGenerator(
            symbol=symbol,
            initial_threshold=current_threshold,
            alpha=alpha
        )
        
        temp_bars = []
        # Iterate through ticks efficiently
        for row in raw_ticks.itertuples():
            price = getattr(row, 'price', getattr(row, 'close', None))
            vol = getattr(row, 'volume', 1.0)
            
            # Handle timestamp
            ts = getattr(row, 'Index', getattr(row, 'time', None))
            if isinstance(ts, (datetime, pd.Timestamp)):
                ts_val = ts.timestamp()
            else:
                ts_val = float(ts)
                
            # L2 Data
            b_vol = 0.0
            s_vol = 0.0
            
            if price is None: continue

            bar = gen.process_tick(price, vol, ts_val, b_vol, s_vol)
            
            if bar:
                temp_bars.append({
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
        
        if len(temp_bars) >= min_bars_needed:
            bars_list = temp_bars
            if attempts > 0:
                log.info(f"✅ {symbol}: Auto-Calibrated Threshold to {current_threshold} (Generated {len(bars_list)} bars)")
            break
        else:
            # Not enough bars, lower threshold
            attempts += 1
            new_threshold = max(5.0, current_threshold * 0.5) # Lower limit
            log.warning(f"⚠️ {symbol}: Insufficient bars ({len(temp_bars)}). Retrying with threshold {new_threshold}...")
            current_threshold = new_threshold
            
    if not bars_list:
        log.warning(f"❌ {symbol}: Failed to generate enough bars even after calibration.")
        return pd.DataFrame()

    # 3. Convert to DataFrame
    df_bars = pd.DataFrame(bars_list)
    
    # 4. Set Index to Datetime for Snapshot compatibility
    df_bars['time'] = pd.to_datetime(df_bars['timestamp'], unit='s', utc=True)
    df_bars.set_index('time', inplace=True, drop=False)
    
    return df_bars

# --- 2. WORKER FUNCTIONS (ISOLATED PROCESSES) ---

def _worker_optimize_task(symbol: str, n_trials: int, train_candles: int, db_url: str) -> None:
    """
    ISOLATED WORKER FUNCTION: Runs in a separate process.
    Executes Global Optimization using "PROFIT IS KING" Logic.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    setup_logging(f"Worker_{symbol}")
    log = logging.getLogger(f"Worker_{symbol}")
    optuna.logging.set_verbosity(optuna.logging.WARN)
    
    try:
        # Use dynamic train_candles passed from pipeline
        df = process_data_into_bars(symbol, n_ticks=train_candles)
        if df.empty: return
        log.info(f"📥 {symbol}: Generated {len(df)} Imbalance Bars for training.")

        def objective(trial):
            space = CONFIG['optimization_search_space']
            params = CONFIG['online_learning'].copy()
            
            # --- HYPERPARAMETER MAPPING ---
            params['n_models'] = trial.suggest_int('n_models', space['n_models']['min'], space['n_models']['max'], step=space['n_models']['step'])
            params['grace_period'] = trial.suggest_int('grace_period', space['grace_period']['min'], space['grace_period']['max'], step=space['grace_period']['step'])
            params['delta'] = trial.suggest_float('delta', float(space['delta']['min']), float(space['delta']['max']), log=space['delta'].get('log', True))
            params['lambda_value'] = trial.suggest_int('lambda_value', space['lambda_value']['min'], space['lambda_value']['max'], step=space['lambda_value']['step'])
            params['max_features'] = trial.suggest_categorical('max_features', ['log2', 'sqrt'])
            
            params['entropy_threshold'] = trial.suggest_float('entropy_threshold', space['entropy_threshold']['min'], space['entropy_threshold']['max'])
            params['vpin_threshold'] = trial.suggest_float('vpin_threshold', space['vpin_threshold']['min'], space['vpin_threshold']['max'])
            
            params['tbm'] = {
                'barrier_width': trial.suggest_float('barrier_width', space['tbm_barrier_width']['min'], space['tbm_barrier_width']['max']),
                'horizon_minutes': trial.suggest_int('horizon_minutes', space['tbm_horizon_minutes']['min'], space['tbm_horizon_minutes']['max'], step=space['tbm_horizon_minutes']['step']),
                'drift_threshold': trial.suggest_float('drift_threshold', space['tbm_drift_threshold']['min'], space['tbm_drift_threshold']['max'])
            }
            
            params['min_calibrated_probability'] = trial.suggest_float('min_calibrated_probability', space['min_calibrated_probability']['min'], space['min_calibrated_probability']['max'])
            
            # V14.0 AGGRESSOR PROTOCOL: Updated Risk Search Space
            # Includes 0.5% (Prob), 1.0% (Base), 1.5% (Strong), 2.0% (House Money)
            risk_options = [0.005, 0.010, 0.015, 0.020] 
            params['risk_per_trade_percent'] = trial.suggest_categorical('risk_per_trade_percent', risk_options)
            trial.set_user_attr("risk_pct", params['risk_per_trade_percent'] * 100)
            
            pipeline_inst = ResearchPipeline()
            init_bal = CONFIG['env'].get('initial_balance', 100000.0)
            
            broker = BacktestBroker(initial_balance=init_bal)
            model = pipeline_inst.get_fresh_model(params)
            strategy = ResearchStrategy(model, symbol, params)
            
            for index, row in df.iterrows():
                snapshot = MarketSnapshot(timestamp=index, data=row)
                if snapshot.get_price(symbol, 'close') == 0: continue
                broker.process_pending(snapshot)
                strategy.on_data(snapshot, broker)
                if broker.is_blown: break
            
            metrics = pipeline_inst.calculate_performance_metrics(broker.trade_log, broker.initial_balance)
            
            # --- PROFIT IS KING OBJECTIVE ---
            total_return = metrics['total_pnl']
            max_dd_pct = metrics['max_dd_pct']
            trades = metrics['total_trades']
            safe_dd = max_dd_pct if max_dd_pct > 0.001 else 0.001
            calmar = (total_return / init_bal) / safe_dd

            trial.set_user_attr("autopsy", strategy.generate_autopsy())
            trial.set_user_attr("pnl", total_return)
            trial.set_user_attr("max_dd_pct", max_dd_pct)
            trial.set_user_attr("win_rate", metrics['win_rate'])
            trial.set_user_attr("trades", trades)
            trial.set_user_attr("profit_factor", metrics['profit_factor'])
            trial.set_user_attr("risk_reward_ratio", metrics['risk_reward_ratio'])
            trial.set_user_attr("sqn", metrics['sqn'])
            trial.set_user_attr("calmar", calmar)

            # V12.3: HARD 8% DRAWDOWN LIMIT
            if max_dd_pct > 0.08: 
                trial.set_user_attr("blown", True)
                return -10000.0 
            
            if broker.is_blown:
                trial.set_user_attr("blown", True)
                return -10000.0

            # V14.0: SIGNIFICANCE FILTER
            min_trades = CONFIG['wfo'].get('min_trades_optimization', 100)
            if trades < min_trades:
                trial.set_user_attr("pruned", True)
                return 0.0 
            
            objective_score = total_return + (calmar * 100.0)
            return objective_score

        study_name = f"study_{symbol}"
        for _ in range(3):
            try:
                study = optuna.load_study(study_name=study_name, storage=db_url)
                break
            except Exception:
                time.sleep(1)
        
        study.optimize(objective, n_trials=n_trials, callbacks=[EmojiCallback()])
        del df
        gc.collect()
        
    except Exception as e:
        log.error(f"CRITICAL WORKER ERROR ({symbol}): {e}")

def _worker_wfo_task(symbol: str, n_trials: int, db_url: str):
    """
    ISOLATED WORKER FUNCTION: Runs Walk-Forward Optimization.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    setup_logging(f"WFO_{symbol}")
    log = logging.getLogger(f"WFO_{symbol}")
    optuna.logging.set_verbosity(optuna.logging.WARN)
    
    try:
        # 1. Load All Data - Dynamic config load
        train_ticks = CONFIG['data'].get('num_candles_train', 10000000)
        df = process_data_into_bars(symbol, n_ticks=train_ticks)
        if df.empty: return
        
        # 2. Define Window Params
        train_months = CONFIG['wfo'].get('train_years', 2) * 12
        test_months = CONFIG['wfo'].get('test_months', 6)
        
        # Create Sliding Windows
        start_date = df.index.min()
        end_date = df.index.max()
        
        current_train_start = start_date
        wfo_results = []
        
        log.info(f"🔄 Starting WFO Loop for {symbol}. Range: {start_date} -> {end_date}")
        
        while True:
            train_end = current_train_start + pd.DateOffset(months=train_months)
            test_end = train_end + pd.DateOffset(months=test_months)
            
            if test_end > end_date:
                break
                
            # Slice Data
            df_train = df[(df.index >= current_train_start) & (df.index < train_end)]
            df_test = df[(df.index >= train_end) & (df.index < test_end)]
            
            if df_train.empty or df_test.empty:
                current_train_start += pd.DateOffset(months=test_months)
                continue
                
            window_id = f"{train_end.strftime('%Y-%m')}"
            log.info(f"🪟 Processing Window {window_id} (Train: {len(df_train)} bars, Test: {len(df_test)} bars)")
            
            # --- OPTIMIZATION STEP (IN-SAMPLE) ---
            study_name = f"wfo_{symbol}_{window_id}"
            storage_url = f"sqlite:///wfo_{symbol}.db" 
            
            try:
                optuna.delete_study(study_name=study_name, storage=storage_url)
            except: pass
            
            study = optuna.create_study(study_name=study_name, storage=storage_url, direction="maximize")
            
            def objective(trial):
                space = CONFIG['optimization_search_space']
                params = CONFIG['online_learning'].copy()
                
                params['n_models'] = trial.suggest_int('n_models', space['n_models']['min'], space['n_models']['max'], step=space['n_models']['step'])
                params['grace_period'] = trial.suggest_int('grace_period', space['grace_period']['min'], space['grace_period']['max'], step=space['grace_period']['step'])
                params['delta'] = trial.suggest_float('delta', float(space['delta']['min']), float(space['delta']['max']), log=space['delta'].get('log', True))
                params['lambda_value'] = trial.suggest_int('lambda_value', space['lambda_value']['min'], space['lambda_value']['max'], step=space['lambda_value']['step'])
                params['max_features'] = trial.suggest_categorical('max_features', ['log2', 'sqrt'])
                params['entropy_threshold'] = trial.suggest_float('entropy_threshold', space['entropy_threshold']['min'], space['entropy_threshold']['max'])
                params['vpin_threshold'] = trial.suggest_float('vpin_threshold', space['vpin_threshold']['min'], space['vpin_threshold']['max'])
                params['tbm'] = {
                    'barrier_width': trial.suggest_float('barrier_width', space['tbm_barrier_width']['min'], space['tbm_barrier_width']['max']),
                    'horizon_minutes': trial.suggest_int('horizon_minutes', space['tbm_horizon_minutes']['min'], space['tbm_horizon_minutes']['max'], step=space['tbm_horizon_minutes']['step']),
                    'drift_threshold': trial.suggest_float('drift_threshold', space['tbm_drift_threshold']['min'], space['tbm_drift_threshold']['max'])
                }
                params['min_calibrated_probability'] = trial.suggest_float('min_calibrated_probability', space['min_calibrated_probability']['min'], space['min_calibrated_probability']['max'])
                
                # V14.0 AGGRESSOR: Updated Risk Search for WFO
                risk_options = [0.005, 0.010, 0.015, 0.020]
                params['risk_per_trade_percent'] = trial.suggest_categorical('risk_per_trade_percent', risk_options)
                
                pipeline_inst = ResearchPipeline()
                broker = BacktestBroker(initial_balance=CONFIG['env']['initial_balance'])
                model = pipeline_inst.get_fresh_model(params)
                strategy = ResearchStrategy(model, symbol, params)
                
                for index, row in df_train.iterrows():
                    snapshot = MarketSnapshot(timestamp=index, data=row)
                    if snapshot.get_price(symbol, 'close') == 0: continue
                    broker.process_pending(snapshot)
                    strategy.on_data(snapshot, broker)
                    if broker.is_blown: break
                
                metrics = pipeline_inst.calculate_performance_metrics(broker.trade_log, broker.initial_balance)
                
                total_return = metrics['total_pnl']
                max_dd_pct = metrics['max_dd_pct']
                trades = metrics['total_trades']
                safe_dd = max_dd_pct if max_dd_pct > 0.001 else 0.001
                calmar = (total_return / 100000.0) / safe_dd
                
                # V12.3: HARD 8% DRAWDOWN LIMIT
                if max_dd_pct > 0.08 or broker.is_blown: return -10000.0
                if trades < 10: return 0.0 
                
                return total_return + (calmar * 100.0)

            study.optimize(objective, n_trials=n_trials)
            
            # --- VALIDATION STEP (OUT-OF-SAMPLE) ---
            best_params = study.best_params
            final_params = CONFIG['online_learning'].copy()
            final_params.update(best_params)
            
            pipeline_inst = ResearchPipeline()
            broker_test = BacktestBroker(initial_balance=CONFIG['env']['initial_balance'])
            model_test = pipeline_inst.get_fresh_model(final_params)
            strategy_test = ResearchStrategy(model_test, symbol, final_params)
            
            for index, row in df_test.iterrows():
                snapshot = MarketSnapshot(timestamp=index, data=row)
                if snapshot.get_price(symbol, 'close') == 0: continue
                broker_test.process_pending(snapshot)
                strategy_test.on_data(snapshot, broker_test)
                
            test_metrics = pipeline_inst.calculate_performance_metrics(broker_test.trade_log, broker_test.initial_balance)
            
            log.info(f"✅ Window {window_id} OOS Result: PnL ${test_metrics['total_pnl']:.0f} | DD {test_metrics['max_dd_pct']*100:.2f}% | Trades {test_metrics['total_trades']}")
            
            wfo_results.append({
                'window': window_id,
                'pnl': test_metrics['total_pnl'],
                'dd': test_metrics['max_dd_pct'],
                'trades': test_metrics['total_trades']
            })
            
            current_train_start += pd.DateOffset(months=test_months)
            
        total_wfo_pnl = sum([r['pnl'] for r in wfo_results])
        log.info(f"🏆 WFO COMPLETE {symbol}: Total OOS PnL ${total_wfo_pnl:.2f}")
        
    except Exception as e:
        log.error(f"WFO Worker Error {symbol}: {e}")

def _worker_finalize_task(symbol: str, train_candles: int, db_url: str, models_dir: Path) -> None:
    """
    Trains the Final Production Model using the Best Params found.
    V12.2 UPDATE: Filters for Statistical Significance (min_trades).
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    setup_logging(f"Worker_Final_{symbol}")
    log = logging.getLogger(f"Worker_Final_{symbol}")
    
    try:
        # Use dynamic train_candles passed from pipeline
        df = process_data_into_bars(symbol, n_ticks=train_candles)
        if df.empty: return

        study_name = f"study_{symbol}"
        study = optuna.load_study(study_name=study_name, storage=db_url)
        
        if len(study.trials) == 0:
            log.warning(f"No trials found for {symbol}. Skipping finalization.")
            return

        # --- V14.0: ROBUST SELECTION LOGIC ---
        # Filter trials that met the trade count threshold
        min_trades = CONFIG['wfo'].get('min_trades_optimization', 100)
        
        valid_trials = [
            t for t in study.trials 
            if t.state == optuna.trial.TrialState.COMPLETE 
            and t.value is not None 
            and t.user_attrs.get('trades', 0) >= min_trades
            and not t.user_attrs.get('blown', False)
        ]
        
        if valid_trials:
            # Select the one with the highest objective value
            best_trial = max(valid_trials, key=lambda t: t.value)
            log.info(f"✅ Selected Robust Trial {best_trial.number} (Trades: {best_trial.user_attrs.get('trades')} >= {min_trades}, Score: {best_trial.value:.2f})")
        else:
            # Fallback (Warning)
            log.warning(f"⚠️ No trials met min_trades={min_trades}. Falling back to absolute best (Risk of Overfitting).")
            best_trial = study.best_trial
            
        best_params = best_trial.params
        
        params_path = models_dir / f"best_params_{symbol}.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=4)

        final_params = CONFIG['online_learning'].copy()
        final_params.update(best_params)
        
        pipeline_inst = ResearchPipeline()
        model = pipeline_inst.get_fresh_model(final_params)
        broker = BacktestBroker(initial_balance=CONFIG['env']['initial_balance'])
        strategy = ResearchStrategy(model, symbol, final_params)

        for index, row in df.iterrows():
            snapshot = MarketSnapshot(timestamp=index, data=row)
            if snapshot.get_price(symbol, 'close') == 0: continue
            
            broker.process_pending(snapshot)
            strategy.on_data(snapshot, broker)

        with open(models_dir / f"river_pipeline_{symbol}.pkl", "wb") as f:
            pickle.dump(strategy.model, f)
        
        with open(models_dir / f"meta_model_{symbol}.pkl", "wb") as f:
            pickle.dump(strategy.meta_labeler, f)
            
        cal_state = {'buy': strategy.calibrator_buy, 'sell': strategy.calibrator_sell}
        with open(models_dir / f"calibrators_{symbol}.pkl", "wb") as f:
            pickle.dump(cal_state, f)
            
        log.info(f"✅ FINALIZED {symbol} | Best Score: {best_trial.value:.4f}")
        gc.collect()
        
    except Exception as e:
        log.error(f"CRITICAL FINALIZE ERROR ({symbol}): {e}")

# --- 3. MAIN PIPELINE CLASS ---

class ResearchPipeline:
    def __init__(self):
        # V12.4: Safe Get for Symbols (Pruned list)
        self.symbols = CONFIG['trading'].get('symbols', [])
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # DYNAMIC CONFIG LOADING FOR MASSIVE DATASETS
        self.train_candles = CONFIG['data'].get('num_candles_train', 10000000)
        self.backtest_candles = CONFIG['data'].get('num_candles_backtest', 1000000)
        
        self.db_url = CONFIG['wfo'].get('db_url', 'sqlite:///optuna.db') 
        
        log_cores = psutil.cpu_count(logical=True)
        self.total_cores = max(1, log_cores - 4) if log_cores else 10

    def get_fresh_model(self, params: Dict[str, Any] = None) -> Any:
        if params is None:
            params = CONFIG['online_learning']
        
        metric_map = {
            "LogLoss": metrics.LogLoss(),
            "F1": metrics.F1(),
            "Accuracy": metrics.Accuracy()
        }
        selected_metric = metric_map.get(params.get('metric', 'LogLoss'), metrics.LogLoss())

        base_clf = forest.ARFClassifier(
            n_models=params.get('n_models', 30),
            seed=42,
            grace_period=params.get('grace_period', 250),
            delta=params.get('delta', 1e-5),
            split_criterion='gini',
            leaf_prediction='mc',
            max_features=params.get('max_features', 'log2'),
            lambda_value=params.get('lambda_value', 10),
            metric=selected_metric,
            warning_detector=drift.ADWIN(delta=params.get('warning_delta', 0.001)),
            drift_detector=drift.ADWIN(delta=params.get('delta', 1e-5))
        )

        return compose.Pipeline(
            preprocessing.StandardScaler(),
            ensemble.ADWINBaggingClassifier(
                model=base_clf,
                n_models=5,
                seed=42
            )
        )

    def calculate_performance_metrics(self, trade_log: List[Dict], initial_capital=100000.0) -> Dict[str, float]:
        metrics_out = {
            'risk_reward_ratio': 0.0,
            'total_pnl': 0.0,
            'max_dd_pct': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'profit_factor': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'sqn': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'dd_events_gt_5': 0
        }
        
        if not trade_log:
            return metrics_out
            
        df = pd.DataFrame(trade_log)
        df['Entry_Time'] = pd.to_datetime(df['Entry_Time'])
        df['Exit_Time'] = pd.to_datetime(df['Exit_Time'])
        df = df.sort_values('Exit_Time') 
        df['Net_PnL'] = pd.to_numeric(df['Net_PnL'], errors='coerce').fillna(0.0)
        
        total_pnl = df['Net_PnL'].sum()
        metrics_out['total_pnl'] = total_pnl
        metrics_out['total_trades'] = len(df)
        
        winners = df[df['Net_PnL'] > 0]
        losers = df[df['Net_PnL'] <= 0]
        metrics_out['win_rate'] = len(winners) / len(df) if len(df) > 0 else 0.0
        
        metrics_out['avg_win'] = winners['Net_PnL'].mean() if not winners.empty else 0.0
        metrics_out['avg_loss'] = losers['Net_PnL'].mean() if not losers.empty else 0.0
        
        avg_loss_abs = abs(metrics_out['avg_loss'])
        if avg_loss_abs > 0:
            metrics_out['risk_reward_ratio'] = metrics_out['avg_win'] / avg_loss_abs
        else:
            if metrics_out['avg_win'] > 0:
                metrics_out['risk_reward_ratio'] = 10.0
            else:
                metrics_out['risk_reward_ratio'] = 0.0

        gross_profit = winners['Net_PnL'].sum()
        gross_loss = abs(losers['Net_PnL'].sum())
        metrics_out['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0.0

        if len(df) > 1:
            pnl_std = df['Net_PnL'].std()
            if pnl_std > 1e-9:
                metrics_out['sqn'] = np.sqrt(len(df)) * (df['Net_PnL'].mean() / pnl_std)

        if initial_capital <= 1000:
             log.warning(f"⚠️ SUSPICIOUS INITIAL CAPITAL: {initial_capital}. Defaulting to 100k.")
             initial_capital = 100000.0

        df['Equity'] = initial_capital + df['Net_PnL'].cumsum()
        df['Peak'] = df['Equity'].cummax()
        df['Peak'] = df['Peak'].clip(lower=initial_capital)
        df['Drawdown_USD'] = df['Equity'] - df['Peak']
        df['Drawdown_Pct'] = (df['Drawdown_USD'] / df['Peak']).abs()
        
        max_dd_pct = df['Drawdown_Pct'].max()
        metrics_out['max_dd_pct'] = max_dd_pct if not pd.isna(max_dd_pct) else 0.0
        
        dd_events = len(df[df['Drawdown_Pct'] > 0.05])
        metrics_out['dd_events_gt_5'] = dd_events

        equity_df = pd.DataFrame({'time': df['Exit_Time'], 'pnl': df['Net_PnL']})
        equity_df.set_index('time', inplace=True)
        hourly_pnl = equity_df['pnl'].resample('1H').sum().fillna(0)
        hourly_equity = initial_capital + hourly_pnl.cumsum()
        hourly_returns = hourly_equity.pct_change().dropna()
        
        if len(hourly_returns) > 1:
            avg_ret = hourly_returns.mean()
            std_ret = hourly_returns.std()
            annual_factor = np.sqrt(252 * 24)
            if std_ret > 1e-9:
                metrics_out['sharpe'] = (avg_ret / std_ret) * annual_factor
            downside_returns = hourly_returns[hourly_returns < 0]
            downside_std = downside_returns.std()
            if downside_std > 1e-9:
                metrics_out['sortino'] = (avg_ret / downside_std) * annual_factor

        return metrics_out

    def _get_sqn_rating(self, sqn: float) -> str:
        if sqn < 1.6: return "POOR 🛑"
        if sqn < 2.0: return "AVERAGE ⚠️"
        if sqn < 2.5: return "GOOD ✅"
        if sqn < 3.0: return "EXCELLENT 🚀"
        if sqn < 5.0: return "SUPERB 💎"
        if sqn < 7.0: return "HOLY GRAIL? 🦄"
        return "GOD MODE ⚡"

    def _purge_models(self):
        """
        Deletes all existing model files to ensure a clean slate for retraining.
        Crucial after logic changes (e.g., Hurst fix).
        """
        # FIX: Replaced LogSymbols.trash with literal emoji to prevent AttributeError
        log.warning(f"🗑️ PURGING OLD MODELS...")
        for p in self.models_dir.glob("*.pkl"):
            try:
                p.unlink()
                log.info(f"Deleted: {p.name}")
            except Exception as e:
                log.error(f"Failed to delete {p.name}: {e}")

    def run_training(self, fresh_start: bool = False):
        # FIX: LogSymbols.training -> literal
        log.info(f"🏋️ STARTING SWARM OPTIMIZATION on {len(self.symbols)} symbols...")
        log.info(f"OBJECTIVE: PROFIT IS KING (Total PnL + Efficiency Tie-Breaker)")
        log.info(f"HARDWARE DETECTED: {psutil.cpu_count(logical=True)} Cores. Using {self.total_cores} workers (Configured).")
        
        # --- V12.33: FRESH START LOGIC ---
        if fresh_start:
            self._purge_models()

        for symbol in self.symbols:
            study_name = f"study_{symbol}"
            if fresh_start:
                print(f"🗑️ ATTEMPTING PURGE: {study_name}...")
                try:
                    optuna.delete_study(study_name=study_name, storage=self.db_url)
                    print(f"✅ PURGED: {study_name}")
                except Exception:
                    pass
            
            try:
                optuna.create_study(study_name=study_name, storage=self.db_url, direction="maximize", load_if_exists=True)
            except Exception as e:
                log.warning(f"Study init warning {symbol}: {e}")

        total_trials_per_symbol = CONFIG['wfo'].get('n_trials', 200)
        tasks = []
        workers_per_symbol = max(1, self.total_cores // len(self.symbols))
        trials_per_worker = math.ceil(total_trials_per_symbol / workers_per_symbol)
        
        log.info(f"DISTRIBUTION: {workers_per_symbol} workers/symbol | {trials_per_worker} trials/worker")
        
        for symbol in self.symbols:
            for _ in range(workers_per_symbol):
                tasks.append((symbol, trials_per_worker, self.train_candles, self.db_url))
        
        start_time = time.time()
        
        Parallel(n_jobs=self.total_cores, backend="loky")(
            delayed(_worker_optimize_task)(*t) for t in tasks
        )
        
        duration = time.time() - start_time
        # FIX: LogSymbols.success -> literal
        log.info(f"✅ Swarm Optimization Complete in {duration:.2f}s")
        log.info(f"💾 Finalizing Models & Artifacts...")
        
        Parallel(n_jobs=len(self.symbols), backend="loky")(
            delayed(_worker_finalize_task)(
                sym,
                self.train_candles,
                self.db_url,
                self.models_dir
            ) for sym in self.symbols
        )
        log.info(f"✅ Training Pipeline Completed.")

    def run_wfo(self):
        log.info(f"{LogSymbols.TIME} STARTING WALK-FORWARD OPTIMIZATION (WFO)...")
        log.info(f"OBJECTIVE: PROFIT IS KING (Rolling Window Validation)")
        
        n_trials = CONFIG['wfo'].get('n_trials', 50)
        
        Parallel(n_jobs=len(self.symbols), backend="loky")(
            delayed(_worker_wfo_task)(
                sym,
                n_trials,
                self.db_url
            ) for sym in self.symbols
        )
        log.info(f"✅ WFO Pipeline Completed.")

    def run_backtest(self):
        # FIX: LogSymbols.backtest -> literal
        log.info(f"📉 Starting BACKTEST verification...")
        
        results = Parallel(n_jobs=len(self.symbols), backend="loky")(
            delayed(self._run_backtest_symbol)(sym) for sym in self.symbols
        )
        
        all_trades = []
        for trades in results:
            all_trades.extend(trades)
            
        self._generate_report(all_trades)

    def _run_backtest_symbol(self, symbol: str) -> List[Dict]:
        try:
            df = process_data_into_bars(symbol, n_ticks=self.train_candles)
            if df.empty: return []

            model_path = self.models_dir / f"river_pipeline_{symbol}.pkl"
            params_path = self.models_dir / f"best_params_{symbol}.json"
            meta_path = self.models_dir / f"meta_model_{symbol}.pkl"
            cal_path = self.models_dir / f"calibrators_{symbol}.pkl"

            if not model_path.exists():
                log.error(f"Model missing for {symbol}")
                return []
                
            with open(model_path, "rb") as f: model = pickle.load(f)
            
            params = CONFIG['online_learning'].copy()
            if params_path.exists():
                with open(params_path, "r") as f: params.update(json.load(f))
            
            strategy = ResearchStrategy(model, symbol, params)
            strategy.debug_mode = True
            
            if meta_path.exists():
                with open(meta_path, "rb") as f: strategy.meta_labeler = pickle.load(f)
            if cal_path.exists():
                with open(cal_path, "rb") as f:
                    cals = pickle.load(f)
                    strategy.calibrator_buy = cals['buy']
                    strategy.calibrator_sell = cals['sell']
            
            broker = BacktestBroker(initial_balance=CONFIG['env']['initial_balance'])
            
            for index, row in df.iterrows():
                snapshot = MarketSnapshot(timestamp=index, data=row)
                if snapshot.get_price(symbol, 'close') > 0:
                    broker.process_pending(snapshot)
                    strategy.on_data(snapshot, broker)
            
            return broker.trade_log
        except Exception as e:
            print(f"Backtest error {symbol}: {e}")
            return []

    def _generate_report(self, trade_log: List[Dict]):
        if not trade_log:
            log.info("No trades executed during backtest.")
            return

        df = pd.DataFrame(trade_log)
        initial_capital = CONFIG['env'].get('initial_balance', 100000.0)
        
        df['Entry_Time'] = pd.to_datetime(df['Entry_Time'])
        df['Exit_Time'] = pd.to_datetime(df['Exit_Time'])
        df = df.sort_values('Exit_Time')
        
        df['Equity'] = initial_capital + df['Net_PnL'].cumsum()
        df['Peak'] = df['Equity'].cummax()
        df['Peak'] = df['Peak'].clip(lower=initial_capital)
        df['Drawdown_USD'] = df['Equity'] - df['Peak']
        df['Drawdown_Pct'] = (df['Drawdown_USD'] / df['Peak']).abs() * 100.0
        
        total_trades = len(df)
        net_pnl = df['Net_PnL'].sum()
        win_count = len(df[df['Net_PnL'] > 0])
        loss_count = len(df[df['Net_PnL'] <= 0])
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0
        
        gross_profit = df[df['Net_PnL'] > 0]['Net_PnL'].sum()
        gross_loss = abs(df[df['Net_PnL'] < 0]['Net_PnL'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        avg_win = df[df['Net_PnL'] > 0]['Net_PnL'].mean() if win_count > 0 else 0.0
        avg_loss = df[df['Net_PnL'] <= 0]['Net_PnL'].mean() if loss_count > 0 else 0.0
        rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        max_dd_pct = df['Drawdown_Pct'].max()
        max_dd_usd = df['Drawdown_USD'].min()
        
        equity_series = pd.Series(df['Net_PnL'].values, index=pd.to_datetime(df['Exit_Time'])).resample('1H').sum().fillna(0)
        hourly_equity = initial_capital + equity_series.cumsum()
        hourly_returns = hourly_equity.pct_change().dropna()
        
        sharpe = 0.0
        if hourly_returns.std() > 1e-9:
            sharpe = (hourly_returns.mean() / hourly_returns.std()) * np.sqrt(252 * 24)

        returns_std = df['Net_PnL'].std()
        sqn = (math.sqrt(total_trades) * (df['Net_PnL'].mean() / returns_std)) if returns_std > 0 else 0.0
        sqn_rating = self._get_sqn_rating(sqn)
        
        # Correctly calculate duration in minutes
        # Ensure Duration_Min exists or calculate it
        if 'Duration_Min' not in df.columns:
             df['Duration_Min'] = (df['Exit_Time'] - df['Entry_Time']).dt.total_seconds() / 60.0
        
        avg_duration = df['Duration_Min'].mean()
        
        log.info("="*60)
        log.info(f"PHOENIX RESEARCH ENGINE - BACKTEST REPORT")
        log.info("="*60)
        log.info(f"{'Metric':<30} | {'Value':<15}")
        log.info("-"*50)
        log.info(f"{'Net Profit':<30} | ${net_pnl:,.2f}")
        log.info(f"{'Initial Capital':<30} | ${initial_capital:,.2f}")
        log.info(f"{'Return %':<30} | {(net_pnl/initial_capital)*100:.2f}%")
        log.info(f"{'Profit Factor':<30} | {profit_factor:.2f}")
        log.info(f"{'Win Rate':<30} | {win_rate:.2f}% ({win_count}/{total_trades})")
        log.info(f"{'Total Trades':<30} | {total_trades}")
        log.info("-"*50)
        log.info(f"{'Max Drawdown':<30} | {max_dd_pct:.2f}% (${abs(max_dd_usd):,.2f})")
        log.info(f"{'Expectancy':<30} | ${expectancy:.2f}")
        log.info(f"{'SQN Score':<30} | {sqn:.2f} ({sqn_rating})")
        log.info(f"{'Sharpe (Hourly)':<30} | {sharpe:.4f}")
        log.info("-"*50)
        log.info(f"{'Avg Win':<30} | ${avg_win:,.2f}")
        log.info(f"{'Avg Loss':<30} | ${avg_loss:,.2f}")
        log.info(f"{'Risk:Reward':<30} | 1:{rr_ratio:.2f}")
        log.info(f"{'Avg Duration':<30} | {avg_duration:.1f} min")
        log.info("="*60)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.reports_dir / f"backtest_trades_{timestamp_str}.csv"
        df.to_csv(csv_path)
        log.info(f"{LogSymbols.DATABASE} Saved Trades to {csv_path}")
        
        try:
            self._plot_equity_curve(df, timestamp_str)
        except Exception as e:
            log.warning(f"Could not generate plot: {e}")

    def _plot_equity_curve(self, df_equity: pd.DataFrame, timestamp_str: str):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3],
                            subplot_titles=("Equity Curve", "Drawdown"))

        fig.add_trace(
            go.Scatter(x=df_equity['Entry_Time'], y=df_equity['Equity'], mode='lines', name='Equity', line=dict(color='#00ff00')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df_equity['Entry_Time'], y=df_equity['Drawdown_Pct'], mode='lines', name='Drawdown %', fill='tozeroy', line=dict(color='#ff0000')),
            row=2, col=1
        )

        fig.update_layout(
            title="Backtest Performance (Aggregate)",
            xaxis_title="Time",
            template="plotly_dark",
            height=800
        )
        
        output_file = self.reports_dir / f"backtest_report_{timestamp_str}.html"
        fig.write_html(output_file)
        log.info(f"✅ HTML Report saved to: {output_file}")

def main():
    optuna.logging.set_verbosity(optuna.logging.WARN)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--fresh-start', action='store_true')
    parser.add_argument('--wfo', action='store_true')
    
    args = parser.parse_args()
    
    pipeline = ResearchPipeline()
    
    if args.wfo:
        pipeline.run_wfo()
    elif args.train:
        pipeline.run_training(fresh_start=args.fresh_start)
    elif args.backtest:
        pipeline.run_backtest()
    else:
        # Default behavior if no flags: Train then Backtest
        log.info("No flags provided. Defaulting to FRESH TRAIN + BACKTEST.")
        pipeline.run_training(fresh_start=True)
        pipeline.run_backtest()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Research process interrupted.")
        sys.exit(0)
    except Exception as e:
        log.critical(f"Research Pipeline Failed: {e}", exc_info=True)
        sys.exit(1)