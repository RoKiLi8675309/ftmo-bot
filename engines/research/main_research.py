# =============================================================================
# FILENAME: engines/research/main_research.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/main_research.py
# DEPENDENCIES: shared, engines.research.backtester, engines.research.strategy, pyyaml
# DESCRIPTION: CLI Entry point for Research, Training, and Backtesting.
# 
# PHOENIX STRATEGY V7.5 (SNIPER COMPLIANCE):
# 1. PRUNING UPDATE: Lowered trade threshold to support "High Conviction" logic.
#    - Sniper Protocol trades less often; dynamic config threshold ensures validity.
# 2. RISK ALIGNMENT: Synced Drawdown Failure threshold with new 9% limit.
# 3. REPORTING: Preserved Autopsy visibility for forensic analysis.
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
    from shared import CONFIG, setup_logging, load_real_data, LogSymbols, batch_generate_volume_bars
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
            icon = "💀" # Blown Account (>10% DD or Ruin)
            status = "BLOWN"
        elif dd > 9.0: # UPDATED: 9.0% aligned with Config
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
            f"R: {risk_pct:>4.2f}% | "
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
        # UPDATED LOGIC: Always print Autopsy if PnL is negative, trades are 0, or account is blown.
        if 'autopsy' in attrs:
            # Victory Lap for significant profit
            if pnl > 1000.0:
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
    Helper to Load Ticks -> Aggregate to Volume Bars -> Return Clean DataFrame.
    Strictly enforcing Volume Bars eliminates "Time-Based noise".
    """
    # 1. Load Massive Amount of Ticks (To get sufficient Bars)
    raw_ticks = load_real_data(symbol, n_candles=n_ticks, days=730 * 2)
    
    if raw_ticks.empty:
        return pd.DataFrame()

    # 2. Aggregate into Volume Bars (CRITICAL STEP)
    threshold = CONFIG['data'].get('volume_bar_threshold', 1000)
    bars_list = batch_generate_volume_bars(raw_ticks, volume_threshold=threshold)
    
    if not bars_list:
        log.warning(f"⚠️ {symbol}: Not enough volume to generate bars from {len(raw_ticks)} ticks.")
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
    Executes Optuna Optimization for a single symbol using "PROFIT IS KING" Logic.
    """
    # Double-Ensure Single Threading in Worker
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    setup_logging(f"Worker_{symbol}")
    log = logging.getLogger(f"Worker_{symbol}")
    
    # Silence third-party libs in worker (Warning/Error only)
    optuna.logging.set_verbosity(optuna.logging.WARN)
    
    try:
        # 1. Load & Aggregate Data
        df = process_data_into_bars(symbol, n_ticks=4000000)
        
        if df.empty:
            log.error(f"❌ CRITICAL: No BAR data generated for {symbol}. Aborting worker.")
            return
        
        # Log progress (Logger handles output)
        log.info(f"📥 {symbol}: Generated {len(df)} Volume Bars for training.")

        # 2. Define Objective
        def objective(trial):
            # Load Search Space dynamically from CONFIG (No more hardcoding)
            space = CONFIG['optimization_search_space']
            
            # Base Params
            params = CONFIG['online_learning'].copy()
            
            # --- HYPERPARAMETER MAPPING ---
            # River Model Params
            params['n_models'] = trial.suggest_int('n_models', space['n_models']['min'], space['n_models']['max'], step=space['n_models']['step'])
            params['grace_period'] = trial.suggest_int('grace_period', space['grace_period']['min'], space['grace_period']['max'], step=space['grace_period']['step'])
            params['delta'] = trial.suggest_float('delta', float(space['delta']['min']), float(space['delta']['max']), log=space['delta'].get('log', True))
            params['lambda_value'] = trial.suggest_int('lambda_value', space['lambda_value']['min'], space['lambda_value']['max'], step=space['lambda_value']['step'])
            params['max_features'] = trial.suggest_categorical('max_features', ['log2', 'sqrt'])
            
            # Filters & Gates
            params['entropy_threshold'] = trial.suggest_float('entropy_threshold', space['entropy_threshold']['min'], space['entropy_threshold']['max'])
            params['vpin_threshold'] = trial.suggest_float('vpin_threshold', space['vpin_threshold']['min'], space['vpin_threshold']['max'])
            
            # Triple Barrier Method (Labeling)
            params['tbm'] = {
                'barrier_width': trial.suggest_float('barrier_width', space['tbm_barrier_width']['min'], space['tbm_barrier_width']['max']),
                'horizon_minutes': trial.suggest_int('horizon_minutes', space['tbm_horizon_minutes']['min'], space['tbm_horizon_minutes']['max'], step=space['tbm_horizon_minutes']['step']),
                'drift_threshold': trial.suggest_float('drift_threshold', space['tbm_drift_threshold']['min'], space['tbm_drift_threshold']['max'])
            }
            
            # Strategy Execution
            params['min_calibrated_probability'] = trial.suggest_float('min_calibrated_probability', space['min_calibrated_probability']['min'], space['min_calibrated_probability']['max'])
            
            # --- RISK OPTIMIZATION ---
            # Retrieve risk options from WFO config or use default safe aggressive set
            risk_options = CONFIG['wfo'].get('risk_per_trade_options', [0.25, 0.50, 0.75, 1.0])
            params['risk_per_trade_percent'] = trial.suggest_categorical('risk_per_trade_percent', risk_options)
            
            # CAPTURE RISK PERCENT FOR REPORTING
            trial.set_user_attr("risk_pct", params['risk_per_trade_percent'])
            
            # Instantiate Pipeline locally
            pipeline_inst = ResearchPipeline()
            
            # AUDIT FIX: Ensure initial_balance is explicitly set from Config or fallback
            init_bal = CONFIG['env'].get('initial_balance', 100000.0)
            if init_bal <= 0: init_bal = 100000.0
            
            broker = BacktestBroker(initial_balance=init_bal)
            model = pipeline_inst.get_fresh_model(params)
            strategy = ResearchStrategy(model, symbol, params)
            
            # Run Simulation
            for index, row in df.iterrows():
                snapshot = MarketSnapshot(timestamp=index, data=row)
                if snapshot.get_price(symbol, 'close') == 0: continue
                
                broker.process_pending(snapshot)
                strategy.on_data(snapshot, broker)
                
                if broker.is_blown: break
            
            # --- RICH METRICS EXTRACTION ---
            metrics = pipeline_inst.calculate_performance_metrics(broker.trade_log, broker.initial_balance)
            
            # --- SECTION 6.1: "PROFIT IS KING" OBJECTIVE FUNCTION ---
            
            total_return = metrics['total_pnl']
            max_dd_pct = metrics['max_dd_pct'] # Ratio (e.g. 0.05)
            trades = metrics['total_trades']
            
            # Calculate Calmar (Used only for tie-breaking efficiency now)
            safe_dd = max_dd_pct if max_dd_pct > 0.001 else 0.001
            calmar = (total_return / init_bal) / safe_dd

            # --- CRITICAL: GENERATE AUTOPSY BEFORE PRUNING ---
            # This ensures we see WHY a bot took 0 trades (e.g. "Low Confidence" vs "Low Fuel")
            trial.set_user_attr("autopsy", strategy.generate_autopsy())
            
            # Pass metrics to Callback (Set these BEFORE returning failure)
            trial.set_user_attr("pnl", total_return)
            trial.set_user_attr("max_dd_pct", max_dd_pct)
            trial.set_user_attr("win_rate", metrics['win_rate'])
            trial.set_user_attr("trades", trades)
            trial.set_user_attr("profit_factor", metrics['profit_factor'])
            trial.set_user_attr("risk_reward_ratio", metrics['risk_reward_ratio'])
            trial.set_user_attr("sqn", metrics['sqn'])
            trial.set_user_attr("calmar", calmar)

            # 1. IMMEDIATE FAILURE: If DD > 9.0% (Safety Buffer for 10% limit)
            # Penalize HEAVILY (-10,000) to ensure PnL doesn't mask risk
            if max_dd_pct > 0.09: # UPDATED: 9% align with config
                trial.set_user_attr("blown", True)
                return -10000.0 
            
            # 2. BLOWOUT CHECK (Broker level - covers floating PnL blowouts)
            if broker.is_blown:
                trial.set_user_attr("blown", True)
                # If trades were 0 but account blown, it implies massive opening gap or error
                if trades == 0:
                    log.warning(f"⚠️ {symbol} BLOWN with 0 closed trades. Likely floating loss stop-out.")
                return -10000.0

            # 3. STABILITY BONUS: Penalize strategies with < MIN_TRADES (Luck/Sniper)
            # DYNAMIC: Use Configured Minimum (Sniper Mode allows lower count)
            min_trades = CONFIG['wfo'].get('min_trades_optimization', 30)
            if trades < min_trades:
                trial.set_user_attr("pruned", True)
                return 0.0 
                
            # --- NEW OBJECTIVE LOGIC ---
            # PROFIT IS KING.
            # We optimize for Total PnL (Dollars).
            # We add a small efficiency boost (Calmar * 100) to break ties between
            # strategies that make the same money, favoring the smoother one.
            # Example:
            # Strat A: PnL $5000, DD 4% -> Calmar 1.25 -> Score 5125
            # Strat B: PnL $100, DD 0.1% -> Calmar 1.0 -> Score 200
            # This ensures the optimizer hunts for BIG MOVES.
            
            objective_score = total_return + (calmar * 100.0)
            
            return objective_score

        # 3. Connect to Shared Study
        study_name = f"study_{symbol}"
        for _ in range(3):
            try:
                study = optuna.load_study(study_name=study_name, storage=db_url)
                break
            except Exception:
                time.sleep(1)
        
        # 4. Execute Optimization
        study.optimize(objective, n_trials=n_trials, callbacks=[EmojiCallback()])
        
        # Cleanup
        del df
        gc.collect()
        
    except Exception as e:
        log.error(f"CRITICAL WORKER ERROR ({symbol}): {e}")
        import traceback
        traceback.print_exc()

def _worker_finalize_task(symbol: str, train_candles: int, db_url: str, models_dir: Path) -> None:
    """
    Trains the Final Production Model using the Best Params found.
    Also needs logging setup for isolated execution.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    setup_logging(f"Worker_Final_{symbol}")
    log = logging.getLogger(f"Worker_Final_{symbol}")
    
    try:
        # 1. Load & Aggregate Data
        df = process_data_into_bars(symbol, n_ticks=4000000)
        if df.empty: return

        # 2. Get Best Params
        study_name = f"study_{symbol}"
        study = optuna.load_study(study_name=study_name, storage=db_url)
        
        if len(study.trials) == 0:
            log.warning(f"No trials found for {symbol}. Skipping finalization.")
            return
            
        best_params = study.best_params
        
        # 3. Save Best Params
        params_path = models_dir / f"best_params_{symbol}.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=4)

        # 4. Final Training Run
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

        # 5. Save Artifacts
        with open(models_dir / f"river_pipeline_{symbol}.pkl", "wb") as f:
            pickle.dump(strategy.model, f)
        
        # Save MetaLabeler and Calibrators
        with open(models_dir / f"meta_model_{symbol}.pkl", "wb") as f:
            pickle.dump(strategy.meta_labeler, f)
            
        cal_state = {'buy': strategy.calibrator_buy, 'sell': strategy.calibrator_sell}
        with open(models_dir / f"calibrators_{symbol}.pkl", "wb") as f:
            pickle.dump(cal_state, f)
            
        log.info(f"✅ FINALIZED {symbol} | Best Score: {study.best_value:.4f}")
        gc.collect()
        
    except Exception as e:
        log.error(f"CRITICAL FINALIZE ERROR ({symbol}): {e}")

# --- 3. MAIN PIPELINE CLASS ---

class ResearchPipeline:
    def __init__(self):
        self.symbols = CONFIG['trading']['symbols']
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        self.train_candles = CONFIG['data'].get('num_candles_train', 4000000)
        self.backtest_candles = CONFIG['data'].get('num_candles_backtest', 500000)
        
        self.db_url = CONFIG['wfo'].get('db_url', 'sqlite:///optuna.db') # Add default fallback
        
        # AUDIT FIX: Use (Logical - 4) for worker count
        log_cores = psutil.cpu_count(logical=True)
        self.total_cores = max(1, log_cores - 4) if log_cores else 10

    def get_fresh_model(self, params: Dict[str, Any] = None) -> Any:
        if params is None:
            params = CONFIG['online_learning']
        
        # Configure Metric
        metric_map = {
            "LogLoss": metrics.LogLoss(),
            "F1": metrics.F1(),
            "Accuracy": metrics.Accuracy()
        }
        selected_metric = metric_map.get(params.get('metric', 'LogLoss'), metrics.LogLoss())

        # Base Classifier: ARF
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

        # ENSEMBLE: ADWIN Bagging
        return compose.Pipeline(
            preprocessing.StandardScaler(),
            ensemble.ADWINBaggingClassifier(
                model=base_clf,
                n_models=5,
                seed=42
            )
        )

    def calculate_performance_metrics(self, trade_log: List[Dict], initial_capital=100000.0) -> Dict[str, float]:
        """
        Calculates Trade Metrics from the log.
        AUDIT FIX: Computes Risk:Reward Ratio (Avg Win / Avg Loss).
        IMPLEMENTATION UPDATE (Rec 5): Counts drawdown frequency events.
        """
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
        
        # 1. Sort & Cleanup (FIX: Sort by Exit_Time for accurate Equity Curve)
        df['Entry_Time'] = pd.to_datetime(df['Entry_Time'])
        df['Exit_Time'] = pd.to_datetime(df['Exit_Time'])
        df = df.sort_values('Exit_Time') # Changed from Entry_Time
        
        # Ensure PnL is numeric and handle NaNs
        df['Net_PnL'] = pd.to_numeric(df['Net_PnL'], errors='coerce').fillna(0.0)
        
        # 2. Basic Trade Stats
        total_pnl = df['Net_PnL'].sum()
        metrics_out['total_pnl'] = total_pnl
        metrics_out['total_trades'] = len(df)
        
        winners = df[df['Net_PnL'] > 0]
        losers = df[df['Net_PnL'] <= 0]
        metrics_out['win_rate'] = len(winners) / len(df) if len(df) > 0 else 0.0
        
        metrics_out['avg_win'] = winners['Net_PnL'].mean() if not winners.empty else 0.0
        metrics_out['avg_loss'] = losers['Net_PnL'].mean() if not losers.empty else 0.0
        
        # --- RISK REWARD CALCULATION ---
        avg_loss_abs = abs(metrics_out['avg_loss'])
        if avg_loss_abs > 0:
            metrics_out['risk_reward_ratio'] = metrics_out['avg_win'] / avg_loss_abs
        else:
            if metrics_out['avg_win'] > 0:
                metrics_out['risk_reward_ratio'] = 10.0 # Cap for Infinite R:R
            else:
                metrics_out['risk_reward_ratio'] = 0.0
        # -------------------------------

        # Profit Factor
        gross_profit = winners['Net_PnL'].sum()
        gross_loss = abs(losers['Net_PnL'].sum())
        metrics_out['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # SQN Calculation
        if len(df) > 1:
            pnl_std = df['Net_PnL'].std()
            if pnl_std > 1e-9:
                metrics_out['sqn'] = np.sqrt(len(df)) * (df['Net_PnL'].mean() / pnl_std)

        # 3. Time-Series Equity Curve
        
        # FIX: Sanity Check for Initial Capital to prevent 500% DD bugs
        if initial_capital <= 1000:
             log.warning(f"⚠️ SUSPICIOUS INITIAL CAPITAL: {initial_capital}. Defaulting to 100k.")
             initial_capital = 100000.0

        # Calculate running equity to detect Drawdown events
        # CRITICAL FIX: Ensure Peak respects initial capital to prevent fake drawdowns on first loss
        df['Equity'] = initial_capital + df['Net_PnL'].cumsum()
        
        # Peak Calculation needs to consider Initial Capital as the starting high water mark
        df['Peak'] = df['Equity'].cummax()
        df['Peak'] = df['Peak'].clip(lower=initial_capital)
        
        df['Drawdown_USD'] = df['Equity'] - df['Peak']
        
        # --- UNIT FIX: Removed * 100.0 so this is a RATIO (0.05 for 5%) ---
        df['Drawdown_Pct'] = (df['Drawdown_USD'] / df['Peak']).abs()
        
        max_dd_pct = df['Drawdown_Pct'].max()
        metrics_out['max_dd_pct'] = max_dd_pct if not pd.isna(max_dd_pct) else 0.0
        
        # --- REC 5: DRAWDOWN FREQUENCY ---
        # Count number of trade closures where the account was in > 5% drawdown
        # This penalizes lingering in the danger zone
        dd_events = len(df[df['Drawdown_Pct'] > 0.05])
        metrics_out['dd_events_gt_5'] = dd_events

        # 4. Hourly Resampling for Sharpe/Sortino
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
        """Categorizes System Quality Number based on Van Tharp's scale."""
        if sqn < 1.6: return "POOR 🛑"
        if sqn < 2.0: return "AVERAGE ⚠️"
        if sqn < 2.5: return "GOOD ✅"
        if sqn < 3.0: return "EXCELLENT 🚀"
        if sqn < 5.0: return "SUPERB 💎"
        if sqn < 7.0: return "HOLY GRAIL? 🦄"
        return "GOD MODE ⚡"

    def run_training(self, fresh_start: bool = False):
        log.info(f"{LogSymbols.TIME} STARTING SWARM OPTIMIZATION on {len(self.symbols)} symbols...")
        log.info(f"OBJECTIVE: PROFIT IS KING (Total PnL + Efficiency Tie-Breaker)")
        log.info(f"HARDWARE DETECTED: {psutil.cpu_count(logical=True)} Cores. Using {self.total_cores} workers (Configured).")
        
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

        # Distribute Workers
        total_trials_per_symbol = CONFIG['wfo'].get('n_trials', 200)
        tasks = []
        workers_per_symbol = max(1, self.total_cores // len(self.symbols))
        trials_per_worker = math.ceil(total_trials_per_symbol / workers_per_symbol)
        
        log.info(f"DISTRIBUTION: {workers_per_symbol} workers/symbol | {trials_per_worker} trials/worker")
        
        for symbol in self.symbols:
            for _ in range(workers_per_symbol):
                tasks.append((symbol, trials_per_worker, self.train_candles, self.db_url))
        
        start_time = time.time()
        
        # AUDIT FIX: Using 'loky' backend but with restricted OMP threads
        Parallel(n_jobs=self.total_cores, backend="loky")(
            delayed(_worker_optimize_task)(*t) for t in tasks
        )
        
        duration = time.time() - start_time
        log.info(f"{LogSymbols.SUCCESS} Swarm Optimization Complete in {duration:.2f}s")
        log.info(f"{LogSymbols.DATABASE} Finalizing Models & Artifacts...")
        
        Parallel(n_jobs=len(self.symbols), backend="loky")(
            delayed(_worker_finalize_task)(
                sym,
                self.train_candles,
                self.db_url,
                self.models_dir
            ) for sym in self.symbols
        )
        log.info(f"{LogSymbols.SUCCESS} Training Pipeline Completed.")

    def run_backtest(self):
        log.info(f"{LogSymbols.TIME} Starting BACKTEST verification...")
        
        results = Parallel(n_jobs=len(self.symbols), backend="loky")(
            delayed(self._run_backtest_symbol)(sym) for sym in self.symbols
        )
        
        all_trades = []
        for trades in results:
            all_trades.extend(trades)
            
        self._generate_report(all_trades)

    def _run_backtest_symbol(self, symbol: str) -> List[Dict]:
        try:
            # --- CRITICAL DATA ALIGNMENT FIX ---
            # Use self.train_candles (High Fidelity) instead of self.backtest_candles
            # to ensure Volume Bars are identical to training phase.
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
        
        # 1. Equity & Drawdown Calculations
        df['Entry_Time'] = pd.to_datetime(df['Entry_Time'])
        df['Exit_Time'] = pd.to_datetime(df['Exit_Time'])
        df = df.sort_values('Exit_Time') # Match metric calc logic
        
        df['Equity'] = initial_capital + df['Net_PnL'].cumsum()
        
        # CRITICAL FIX for BACKTEST REPORT: ensure Peak is correctly initialized
        df['Peak'] = df['Equity'].cummax()
        df['Peak'] = df['Peak'].clip(lower=initial_capital)
        
        df['Drawdown_USD'] = df['Equity'] - df['Peak']
        df['Drawdown_Pct'] = (df['Drawdown_USD'] / df['Peak']).abs() * 100.0
        
        # 2. Core Metrics
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
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        
        max_dd_pct = df['Drawdown_Pct'].max()
        max_dd_usd = df['Drawdown_USD'].min()
        
        # 3. Advanced Metrics (SQN, Sharpe)
        # Re-using the robust calculation from calculate_performance_metrics
        # Construct hourly equity curve for Sharpe
        equity_series = pd.Series(df['Net_PnL'].values, index=pd.to_datetime(df['Exit_Time'])).resample('1H').sum().fillna(0)
        hourly_equity = initial_capital + equity_series.cumsum()
        hourly_ret = hourly_equity.pct_change().dropna()
        
        sharpe = 0.0
        if hourly_ret.std() > 1e-9:
            sharpe = (hourly_ret.mean() / hourly_ret.std()) * np.sqrt(252 * 24)

        returns_std = df['Net_PnL'].std()
        sqn = (math.sqrt(total_trades) * (df['Net_PnL'].mean() / returns_std)) if returns_std > 0 else 0.0
        sqn_rating = self._get_sqn_rating(sqn)
        
        # 4. Duration
        avg_duration = df['Duration_Min'].mean()
        
        # 5. CONSOLE OUTPUT REPORT
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
        log.info(f"{LogSymbols.SUCCESS} HTML Report saved to: {output_file}")

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
        log.info("WFO Mode not fully implemented in this forensic context. Use --train.")
    elif args.train:
        pipeline.run_training(fresh_start=args.fresh_start)
    elif args.backtest:
        pipeline.run_backtest()
    else:
        pipeline.run_training(fresh_start=args.fresh_start)
        pipeline.run_backtest()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.critical(f"FATAL crash: {e}")
        input("Press Enter to exit...")