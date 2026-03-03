import os
import sys

# --- CRITICAL STABILITY FIX: FORCE SINGLE THREADING FOR MATH LIBS ---
# This must happen before ANY numpy/pandas/scikit-learn imports
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

# =============================================================================
# PHOENIX RESEARCH ENGINE V20.2 – ORDER-FLOW SWING PROTOCOL (PROFITABILITY PATCH)
# =============================================================================

class EmojiCallback:
    """
    Injects FTMO-style Emojis and RICH TELEMETRY into Optuna Trial reporting.
    """
    def __call__(self, study, trial):
        if trial.value is None: return
        val = trial.value
        attrs = trial.user_attrs
        
        symbol = study.study_name.replace("study_", "")
        
        risk_pct = attrs.get('risk_pct', 0.0)
        trades = attrs.get('trades', 0)
        pnl = attrs.get('pnl', 0.0)
        dd = attrs.get('max_dd_pct', 0.0) * 100
        pf = attrs.get('profit_factor', 0.0)
        sqn = attrs.get('sqn', 0.0)
        calmar = attrs.get('calmar', 0.0)
        
        status = "FAIL"
        icon = "🔻"
        
        if attrs.get('blown', False):
            icon = "💀" # Blown Account or Daily Breach
            status = "BLOWN"
        elif dd > 5.0: # FTMO Buffer: 5.0%
            icon = "⚠️" 
            status = "RISKY"
        elif attrs.get('pruned', False):
            icon = "✂️" # Pruned (Low Trades / Sterile)
            status = "PRUNE"
        elif pnl > 2000 and sqn > 1.0:
            icon = "🚀" # High Alpha / Smooth Curve
            status = "ALPHA"
        elif pnl > 0:
            icon = "✅" # Profitable
            status = "PASS"

        wr = attrs.get('win_rate', 0.0) * 100
        
        msg = (
            f"{icon} {status:<6} | {symbol:<6} | Trial {trial.number:<3} | "
            f"R: {risk_pct:>4.3f}% | "
            f"🏆 {val:>8.1f} | "
            f"💰 ${pnl:>9,.0f} | "
            f"📉 {dd:>5.2f}% | "
            f"🎯 {wr:>5.1f}% | "
            f"⚡ PF:{pf:>4.2f} SQN:{sqn:>4.2f} | "
            f"#️⃣ {trades:<4}"
        )
        log.info(msg.strip())
        
        if 'autopsy' in attrs:
            if pnl > 1000.0 and not attrs.get('blown', False):
                log.info(f"\n🏁 VICTORY LAP (Trial {trial.number}): {attrs['autopsy'].strip()}\n" + ("-" * 80))
            elif pnl < 0 or trades == 0 or attrs.get('blown', False):
                log.info(f"\n🔎 FAILURE AUTOPSY (Trial {trial.number}): {attrs['autopsy'].strip()}\n" + ("-" * 80))

def process_data_into_bars(symbol: str, n_ticks: int = 4000000) -> pd.DataFrame:
    raw_ticks = load_real_data(symbol, n_candles=n_ticks, days=730 * 2)
    if raw_ticks.empty: return pd.DataFrame()

    config_threshold = CONFIG['data'].get('volume_bar_threshold', 10.0) 
    alpha = CONFIG['data'].get('imbalance_alpha', 0.05)
    
    first_ts = raw_ticks.index.min()
    cal_cutoff = first_ts + timedelta(days=30)
    cal_df = raw_ticks[raw_ticks.index < cal_cutoff]
    if cal_df.empty: cal_df = raw_ticks
        
    current_thresh = config_threshold
    min_bars = 500
    final_thresh = config_threshold 
    
    for attempt in range(4):
        gen = AdaptiveImbalanceBarGenerator(symbol=symbol, initial_threshold=current_thresh, alpha=alpha)
        count = 0
        for row in cal_df.itertuples():
            p = getattr(row, 'price', getattr(row, 'close', None))
            v = getattr(row, 'volume', 1.0)
            t = getattr(row, 'Index', getattr(row, 'time', None))
            ts_val = t.timestamp() if isinstance(t, (datetime, pd.Timestamp)) else float(t)
            if p is None: continue
            if gen.process_tick(p, v, ts_val, 0.0, 0.0): count += 1
        
        if count >= min_bars:
            final_thresh = current_thresh
            break
        else:
            current_thresh = max(2.0, current_thresh * 0.4)
            final_thresh = current_thresh

    log.info(f"📊 {symbol}: Calibrated Imbalance Threshold to {final_thresh:.2f}")

    gen = AdaptiveImbalanceBarGenerator(symbol=symbol, initial_threshold=final_thresh, alpha=alpha)
    bars = []
    for row in raw_ticks.itertuples():
        p = getattr(row, 'price', getattr(row, 'close', None))
        v = getattr(row, 'volume', 1.0)
        t = getattr(row, 'Index', getattr(row, 'time', None))
        ts_val = t.timestamp() if isinstance(t, (datetime, pd.Timestamp)) else float(t)
        if p is None: continue
        bar = gen.process_tick(p, v, ts_val, 0.0, 0.0)
        if bar:
            bars.append({
                'timestamp': bar.timestamp, 'open': bar.open, 'high': bar.high, 
                'low': bar.low, 'close': bar.close, 'volume': bar.volume,
                'vwap': bar.vwap, 'tick_count': bar.tick_count, 
                'buy_vol': bar.buy_vol, 'sell_vol': bar.sell_vol
            })

    if not bars: return pd.DataFrame()
    df_bars = pd.DataFrame(bars)
    df_bars['time'] = pd.to_datetime(df_bars['timestamp'], unit='s', utc=True)
    df_bars.set_index('time', inplace=True, drop=False)
    return df_bars

# --- 2. WORKER FUNCTIONS ---

def _worker_optimize_task(symbol: str, n_trials: int, train_candles: int, db_url: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    setup_logging(f"Worker_{symbol}")
    optuna.logging.set_verbosity(optuna.logging.WARN)
    
    try:
        df = process_data_into_bars(symbol, n_ticks=train_candles)
        if df.empty: return

        def objective(trial):
            space = CONFIG['optimization_search_space']
            params = CONFIG['online_learning'].copy()
            
            # --- V20.2 CRITICAL FIX: FORCE RISK SCALING FOR WFO ---
            # Even if Live mode uses fixed 0.01 lots, the Optimizer MUST see 
            # actual compounded dollars to calculate PF/SQN accurately.
            params['risk_management'] = CONFIG.get('risk_management', {}).copy()
            params['risk_management']['sizing_method'] = 'risk_percentage'
            
            # 1. Hyperparameter Suggestions
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
                'drift_threshold': trial.suggest_float('drift_threshold', space['tbm_drift_threshold']['min'], space['tbm_drift_threshold']['max']),
                'min_profit_pips': 12.0 
            }
            
            params['min_calibrated_probability'] = trial.suggest_float('min_calibrated_probability', space['min_calibrated_probability']['min'], space['min_calibrated_probability']['max'])
            
            risk_options = CONFIG.get('wfo', {}).get('risk_per_trade_options', [0.0025, 0.0050])
            params['risk_per_trade_percent'] = trial.suggest_categorical('risk_per_trade_percent', risk_options)
            trial.set_user_attr("risk_pct", params['risk_per_trade_percent'] * 100)
            
            pipeline_inst = ResearchPipeline()
            init_bal = CONFIG['env'].get('initial_balance', 50000.0)
            
            broker = BacktestBroker(initial_balance=init_bal)
            model = pipeline_inst.get_fresh_model(params)
            strategy = ResearchStrategy(model, symbol, params)
            
            for index, row in df.iterrows():
                snap = MarketSnapshot(timestamp=index, data=row)
                if snap.get_price(symbol, 'close') == 0: continue
                broker.process_pending(snap)
                strategy.on_data(snap, broker)
                if getattr(broker, 'is_totally_blown', False): break
            
            pm = pipeline_inst.calculate_performance_metrics(broker.trade_log, broker.initial_balance)
            
            total_ret = pm['total_pnl']
            max_dd = pm['max_dd_pct']
            trades = pm['total_trades']
            sqn = pm['sqn']
            pf = pm['profit_factor']
            safe_dd = max_dd if max_dd > 0.001 else 0.001
            calmar = (total_ret / init_bal) / safe_dd

            trial.set_user_attr("autopsy", strategy.generate_autopsy())
            trial.set_user_attr("pnl", total_ret)
            trial.set_user_attr("max_dd_pct", max_dd)
            trial.set_user_attr("win_rate", pm['win_rate'])
            trial.set_user_attr("trades", trades)
            trial.set_user_attr("profit_factor", pf)
            trial.set_user_attr("risk_reward_ratio", pm['risk_reward_ratio'])
            trial.set_user_attr("sqn", sqn)
            trial.set_user_attr("calmar", calmar)

            score = total_ret + (sqn * 1000.0) + (pf * 500.0) + (calmar * 10.0)

            daily_hits = getattr(broker, 'daily_limit_hits', 0)
            total_blown = getattr(broker, 'is_totally_blown', False)
            
            if total_blown or max_dd > 0.08 or daily_hits > 0:
                trial.set_user_attr("blown", True)
                dd_penalty = (max_dd - 0.08) * 500000.0 if max_dd > 0.08 else 0
                daily_penalty = daily_hits * 10000.0
                return -10000.0 + total_ret - dd_penalty - daily_penalty

            # V20.2 SOFTER SIGNIFICANCE FILTER: Lowered penalty to allow AI to explore
            min_trades = 30 # Lowered from 50 to allow medium-frequency trials
            if trades < min_trades:
                trial.set_user_attr("pruned", True)
                trade_shortfall = (min_trades - trades) * 10.0 # Softened penalty so it doesn't crush scores
                return -100.0 + total_ret - trade_shortfall 
                
            return score

        study_name = f"study_{symbol}"
        for _ in range(5):
            try:
                study = optuna.load_study(study_name=study_name, storage=db_url)
                break
            except: time.sleep(1)
        
        study.optimize(objective, n_trials=n_trials, callbacks=[EmojiCallback()])
        del df; gc.collect()
        
    except Exception as e:
        log.error(f"CRITICAL WORKER ERROR ({symbol}): {e}", exc_info=True)

def _worker_wfo_task(symbol: str, n_trials: int, db_url: str):
    os.environ["OMP_NUM_THREADS"] = "1"
    setup_logging(f"WFO_{symbol}")
    optuna.logging.set_verbosity(optuna.logging.WARN)
    
    try:
        train_ticks = CONFIG['data'].get('num_candles_train', 10000000)
        df = process_data_into_bars(symbol, n_ticks=train_ticks)
        if df.empty: return
        
        train_months = CONFIG['wfo'].get('train_years', 2) * 12
        test_months = CONFIG['wfo'].get('test_months', 6)
        
        start_date, end_date = df.index.min(), df.index.max()
        curr_train_start = start_date
        results = []
        
        while True:
            train_end = curr_train_start + pd.DateOffset(months=train_months)
            test_end = train_end + pd.DateOffset(months=test_months)
            if test_end > end_date: break
                
            df_train = df[(df.index >= curr_train_start) & (df.index < train_end)]
            df_test = df[(df.index >= train_end) & (df.index < test_end)]
            if df_train.empty or df_test.empty:
                curr_train_start += pd.DateOffset(months=test_months); continue
                
            win_id = f"{train_end.strftime('%Y-%m')}"
            study_name = f"wfo_{symbol}_{win_id}"
            try: optuna.delete_study(study_name=study_name, storage=f"sqlite:///wfo_{symbol}.db")
            except: pass
            
            study = optuna.create_study(study_name=study_name, storage=f"sqlite:///wfo_{symbol}.db", direction="maximize")
            
            def wfo_objective(trial):
                space = CONFIG['optimization_search_space']
                params = CONFIG['online_learning'].copy()
                
                # FORCE RISK PERCENTAGE FOR WFO
                params['risk_management'] = CONFIG.get('risk_management', {}).copy()
                params['risk_management']['sizing_method'] = 'risk_percentage'
                
                params['n_models'] = trial.suggest_int('n_models', 30, 60)
                params['barrier_width'] = trial.suggest_float('barrier_width', space['tbm_barrier_width']['min'], space['tbm_barrier_width']['max'])
                params['min_calibrated_probability'] = trial.suggest_float('min_calibrated_probability', space['min_calibrated_probability']['min'], space['min_calibrated_probability']['max'])
                params['risk_per_trade_percent'] = trial.suggest_categorical('risk_per_trade_percent', [0.0025, 0.0050])
                
                pipeline = ResearchPipeline()
                broker = BacktestBroker(initial_balance=50000.0)
                model = pipeline.get_fresh_model(params)
                strat = ResearchStrategy(model, symbol, params)
                
                for idx, row in df_train.iterrows():
                    snap = MarketSnapshot(timestamp=idx, data=row)
                    broker.process_pending(snap)
                    strat.on_data(snap, broker)
                    if getattr(broker, 'is_totally_blown', False): break
                
                pm = pipeline.calculate_performance_metrics(broker.trade_log)
                
                if pm['total_trades'] < 5: return -100.0
                return pm['total_pnl']

            study.optimize(wfo_objective, n_trials=n_trials)
            
            final_p = CONFIG['online_learning'].copy()
            final_p.update(study.best_params)
            
            pipe = ResearchPipeline()
            broker_oos = BacktestBroker(initial_balance=50000.0)
            model_oos = pipe.get_fresh_model(final_p)
            strat_oos = ResearchStrategy(model_oos, symbol, final_p)
            
            for idx, row in df_test.iterrows():
                snap = MarketSnapshot(timestamp=idx, data=row)
                broker_oos.process_pending(snap)
                strat_oos.on_data(snap, broker_oos)
            
            oos_m = pipe.calculate_performance_metrics(broker_oos.trade_log)
            results.append({'window': win_id, 'pnl': oos_m['total_pnl']})
            curr_train_start += pd.DateOffset(months=test_months)
            
        log.info(f"🏆 WFO COMPLETE {symbol}: Total OOS PnL ${sum([r['pnl'] for r in results]):.2f}")
        
    except Exception as e:
        log.error(f"WFO Worker Error {symbol}: {e}")

def _worker_finalize_task(symbol: str, train_candles: int, db_url: str, models_dir: Path) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    setup_logging(f"Worker_Final_{symbol}")
    log = logging.getLogger(f"Worker_Final_{symbol}")
    
    try:
        df = process_data_into_bars(symbol, n_ticks=train_candles)
        if df.empty: return

        study_name = f"study_{symbol}"
        study = optuna.load_study(study_name=study_name, storage=db_url)
        
        if not study.trials: return

        min_trades = 30 # V20.2 Matching
        valid = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE 
                 and t.value is not None and t.user_attrs.get('trades', 0) >= min_trades
                 and not t.user_attrs.get('blown', False)]
        
        if valid:
            best_trial = max(valid, key=lambda t: t.value)
            log.info(f"✅ Selected Robust Trial {best_trial.number} (Trades: {best_trial.user_attrs.get('trades')}, Score: {best_trial.value:.2f})")
        else:
            log.warning(f"⚠️ No trials met min_trades={min_trades}. Selecting best available outlier.")
            best_trial = study.best_trial
            
        best_params = best_trial.params
        with open(models_dir / f"best_params_{symbol}.json", "w") as f:
            json.dump(best_params, f, indent=4)

        final_p = CONFIG['online_learning'].copy()
        final_p.update(best_params)
        
        # FORCE RISK PERCENTAGE FOR FINAL VERIFICATION BACKTEST TOO
        final_p['risk_management'] = CONFIG.get('risk_management', {}).copy()
        final_p['risk_management']['sizing_method'] = 'risk_percentage'
        
        pipe = ResearchPipeline()
        model = pipe.get_fresh_model(final_p)
        broker = BacktestBroker(initial_balance=CONFIG['env']['initial_balance'])
        strat = ResearchStrategy(model, symbol, final_p)

        for idx, row in df.iterrows():
            snap = MarketSnapshot(timestamp=idx, data=row)
            broker.process_pending(snap)
            strat.on_data(snap, broker)

        # Persistence
        for name, obj in [("river_pipeline", strat.model), ("meta_model", strat.meta_labeler), 
                          ("calibrators", {'buy': strat.calibrator_buy, 'sell': strat.calibrator_sell})]:
            with open(models_dir / f"{name}_{symbol}.pkl", "wb") as f:
                pickle.dump(obj, f)
            
        log.info(f"✅ FINALIZED {symbol} | Metrics: {pipe.calculate_performance_metrics(broker.trade_log)}")
        gc.collect()
        
    except Exception as e:
        log.error(f"CRITICAL FINALIZE ERROR ({symbol}): {e}")

# --- 3. MAIN PIPELINE CLASS ---

class ResearchPipeline:
    def __init__(self):
        self.symbols = CONFIG['trading'].get('symbols', [])
        self.models_dir = Path("models"); self.models_dir.mkdir(exist_ok=True)
        self.reports_dir = Path("reports"); self.reports_dir.mkdir(exist_ok=True)
        self.train_candles = CONFIG['data'].get('num_candles_train', 10000000)
        self.db_url = CONFIG['wfo'].get('db_url', 'sqlite:///optuna.db') 
        log_cores = psutil.cpu_count(logical=True)
        self.total_cores = max(1, log_cores - 2) if log_cores else 10

    def get_fresh_model(self, params: Dict[str, Any] = None) -> Any:
        if params is None: params = CONFIG['online_learning']
        base_clf = forest.ARFClassifier(
            n_models=params.get('n_models', 50), seed=42,
            grace_period=params.get('grace_period', 200),
            delta=params.get('delta', 1e-5),
            lambda_value=params.get('lambda_value', 10),
            metric=metrics.LogLoss(),
            warning_detector=drift.ADWIN(delta=0.001),
            drift_detector=drift.ADWIN(delta=1e-5)
        )
        return compose.Pipeline(preprocessing.StandardScaler(), base_clf)

    def calculate_performance_metrics(self, trade_log: List[Dict], initial_capital=50000.0) -> Dict[str, float]:
        m = {k: 0.0 for k in ['risk_reward_ratio', 'total_pnl', 'max_dd_pct', 'win_rate', 'total_trades', 'profit_factor', 'sharpe', 'sortino', 'sqn']}
        if not trade_log: return m
        df = pd.DataFrame(trade_log)
        df['Exit_Time'] = pd.to_datetime(df['Exit_Time'])
        df = df.sort_values('Exit_Time')
        df['Net_PnL'] = pd.to_numeric(df['Net_PnL'], errors='coerce').fillna(0.0)
        
        m['total_pnl'] = df['Net_PnL'].sum()
        m['total_trades'] = len(df)
        wins, loss = df[df['Net_PnL'] > 0], df[df['Net_PnL'] <= 0]
        m['win_rate'] = len(wins) / len(df)
        m['profit_factor'] = wins['Net_PnL'].sum() / abs(loss['Net_PnL'].sum()) if not loss.empty else 0.0
        
        df['Equity'] = initial_capital + df['Net_PnL'].cumsum()
        df['Drawdown'] = (df['Equity'] - df['Equity'].cummax()) / df['Equity'].cummax()
        m['max_dd_pct'] = abs(df['Drawdown'].min())
        
        if len(df) > 5:
            m['sqn'] = np.sqrt(len(df)) * (df['Net_PnL'].mean() / df['Net_PnL'].std())
        return m

    def run_training(self, fresh_start: bool = False):
        log.info(f"{LogSymbols.TRAINING} STARTING V20.2 SWARM OPTIMIZATION...")
        if fresh_start:
            for p in self.models_dir.glob("*.pkl"): p.unlink()
            for s in self.symbols:
                try: optuna.delete_study(study_name=f"study_{s}", storage=self.db_url)
                except: pass

        tasks = []
        workers_per_symbol = max(1, self.total_cores // len(self.symbols))
        trials_per_worker = math.ceil(CONFIG['wfo'].get('n_trials', 100) / workers_per_symbol)
        
        for s in self.symbols:
            optuna.create_study(study_name=f"study_{s}", storage=self.db_url, direction="maximize", load_if_exists=True)
            for _ in range(workers_per_symbol):
                tasks.append((s, trials_per_worker, self.train_candles, self.db_url))
        
        start = time.time()
        Parallel(n_jobs=self.total_cores, backend="loky")(delayed(_worker_optimize_task)(*t) for t in tasks)
        log.info(f"{LogSymbols.SUCCESS} Swarm Complete in {time.time()-start:.2f}s. Finalizing...")
        
        Parallel(n_jobs=len(self.symbols), backend="loky")(delayed(_worker_finalize_task)(s, self.train_candles, self.db_url, self.models_dir) for s in self.symbols)

    def run_backtest(self):
        log.info(f"{LogSymbols.BACKTEST} Running Verification Backtest...")
        results = Parallel(n_jobs=len(self.symbols), backend="loky")(delayed(self._run_backtest_symbol)(s) for s in self.symbols)
        all_trades = [t for sub in results for t in sub]
        self._generate_report(all_trades)

    def _run_backtest_symbol(self, s: str) -> List[Dict]:
        try:
            df = process_data_into_bars(s, n_ticks=self.train_candles)
            if df.empty: return []
            with open(self.models_dir / f"river_pipeline_{s}.pkl", "rb") as f: model = pickle.load(f)
            params = CONFIG['online_learning'].copy()
            with open(self.models_dir / f"best_params_{s}.json", "r") as f: params.update(json.load(f))
            
            # FORCE RISK PERCENTAGE FOR FINAL VERIFICATION BACKTEST
            params['risk_management'] = CONFIG.get('risk_management', {}).copy()
            params['risk_management']['sizing_method'] = 'risk_percentage'

            strat = ResearchStrategy(model, s, params)
            broker = BacktestBroker(initial_balance=CONFIG['env']['initial_balance'])
            for idx, row in df.iterrows():
                snap = MarketSnapshot(timestamp=idx, data=row)
                broker.process_pending(snap); strat.on_data(snap, broker)
            return broker.trade_log
        except: return []

    def _generate_report(self, trade_log: List[Dict]):
        if not trade_log: return
        df = pd.DataFrame(trade_log)
        df['Exit_Time'] = pd.to_datetime(df['Exit_Time'])
        df = df.sort_values('Exit_Time')
        df['Equity'] = 50000.0 + df['Net_PnL'].cumsum()
        log.info("="*60 + "\nPHOENIX V20.2 FINAL REPORT\n" + "="*60)
        log.info(f"Total Trades: {len(df)} | Net Profit: ${df['Net_PnL'].sum():,.2f}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(self.reports_dir / f"backtest_trades_{timestamp}.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--fresh-start', action='store_true')
    parser.add_argument('--wfo', action='store_true')
    args = parser.parse_args()
    pipeline = ResearchPipeline()
    if args.wfo: pipeline.run_wfo()
    elif args.train: pipeline.run_training(fresh_start=args.fresh_start)
    elif args.backtest: pipeline.run_backtest()
    else:
        pipeline.run_training(fresh_start=True); pipeline.run_backtest()

if __name__ == "__main__":
    try: main()
    except Exception as e:
        log.critical(f"Research Pipeline Failed: {e}", exc_info=True)
        sys.exit(1)