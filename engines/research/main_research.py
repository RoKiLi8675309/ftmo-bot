# =============================================================================
# FILENAME: engines/research/main_research.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/main_research.py
# DEPENDENCIES: shared, engines.research.backtester, engines.research.strategy, pyyaml
# DESCRIPTION: CLI Entry point for Research, Training, and Backtesting.
#
# FORENSIC REMEDIATION LOG (2025-12-22):
# 1. METRIC SHIFT: Switched ARFClassifier metric to 'Recall' to cure paralysis.
# 2. DATA SCALING: Increased training data load to 2,000,000 ticks (Rec #3).
# 3. REPORTING: Enhanced EmojiCallback to track 'Win Rate' vs 'Risk-Adjusted Return'.
# 4. OPTIMIZATION FIX: Replaced Hard Pruning with Soft Penalties (Gradient Descent).
# =============================================================================
import sys
import os
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
import yaml # Required for Auto-Deployment
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from joblib import Parallel, delayed
from river import compose, preprocessing, forest, metrics

# Ensure project root is in sys.path to resolve 'shared' and 'engines' modules
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
    Uses sys.stdout.write to ensure output visibility in parallel workers.
    """
    def __call__(self, study, trial):
        val = trial.value
        attrs = trial.user_attrs
        
        # Extract Symbol from Study Name (Convention: "study_SYMBOL")
        symbol = study.study_name.replace("study_", "")
        
        # 1. Determine Status Icon & Rank
        sqn = attrs.get('sqn', 0.0)
        trades = attrs.get('trades', 0)
        
        if attrs.get('blown', False):
            icon = "💀"  # Blown Account
            status = "BLOWN"
        elif attrs.get('pruned', False):
            icon = "✂️"  # Pruned (Low Trades)
            status = "PRUNE"
        elif trades == 0:
            icon = "💤"  # No Trades
            status = "IDLE "
        elif val is not None and val >= 3.0 and sqn > 2.0:
            icon = "💎"  # Diamond Hand / Excellent
            status = "ELITE"
        elif val is not None and val >= 1.0:
            icon = "🚀"  # Good
            status = "PROFIT"
        elif val is not None and val > 0.0:
            icon = "🛡️"  # Weak Profit
            status = "WEAK "
        else:
            icon = "🔻"  # Loss
            status = "LOSS "

        # 2. Extract Metrics (Safe Defaults)
        pnl = attrs.get('pnl', 0.0)
        dd = attrs.get('max_dd_pct', 0.0) * 100
        wr = attrs.get('win_rate', 0.0) * 100
        pf = attrs.get('profit_factor', 0.0)
        sharpe = attrs.get('sharpe', 0.0)
        
        # 3. Format Output (Column Aligned)
        # Structure: [ICON] STATUS | SYMBOL | ID | SCORE | PnL | WR | DD | PF | SQN | TRADES
        msg = (
            f"{icon} {status:<6} | {symbol:<6} | Trial {trial.number:<3} | "
            f"🏆 Score: {val:>6.2f} | "
            f"💰 PnL: ${pnl:>9,.2f} | "
            f"🎯 WR: {wr:>5.1f}% | "
            f"📉 DD: {dd:>5.2f}% | "
            f"⚖️ PF: {pf:>4.2f} | "
            f"🧠 SQN: {sqn:>4.2f} | "
            f"⚡ Sharpe: {sharpe:>4.2f} | "
            f"#️⃣ {trades:<4}"
        )
        
        # 4. FORCE PRINT to Console (Bypass Logger Buffering)
        print(msg, flush=True)
        
        # 5. Log to File (for persistence)
        log.info(msg.strip())
        
        # 6. Conditional Autopsy (Debug Info for Failed/Weak Trials)
        if 'autopsy' in attrs and (attrs.get('blown', False) or (val is not None and val < 0.5 and not attrs.get('pruned', False))):
            autopsy_msg = f"\n🔎 AUTOPSY (Trial {trial.number}): {attrs['autopsy'].strip()}\n" + ("-" * 80)
            print(autopsy_msg, flush=True)
            log.info(autopsy_msg)

def process_data_into_bars(symbol: str, n_ticks: int = 2000000) -> pd.DataFrame:
    """
    Helper to Load Ticks -> Aggregate to Volume Bars -> Return Clean DataFrame.
    Strictly enforcing Volume Bars eliminates "Time-Based noise".
    UPDATED: Default n_ticks increased to 2M per recommendation.
    """
    # 1. Load Massive Amount of Ticks (To get sufficient Bars)
    raw_ticks = load_real_data(symbol, n_candles=n_ticks, days=730 * 2) # Double days for safety
    
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
    Executes Optuna Optimization for a single symbol.
    """
    # RESTORED: Optuna INFO Logging for Analysis
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    try:
        # 1. Load & Aggregate Data (Using updated 2M limit)
        df = process_data_into_bars(symbol, n_ticks=2000000) 
        
        if df.empty:
            log.error(f"❌ CRITICAL: No BAR data generated for {symbol}. Aborting worker.")
            return
        
        # Force print to show progress
        print(f"📥 {symbol}: Generated {len(df)} Volume Bars for training.", flush=True)

        # 2. Define Objective
        def objective(trial):
            # Define Search Space (Focused on Tree Structure & Adaptation)
            params = CONFIG['online_learning'].copy()
            params.update({
                'n_models': trial.suggest_int('n_models', 10, 45, step=5),
                'grace_period': trial.suggest_int('grace_period', 20, 100),
                'delta': trial.suggest_float('delta', 0.0001, 0.01, log=True),
                
                # REMEDIATION: Expanded range 0.85 -> 0.99 to allow "Noisy" Trades (FOMO)
                'entropy_threshold': trial.suggest_float('entropy_threshold', 0.85, 0.99),
                # NEW: Allow optimizing VPIN threshold dynamically
                'vpin_threshold': trial.suggest_float('vpin_threshold', 0.75, 0.95),
                
                'tbm': {
                    # REMEDIATION: Relaxed targets for Volatility-Adjusted Environment
                    'barrier_width': trial.suggest_float('barrier_width', 1.5, 3.5),
                    'horizon_minutes': trial.suggest_int('horizon_minutes', 30, 240) 
                },
                # REMEDIATION: Lowered floor to 0.50 to catch weak signals
                'min_calibrated_probability': trial.suggest_float('min_calibrated_probability', 0.50, 0.70)
            })
            
            # Instantiate Pipeline locally
            pipeline_inst = ResearchPipeline()
            broker = BacktestBroker(starting_cash=CONFIG['env']['initial_balance'])
            model = pipeline_inst.get_fresh_model(params)
            strategy = ResearchStrategy(model, symbol, params)
            
            # Run Simulation
            for index, row in df.iterrows():
                snapshot = MarketSnapshot(timestamp=index, data=row)
                if snapshot.get_price(symbol, 'close') == 0: continue
                
                # Check broker pending orders (stops/limits)
                broker.process_pending(snapshot)
                
                # Strategy makes decisions
                strategy.on_data(snapshot, broker)
                
                if broker.is_blown: break
            
            # --- RICH METRICS EXTRACTION ---
            metrics = pipeline_inst.calculate_performance_metrics(broker.trade_log, broker.starting_cash)
            
            # Pass metrics to Callback (Crucial for Emojis)
            trial.set_user_attr("pnl", metrics['total_pnl'])
            trial.set_user_attr("max_dd_pct", metrics['max_dd_pct'])
            trial.set_user_attr("win_rate", metrics['win_rate'])
            trial.set_user_attr("trades", metrics['total_trades'])
            trial.set_user_attr("profit_factor", metrics['profit_factor'])
            trial.set_user_attr("sqn", metrics['sqn'])
            trial.set_user_attr("sharpe", metrics['sharpe'])
            
            # --- CRITICAL FIX: SOFT PENALTY (GRADIENT DESCENT) ---
            # Instead of a hard prune returning -10.0, we apply a scaled penalty.
            # This allows Optuna to see "15 trades" as better than "0 trades".
            
            min_trades = CONFIG['wfo'].get('min_trades_optimization', 5) # Default lowered to 5
            total_trades = metrics['total_trades']
            final_score = metrics['score']
            
            if total_trades < min_trades:
                trial.set_user_attr("pruned", True)
                
                # Calculate Deficit
                deficit = min_trades - total_trades
                
                # Penalty: -0.5 per missing trade.
                # 0 trades = -2.5 penalty (if min is 5).
                # 4 trades = -0.5 penalty.
                penalty = deficit * 0.5
                
                final_score -= penalty
                
                trial.set_user_attr("autopsy", f"LOW ACTIVITY: {total_trades}/{min_trades} trades. Soft Penalty applied: -{penalty:.2f}")
                
                # If 0 trades, ensure score is low enough to discourage but not break gradient
                if total_trades == 0:
                    final_score = -5.0 

                return final_score

            # CRITICAL CONSTRAINT 2: Profitability (Profit Factor)
            # Stop optimizing losers. If active (>50 trades) but losing money, penalize heavily.
            if metrics['total_trades'] > 50 and metrics['profit_factor'] < 0.9:
                trial.set_user_attr("pruned", True)
                trial.set_user_attr("autopsy", f"PRUNED: Active but Losing (PF {metrics['profit_factor']:.2f}).")
                return -5.0 # Moderate penalty, not -10.0

            # Generate Autopsy if result is poor (Debugging)
            if metrics['score'] < 0.5 or broker.is_blown:
                trial.set_user_attr("autopsy", strategy.generate_autopsy())
                
            if broker.is_blown:
                trial.set_user_attr("blown", True)
                return -50.0 # Max Penalty
            
            return final_score

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
        
        # Cleanup memory for this worker
        del df
        gc.collect()

    except Exception as e:
        log.error(f"CRITICAL WORKER ERROR ({symbol}): {e}")
        import traceback
        traceback.print_exc()

def _worker_finalize_task(symbol: str, train_candles: int, db_url: str, models_dir: Path) -> None:
    """
    Trains the Final Production Model using the Best Params found.
    """
    try:
        # 1. Load & Aggregate Data (Using updated 2M limit)
        df = process_data_into_bars(symbol, n_ticks=2000000)
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
        broker = BacktestBroker(starting_cash=CONFIG['env']['initial_balance'])
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
        
        # Increased Tick counts to ensure Bar generation works
        self.train_candles = CONFIG['data'].get('num_candles_train', 2000000)
        self.backtest_candles = CONFIG['data'].get('num_candles_backtest', 500000)
        
        self.db_url = CONFIG['wfo']['db_url']
        self.total_cores = max(1, psutil.cpu_count(logical=True) - 2)
        
        log.info(f"HARDWARE DETECTED: {psutil.cpu_count(logical=True)} Cores. Using {self.total_cores} workers.")

    def get_fresh_model(self, params: Dict[str, Any] = None) -> Any:
        if params is None:
            params = CONFIG['online_learning']
        
        # FORENSIC REMEDIATION: Using ARFClassifier
        # CRITICAL FIX: Changed metric from Precision to Recall to cure Analysis Paralysis
        return compose.Pipeline(
            preprocessing.StandardScaler(),
            forest.ARFClassifier(
                n_models=params.get('n_models', 25),
                seed=42,
                grace_period=params.get('grace_period', 50),
                delta=params.get('delta', 0.001),
                split_criterion='gini',
                leaf_prediction='mc',
                max_features='log2',
                lambda_value=6,
                # Optimization Target: RECALL (Prioritize finding trades over being perfect)
                metric=metrics.Recall() 
            )
        )

    def calculate_performance_metrics(self, trade_log: List[Dict], initial_capital=100000.0) -> Dict[str, float]:
        """
        Calculates extended metrics for robust debugging and scoring.
        """
        metrics = {
            'score': -10.0,
            'total_pnl': 0.0,
            'max_dd_pct': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'profit_factor': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'sqn': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
        
        if not trade_log:
            return metrics

        df = pd.DataFrame(trade_log)
        
        # Basic Stats
        total_pnl = df['Net_PnL'].sum()
        metrics['total_pnl'] = total_pnl
        metrics['total_trades'] = len(df)
        
        winners = df[df['Net_PnL'] > 0]
        losers = df[df['Net_PnL'] <= 0]
        metrics['win_rate'] = len(winners) / len(df) if len(df) > 0 else 0.0
        
        metrics['avg_win'] = winners['Net_PnL'].mean() if not winners.empty else 0.0
        metrics['avg_loss'] = losers['Net_PnL'].mean() if not losers.empty else 0.0

        # Profit Factor
        gross_profit = winners['Net_PnL'].sum()
        gross_loss = abs(losers['Net_PnL'].sum())
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Drawdown
        df['equity'] = initial_capital + df['Net_PnL'].cumsum()
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = df['equity'] - df['peak']
        max_dd_val = df['drawdown'].min()
        metrics['max_dd_pct'] = abs(max_dd_val / initial_capital)

        # Fail Conditions
        if metrics['max_dd_pct'] > 0.10: # 10% Hard Limit
            metrics['score'] = -100.0
            return metrics

        avg_ret = df['Net_PnL'].mean()
        if avg_ret <= 0:
            metrics['score'] = -1.0
            return metrics

        # Volatility Based Metrics
        pnl_std = df['Net_PnL'].std() if len(df) > 1 else 1.0
        downside_std = losers['Net_PnL'].std() if len(losers) > 1 else 1.0
        
        # Sharpe (Annualized approx assuming these are sequential trades in sample)
        metrics['sharpe'] = avg_ret / (pnl_std + 1e-9)
        
        # Sortino
        metrics['sortino'] = avg_ret / (downside_std + 1e-9)
        
        # SQN (System Quality Number)
        if len(df) > 1 and pnl_std > 1e-9:
            metrics['sqn'] = math.sqrt(len(df)) * (avg_ret / pnl_std)
        
        # Scoring Formula (Sortino + Frequency Bonus)
        freq_bonus = np.log10(len(df) + 1)
        metrics['score'] = metrics['sortino'] * freq_bonus
        
        return metrics

    def run_training(self, fresh_start: bool = False):
        log.info(f"{LogSymbols.TIME} STARTING SWARM OPTIMIZATION on {len(self.symbols)} symbols...")
        
        # 1. Init DB Studies
        for symbol in self.symbols:
            study_name = f"study_{symbol}"
            if fresh_start:
                print(f"🗑️  ATTEMPTING PURGE: {study_name}...")
                try:
                    optuna.delete_study(study_name=study_name, storage=self.db_url)
                    print(f"✅ PURGED: {study_name}")
                except Exception:
                    pass
            try:
                optuna.create_study(study_name=study_name, storage=self.db_url, direction="maximize", load_if_exists=True)
            except Exception as e:
                log.warning(f"Study init warning {symbol}: {e}")

        # 2. Distribute Workers
        total_trials_per_symbol = CONFIG['wfo'].get('n_trials', 50)
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
        
        # 3. Finalize
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
            df = process_data_into_bars(symbol, n_ticks=self.backtest_candles)
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
            
            broker = BacktestBroker(starting_cash=CONFIG['env']['initial_balance'])
            
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
        initial_capital = 100000.0 # Assumed start
        
        # Metrics
        total_pnl = df['Net_PnL'].sum()
        win_rate = len(df[df['Net_PnL']>0]) / len(df)
        
        log.info(f"--- BACKTEST RESULTS ---")
        log.info(f"Total PnL: ${total_pnl:,.2f}")
        log.info(f"Win Rate: {win_rate:.1%}")
        log.info(f"Trades: {len(df)}")
        
        # Save CSV
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.reports_dir / f"backtest_trades_{timestamp_str}.csv"
        df.to_csv(csv_path)
        log.info(f"{LogSymbols.DATABASE} Saved Trades to {csv_path}")

        # --- PLOTLY VISUALIZATION ---
        try:
            df['Entry_Time'] = pd.to_datetime(df['Entry_Time'])
            df = df.sort_values('Entry_Time')
            df['Equity'] = initial_capital + df['Net_PnL'].cumsum()
            df['Peak'] = df['Equity'].cummax()
            df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
            
            self._plot_equity_curve(df, timestamp_str)
        except Exception as e:
            log.warning(f"Could not generate plot: {e}")

    def _plot_equity_curve(self, df_equity: pd.DataFrame, timestamp_str: str):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3],
                            subplot_titles=("Equity Curve", "Drawdown"))

        # Equity Line
        fig.add_trace(
            go.Scatter(x=df_equity['Entry_Time'], y=df_equity['Equity'], mode='lines', name='Equity', line=dict(color='#00ff00')),
            row=1, col=1
        )
        
        # Drawdown Area
        fig.add_trace(
            go.Scatter(x=df_equity['Entry_Time'], y=df_equity['Drawdown'], mode='lines', name='Drawdown', fill='tozeroy', line=dict(color='#ff0000')),
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
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
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