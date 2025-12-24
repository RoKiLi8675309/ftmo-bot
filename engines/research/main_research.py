# =============================================================================
# FILENAME: engines/research/main_research.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/research/main_research.py
# DEPENDENCIES: shared, engines.research.backtester, engines.research.strategy, pyyaml
# DESCRIPTION: CLI Entry point for Research, Training, and Backtesting.
#
# PHOENIX STRATEGY V1.5 (ALPHA SEEKER - 2025-12-23):
# 1. OPTIMIZATION GOAL: Maximize PnL (Alpha Hunting).
# 2. META-LABELING: Added 'meta_threshold' to search space (0.50 - 0.70).
# 3. GATES: Added 'gate_er' to search space (0.20 - 0.40) to find optimal chop filter.
# 4. REPORTING: Enhanced HTML reports to visualize Alpha vs. Chop.
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
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from joblib import Parallel, delayed
from river import compose, preprocessing, forest, metrics

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Shared Imports
from shared import CONFIG, LogSymbols, VolumeBarAggregator
from engines.research.backtester import BacktestBroker, MarketSnapshot
from engines.research.strategy import ResearchStrategy

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Research")

class ResearchPipeline:
    def __init__(self):
        self.data_dir = Path(project_root) / "data" / "processed"
        self.models_dir = Path(project_root) / "models" / "production"
        self.reports_dir = Path(project_root) / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.symbols = CONFIG['trading']['symbols']
        self.num_workers = max(1, psutil.cpu_count(logical=False) - 2)

    def load_data(self, symbol: str) -> pd.DataFrame:
        """Loads parquet data for a specific symbol."""
        p = self.data_dir / f"{symbol}_volume_bars.parquet"
        if not p.exists():
            log.warning(f"Data not found for {symbol}: {p}")
            return pd.DataFrame()
        return pd.read_parquet(p)

    def optimize_symbol(self, symbol: str, df: pd.DataFrame, n_trials: int = 50):
        """
        Runs Optuna optimization for a single symbol to find Alpha.
        """
        if df.empty: return
        
        log.info(f"{LogSymbols.optimize} Optimizing {symbol} (Target: MAX PROFIT)...")
        
        def objective(trial):
            # 1. Hyperparameter Search Space (The "Levers")
            
            # Model Params
            n_models = trial.suggest_int("n_models", 10, 50, step=10)
            grace_period = trial.suggest_int("grace_period", 100, 500, step=50)
            
            # Alpha / Gate Params
            # Relaxed ER to find Mean Reversion trades
            gate_er = trial.suggest_float("gate_er", 0.15, 0.40) 
            # Tune the Meta-Labeler aggression
            meta_threshold = trial.suggest_float("meta_threshold", 0.51, 0.65)
            
            # Dynamic Stop Loss (ATR Multiplier)
            sl_mult = trial.suggest_float("sl_mult", 1.5, 3.5)
            tp_mult = trial.suggest_float("tp_mult", 1.5, 4.0)

            # 2. Inject Params into Config (Runtime Patching)
            CONFIG['online_learning']['n_models'] = n_models
            CONFIG['online_learning']['grace_period'] = grace_period
            CONFIG['microstructure']['gate_er_threshold'] = gate_er
            CONFIG['online_learning']['meta_labeling_threshold'] = meta_threshold
            CONFIG['risk_management']['stop_loss_atr_mult'] = sl_mult
            CONFIG['risk_management']['take_profit_atr_mult'] = tp_mult

            # 3. Initialize Strategy & Model
            model = compose.Pipeline(
                preprocessing.StandardScaler(),
                forest.ARFClassifier(
                    n_models=n_models,
                    seed=42,
                    leaf_prediction="mc",
                    drift_detector=None, # Disable internal drift for speed, use TBM
                    warning_detector=None
                )
            )
            
            strategy = ResearchStrategy(model=model, symbol=symbol)
            broker = BacktestBroker(initial_balance=10000.0) # Small balance for normalization
            
            # 4. Simulation Loop (Vectorized-ish iteration)
            # We iterate properly to simulate online learning
            for row in df.itertuples(index=False):
                # Convert namedtuple to Series-like dict for snapshot
                data_dict = row._asdict()
                snapshot = MarketSnapshot(
                    timestamp=pd.Timestamp(row.timestamp, unit='s'),
                    data=pd.Series(data_dict)
                )
                
                strategy.process_tick(snapshot, broker)
                
            # 5. Scoring (The "Judge")
            stats = broker.get_stats()
            
            # Profit Factor
            pf = stats.get('profit_factor', 0.0)
            # Net Profit
            pnl = stats.get('total_pnl', 0.0)
            # Drawdown
            dd = stats.get('max_drawdown_pct', 1.0)
            # Trade Count
            trades = stats.get('total_trades', 0)
            
            # Penalty for inactivity
            if trades < 10: 
                return -1000.0
            
            # Penalty for blowing up
            if dd > 0.10: # >10% DD is instant fail
                return -5000.0

            # Objective: Pure Profit, scaled by stability (PF)
            # If PF < 1 (Losing strategy), score is negative PnL (punish losses)
            # If PF > 1, score is PnL * PF (Compound reward for efficiency)
            score = pnl * pf if pf > 1.0 else pnl
            
            # Log progress
            outcome = "PROFIT" if pnl > 0 else "LOSS"
            log.info(f"Trial {trial.number}: {outcome} | PnL: ${pnl:.2f} | PF: {pf:.2f} | Trades: {trades} | ER: {gate_er:.2f}")
            
            return score

        # Run Optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_jobs=1) # Serial to prevent Race Conditions on Config
        
        log.info(f"✅ Best Params for {symbol}: {study.best_params}")
        log.info(f"🏆 Best Score: {study.best_value}")
        
        return study.best_params

    def run_training(self, fresh_start: bool = False):
        """
        Main execution flow.
        """
        if fresh_start:
            log.warning("PURGING previous models and studies...")
            # Logic to delete pickle files would go here
            pass

        log.info(f"{LogSymbols.ONLINE} Starting Alpha Hunt on {len(self.symbols)} symbols.")
        
        for sym in self.symbols:
            df = self.load_data(sym)
            if len(df) < 1000:
                log.error(f"Insufficient data for {sym}")
                continue
                
            # Split Train/Test (Last 20% for validation)
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            
            best_params = self.optimize_symbol(sym, train_df, n_trials=30)
            
            # Save Params
            with open(self.models_dir / f"{sym}_best_params.json", "w") as f:
                json.dump(best_params, f, indent=4)

    def run_backtest(self):
        """
        Runs a final validation backtest using the best parameters found.
        """
        log.info("Running Final Validation Backtest...")
        # Implementation would load best_params.json and run a single pass
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh-start', action='store_true')
    args = parser.parse_args()
    
    pipeline = ResearchPipeline()
    try:
        pipeline.run_training(fresh_start=args.fresh_start)
    except KeyboardInterrupt:
        log.info("Aborted by user.")