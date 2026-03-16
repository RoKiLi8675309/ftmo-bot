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
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from joblib import Parallel, delayed
from river import compose, preprocessing, forest, metrics, drift

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after path fix
try:
    from engines.research.backtester import BacktestBroker, MarketSnapshot
    from engines.research.strategy import ResearchStrategy
    from shared import CONFIG, setup_logging, load_real_data, LogSymbols, RiskManager
    # V20.18 DRY FIX: Import the unified generator to ensure total feature parity with Live Engine
    from shared.financial.features import AdaptiveImbalanceBarGenerator
except ImportError as e:
    print(f"CRITICAL: Failed to import dependencies. Ensure you are running from the project root or 'shared' is accessible.\nError: {e}")
    sys.exit(1)

setup_logging("Research")
log = logging.getLogger("Research")

# =============================================================================
# PHOENIX RESEARCH ENGINE V20.18 – DUAL-MODEL ASYMMETRY PIPELINE
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
        pips = attrs.get('pips', 0.0)
        dd = attrs.get('max_dd_pct', 0.0) * 100
        pf = attrs.get('profit_factor', 0.0)
        sqn = attrs.get('sqn', 0.0)
        
        status = "FAIL"
        icon = "🔻"
        
        if attrs.get('blown', False):
            icon = "💀" # Blown Account or Daily Breach
            status = "BLOWN"
        elif dd > 8.5: 
            icon = "⚠️" 
            status = "RISKY"
        elif attrs.get('pruned', False):
            icon = "✂️" # Pruned (Low Trades / Sterile)
            status = "PRUNE"
        elif pnl > 1000 and sqn > 1.0:
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
            f"💰 Pips {pips:>7.1f} | "
            f"📉 {dd:>5.2f}% | "
            f"🎯 {wr:>5.1f}% | "
            f"⚡ PF:{pf:>4.2f} SQN:{sqn:>4.2f} | "
            f"#️⃣ {trades:<4}"
        )
        log.info(msg.strip())
        
        if 'autopsy' in attrs:
            autopsy_text = attrs['autopsy'].strip()
            if status in ["PASS", "ALPHA"]:
                autopsy_text = autopsy_text.replace("💀", "🏆").replace("FAILURE", "SUCCESS")
                log.info(f"\n✅ SUCCESS AUTOPSY (Trial {trial.number}): {autopsy_text}\n" + ("-" * 80))
            else:
                log.info(f"\n🔎 FAILURE AUTOPSY (Trial {trial.number}): {autopsy_text}\n" + ("-" * 80))

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
        # V20.18 FIX: Using the unified AdaptiveImbalanceBarGenerator
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
            # V20.18 Parity: Slower backoff (x0.5) and higher minimum (10.0)
            current_thresh = max(10.0, current_thresh * 0.5)
            final_thresh = current_thresh

    log.info(f"📊 {symbol}: Calibrated Imbalance Threshold to {final_thresh:.2f} (V20.18 Parity)")

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
            # Convert VolumeBar dataclass back to dict format for dataframe construction
            bars.append({
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

        if len(df) > 500:
            warmup_df = df.head(500)
            eval_df = df.iloc[500:]
        else:
            warmup_df = df.iloc[:0]
            eval_df = df

        def objective(trial):
            space = CONFIG['optimization_search_space']
            params = CONFIG['online_learning'].copy()
            
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
            
            # V20.18 FIX: Meta-Filter tuning & Edge Tuning
            params['meta_labeling_threshold'] = trial.suggest_float('meta_labeling_threshold', space['meta_labeling_threshold']['min'], space['meta_labeling_threshold']['max'])
            params['model_edge_threshold'] = trial.suggest_float('model_edge_threshold', space['model_edge_threshold']['min'], space['model_edge_threshold']['max'])
            
            sl_atr_mult = float(CONFIG.get('risk_management', {}).get('stop_loss_atr_mult', 1.5))
            min_barrier = max(sl_atr_mult * 2.0, float(space['tbm_barrier_width']['min']))
            max_barrier = max(min_barrier + 0.5, float(space['tbm_barrier_width']['max']))
            min_profit_pips = float(CONFIG.get('online_learning', {}).get('tbm', {}).get('min_profit_pips', 40.0))
            
            params['tbm'] = {
                'barrier_width': trial.suggest_float('barrier_width', min_barrier, max_barrier),
                'horizon_minutes': trial.suggest_int('horizon_minutes', space['tbm_horizon_minutes']['min'], space['tbm_horizon_minutes']['max'], step=space['tbm_horizon_minutes']['step']),
                'drift_threshold': trial.suggest_float('drift_threshold', space['tbm_drift_threshold']['min'], space['tbm_drift_threshold']['max']),
                'min_profit_pips': min_profit_pips 
            }
            
            risk_options = CONFIG.get('wfo', {}).get('risk_per_trade_options', [0.0100, 0.0150, 0.0200])
            params['risk_per_trade_percent'] = trial.suggest_categorical('risk_per_trade_percent', risk_options)
            trial.set_user_attr("risk_pct", params['risk_per_trade_percent'] * 100)
            
            pipeline_inst = ResearchPipeline()
            init_bal = CONFIG['env'].get('initial_balance', 50000.0)
            
            broker = BacktestBroker(initial_balance=init_bal)
            
            # V20.18 FIX: model is now a dict containing 'buy' and 'sell' pipelines
            model = pipeline_inst.get_fresh_model(params)
            
            strategy = ResearchStrategy(model, symbol, params, historical_df=warmup_df)
            
            for index, row in eval_df.iterrows():
                snap = MarketSnapshot(timestamp=index, data=row)
                if snap.get_price(symbol, 'close') == 0: continue
                broker.process_pending(snap)
                strategy.on_data(snap, broker)
                if getattr(broker, 'is_totally_blown', False): break
            
            pm = pipeline_inst.calculate_performance_metrics(broker.trade_log, symbol, broker.initial_balance)
            
            total_ret = pm['total_pnl']
            total_pips = pm['total_pips']
            max_dd = pm['max_dd_pct']
            trades = pm['total_trades']
            sqn = pm['sqn']
            pf = pm['profit_factor']

            trial.set_user_attr("autopsy", strategy.generate_autopsy())
            trial.set_user_attr("pnl", total_ret)
            trial.set_user_attr("pips", total_pips)
            trial.set_user_attr("max_dd_pct", max_dd)
            trial.set_user_attr("win_rate", pm['win_rate'])
            trial.set_user_attr("trades", trades)
            trial.set_user_attr("profit_factor", pf)
            trial.set_user_attr("risk_reward_ratio", pm['risk_reward_ratio'])
            trial.set_user_attr("sqn", sqn)

            daily_hits = getattr(broker, 'daily_limit_hits', 0)
            total_blown = getattr(broker, 'is_totally_blown', False)
            
            if total_blown or max_dd > 0.085 or daily_hits > 0:
                trial.set_user_attr("blown", True)
                dd_penalty = (max_dd - 0.085) * 1000000.0 if max_dd > 0.085 else 0
                daily_penalty = daily_hits * 20000.0
                return -20000.0 + total_ret - dd_penalty - daily_penalty

            min_trades = max(5, int(CONFIG.get('wfo', {}).get('min_trades_optimization', 20)))
            if trades < min_trades:
                trial.set_user_attr("pruned", True)
                trade_shortfall = (min_trades - trades) * 500.0 
                return -5000.0 + total_ret - trade_shortfall 
                
            if total_ret <= 0:
                score = total_ret 
            else:
                safe_sqn = max(0.1, sqn)
                safe_pf = max(0.1, min(pf, 10.0))
                volume_multiplier = math.log10(max(10, trades))
                score = total_ret * safe_pf * safe_sqn * volume_multiplier

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
                
                params['risk_management'] = CONFIG.get('risk_management', {}).copy()
                params['risk_management']['sizing_method'] = 'risk_percentage'
                
                params['n_models'] = trial.suggest_int('n_models', space['n_models']['min'], space['n_models']['max'], step=space['n_models']['step'])
                params['grace_period'] = trial.suggest_int('grace_period', space['grace_period']['min'], space['grace_period']['max'], step=space['grace_period']['step'])
                params['delta'] = trial.suggest_float('delta', float(space['delta']['min']), float(space['delta']['max']), log=space['delta'].get('log', True))
                params['lambda_value'] = trial.suggest_int('lambda_value', space['lambda_value']['min'], space['lambda_value']['max'], step=space['lambda_value']['step'])
                params['max_features'] = trial.suggest_categorical('max_features', ['log2', 'sqrt'])
                
                params['entropy_threshold'] = trial.suggest_float('entropy_threshold', space['entropy_threshold']['min'], space['entropy_threshold']['max'])
                params['vpin_threshold'] = trial.suggest_float('vpin_threshold', space['vpin_threshold']['min'], space['vpin_threshold']['max'])
                params['meta_labeling_threshold'] = trial.suggest_float('meta_labeling_threshold', space['meta_labeling_threshold']['min'], space['meta_labeling_threshold']['max'])
                params['model_edge_threshold'] = trial.suggest_float('model_edge_threshold', space['model_edge_threshold']['min'], space['model_edge_threshold']['max'])
                
                sl_atr_mult = float(CONFIG.get('risk_management', {}).get('stop_loss_atr_mult', 1.5))
                min_barrier = max(sl_atr_mult * 2.0, float(space['tbm_barrier_width']['min']))
                max_barrier = max(min_barrier + 0.5, float(space['tbm_barrier_width']['max']))
                min_profit_pips = float(CONFIG.get('online_learning', {}).get('tbm', {}).get('min_profit_pips', 40.0))

                params['tbm'] = {
                    'barrier_width': trial.suggest_float('barrier_width', min_barrier, max_barrier),
                    'horizon_minutes': trial.suggest_int('horizon_minutes', space['tbm_horizon_minutes']['min'], space['tbm_horizon_minutes']['max'], step=space['tbm_horizon_minutes']['step']),
                    'drift_threshold': trial.suggest_float('drift_threshold', space['tbm_drift_threshold']['min'], space['tbm_drift_threshold']['max']),
                    'min_profit_pips': min_profit_pips 
                }
                
                risk_options = CONFIG.get('wfo', {}).get('risk_per_trade_options', [0.0100, 0.0150, 0.0200])
                params['risk_per_trade_percent'] = trial.suggest_categorical('risk_per_trade_percent', risk_options)
                
                pipeline = ResearchPipeline()
                broker = BacktestBroker(initial_balance=50000.0)
                model = pipeline.get_fresh_model(params)
                
                if len(df_train) > 500:
                    train_warmup = df_train.head(500)
                    train_eval = df_train.iloc[500:]
                else:
                    train_warmup = df_train.iloc[:0]
                    train_eval = df_train
                    
                strat = ResearchStrategy(model, symbol, params, historical_df=train_warmup)
                
                for idx, row in train_eval.iterrows():
                    snap = MarketSnapshot(timestamp=idx, data=row)
                    broker.process_pending(snap)
                    strat.on_data(snap, broker)
                    if getattr(broker, 'is_totally_blown', False): break
                
                pm = pipeline.calculate_performance_metrics(broker.trade_log, symbol, broker.initial_balance)
                
                total_ret = pm['total_pnl']
                max_dd = pm['max_dd_pct']
                trades = pm['total_trades']
                sqn = pm['sqn']
                pf = pm['profit_factor']
                
                daily_hits = getattr(broker, 'daily_limit_hits', 0)
                total_blown = getattr(broker, 'is_totally_blown', False)
                
                if total_blown or max_dd > 0.085 or daily_hits > 0:
                    dd_penalty = (max_dd - 0.085) * 1000000.0 if max_dd > 0.085 else 0
                    daily_penalty = daily_hits * 20000.0
                    return -20000.0 + total_ret - dd_penalty - daily_penalty
                
                min_trades = max(5, int(CONFIG.get('wfo', {}).get('min_trades_optimization', 20)))
                if trades < min_trades: 
                    trade_shortfall = (min_trades - trades) * 500.0
                    return -5000.0 + total_ret - trade_shortfall
                    
                if total_ret <= 0:
                    score = total_ret
                else:
                    safe_sqn = max(0.1, sqn)
                    safe_pf = max(0.1, min(pf, 10.0))
                    volume_multiplier = math.log10(max(10, trades))
                    score = total_ret * safe_pf * safe_sqn * volume_multiplier
                return score

            study.optimize(wfo_objective, n_trials=n_trials)
            
            final_p = CONFIG['online_learning'].copy()
            final_p.update(study.best_params)
            
            final_p['risk_management'] = CONFIG.get('risk_management', {}).copy()
            final_p['risk_management']['sizing_method'] = 'risk_percentage'
            if 'tbm' not in final_p: final_p['tbm'] = CONFIG['online_learning'].get('tbm', {}).copy()
            final_p['tbm']['barrier_width'] = study.best_params.get('barrier_width', 3.0)
            final_p['tbm']['min_profit_pips'] = float(CONFIG.get('online_learning', {}).get('tbm', {}).get('min_profit_pips', 40.0))
            if 'horizon_minutes' in study.best_params:
                final_p['tbm']['horizon_minutes'] = study.best_params['horizon_minutes']
            if 'drift_threshold' in study.best_params:
                final_p['tbm']['drift_threshold'] = study.best_params['drift_threshold']
            
            pipe = ResearchPipeline()
            broker_oos = BacktestBroker(initial_balance=50000.0)
            model_oos = pipe.get_fresh_model(final_p)
            
            strat_oos = ResearchStrategy(model_oos, symbol, final_p, historical_df=df_train)
            
            for idx, row in df_test.iterrows():
                snap = MarketSnapshot(timestamp=idx, data=row)
                broker_oos.process_pending(snap)
                strat_oos.on_data(snap, broker_oos)
            
            oos_m = pipe.calculate_performance_metrics(broker_oos.trade_log, symbol, broker_oos.initial_balance)
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

        min_trades = max(5, int(CONFIG.get('wfo', {}).get('min_trades_optimization', 20))) 
        valid = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE 
                 and t.value is not None and t.user_attrs.get('trades', 0) >= min_trades
                 and not t.user_attrs.get('blown', False)]
        
        if valid:
            best_trial = max(valid, key=lambda t: t.value)
            
            pnl = best_trial.user_attrs.get('pnl', 0.0)
            if best_trial.value < 0 or pnl <= 0:
                log.warning(f"🛑 VETO GUARD: Best valid trial for {symbol} lost money (PnL: ${pnl:.2f}). PRUNING PAIR.")
                return
                
            log.info(f"✅ Selected Robust Trial {best_trial.number} (Trades: {best_trial.user_attrs.get('trades')}, Score: {best_trial.value:.2f})")
        else:
            log.warning(f"🛑 VETO GUARD: No trials survived safety constraints for {symbol}. PRUNING PAIR.")
            return
            
        best_params = best_trial.params
        with open(models_dir / f"best_params_{symbol}.json", "w") as f:
            json.dump(best_params, f, indent=4)

        final_p = CONFIG['online_learning'].copy()
        final_p.update(best_params)
        
        final_p['risk_management'] = CONFIG.get('risk_management', {}).copy()
        final_p['risk_management']['sizing_method'] = 'risk_percentage'
        
        if 'tbm' not in final_p: final_p['tbm'] = CONFIG['online_learning'].get('tbm', {}).copy()
        final_p['tbm']['barrier_width'] = best_params.get('barrier_width', 3.0)
        final_p['tbm']['min_profit_pips'] = float(CONFIG.get('online_learning', {}).get('tbm', {}).get('min_profit_pips', 40.0))
        
        if 'horizon_minutes' in best_params:
            final_p['tbm']['horizon_minutes'] = best_params['horizon_minutes']
        if 'drift_threshold' in best_params:
            final_p['tbm']['drift_threshold'] = best_params['drift_threshold']
        if 'meta_labeling_threshold' in best_params:
            final_p['meta_labeling_threshold'] = best_params['meta_labeling_threshold']
        if 'model_edge_threshold' in best_params:
            final_p['model_edge_threshold'] = best_params['model_edge_threshold']
        
        pipe = ResearchPipeline()
        model = pipe.get_fresh_model(final_p)
        broker = BacktestBroker(initial_balance=CONFIG['env'].get('initial_balance', 50000.0))
        
        if len(df) > 500:
            warmup_df = df.head(500)
            eval_df = df.iloc[500:]
        else:
            warmup_df = df.iloc[:0]
            eval_df = df
            
        strat = ResearchStrategy(model, symbol, final_p, historical_df=warmup_df)

        for idx, row in eval_df.iterrows():
            snap = MarketSnapshot(timestamp=idx, data=row)
            broker.process_pending(snap)
            strat.on_data(snap, broker)

        # V20.18 FIX: Save Dual Models independently
        for name, obj in [("river_pipeline_buy", strat.model['buy']), 
                          ("river_pipeline_sell", strat.model['sell']),
                          ("meta_model", strat.meta_labeler), 
                          ("calibrators", {'buy': strat.calibrator_buy, 'sell': strat.calibrator_sell}),
                          ("labeler", strat.labeler)]: 
            with open(models_dir / f"{name}_{symbol}.pkl", "wb") as f:
                pickle.dump(obj, f)
            
        log.info(f"✅ FINALIZED {symbol} | Metrics: {pipe.calculate_performance_metrics(broker.trade_log, symbol, broker.initial_balance)}")
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

    def get_fresh_model(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        V20.18 FIX: Returns a dictionary containing two completely independent models.
        """
        if params is None: params = CONFIG['online_learning']
        
        def create_pipeline():
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
            
        return {
            'buy': create_pipeline(),
            'sell': create_pipeline()
        }

    def calculate_performance_metrics(self, trade_log: List[Dict], symbol: str, initial_capital=50000.0) -> Dict[str, float]:
        m = {k: 0.0 for k in ['risk_reward_ratio', 'total_pnl', 'total_pips', 'max_dd_pct', 'win_rate', 'total_trades', 'profit_factor', 'sharpe', 'sortino', 'sqn', 'expectancy']}
        if not trade_log: return m
        df = pd.DataFrame(trade_log)
        df['Exit_Time'] = pd.to_datetime(df['Exit_Time'])
        df = df.sort_values('Exit_Time')
        df['Net_PnL'] = pd.to_numeric(df['Net_PnL'], errors='coerce').fillna(0.0)
        
        try:
            pip_val, _ = RiskManager.get_pip_info(symbol)
            if pip_val <= 0: pip_val = 0.0001
            
            direction_mult = np.where(df['Action'] == 'BUY', 1.0, -1.0)
            price_diff = (df['Exit_Price'] - df['Entry_Price']) * direction_mult
            df['Pips'] = price_diff / pip_val
            m['total_pips'] = df['Pips'].sum()
        except Exception as e:
            log.warning(f"Failed to calculate pips: {e}")
            m['total_pips'] = 0.0

        m['total_pnl'] = df['Net_PnL'].sum()
        m['total_trades'] = len(df)
        wins, loss = df[df['Net_PnL'] > 0], df[df['Net_PnL'] <= 0]
        m['win_rate'] = len(wins) / len(df) if len(df) > 0 else 0.0
        m['profit_factor'] = wins['Net_PnL'].sum() / abs(loss['Net_PnL'].sum()) if not loss.empty else 0.0
        
        avg_win = wins['Net_PnL'].mean() if len(wins) > 0 else 0.0
        avg_loss = loss['Net_PnL'].mean() if len(loss) > 0 else 0.0
        m['expectancy'] = (m['win_rate'] * avg_win) + ((1 - m['win_rate']) * avg_loss)
        
        df['Equity'] = initial_capital + df['Net_PnL'].cumsum()
        df['Drawdown'] = (df['Equity'] - df['Equity'].cummax()) / df['Equity'].cummax()
        m['max_dd_pct'] = abs(df['Drawdown'].min())
        
        if len(df) > 5:
            std_pnl = df['Net_PnL'].std()
            if std_pnl > 1e-9:
                m['sqn'] = np.sqrt(len(df)) * (df['Net_PnL'].mean() / std_pnl)
            else:
                m['sqn'] = 0.0
                
        return m

    def run_training(self, fresh_start: bool = False):
        log.info(f"{LogSymbols.TRAINING} STARTING V20.18 DUAL-MODEL SWARM OPTIMIZATION...")
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
            
            # V20.18 FIX: Load Dual Models
            buy_model_path = self.models_dir / f"river_pipeline_buy_{s}.pkl"
            sell_model_path = self.models_dir / f"river_pipeline_sell_{s}.pkl"
            if not buy_model_path.exists() or not sell_model_path.exists():
                return []
                
            with open(buy_model_path, "rb") as f: model_buy = pickle.load(f)
            with open(sell_model_path, "rb") as f: model_sell = pickle.load(f)
            model = {'buy': model_buy, 'sell': model_sell}
            
            params = CONFIG['online_learning'].copy()
            with open(self.models_dir / f"best_params_{s}.json", "r") as f: params.update(json.load(f))
            
            params['risk_management'] = CONFIG.get('risk_management', {}).copy()
            params['risk_management']['sizing_method'] = 'risk_percentage' 
            if 'tbm' not in params: params['tbm'] = CONFIG['online_learning'].get('tbm', {}).copy()
            params['tbm']['barrier_width'] = params.get('barrier_width', 3.0)
            params['tbm']['min_profit_pips'] = float(CONFIG.get('online_learning', {}).get('tbm', {}).get('min_profit_pips', 40.0))

            if len(df) > 500:
                warmup_df = df.head(500)
                eval_df = df.iloc[500:]
            else:
                warmup_df = df.iloc[:0]
                eval_df = df
                
            strat = ResearchStrategy(model, s, params, historical_df=warmup_df)
            broker = BacktestBroker(initial_balance=CONFIG['env'].get('initial_balance', 50000.0))
            
            for idx, row in eval_df.iterrows():
                snap = MarketSnapshot(timestamp=idx, data=row)
                broker.process_pending(snap); strat.on_data(snap, broker)
            return broker.trade_log
        except: return []

    def _generate_report(self, trade_log: List[Dict]):
        """
        V20.18: Generates a beautiful, FTMO-centric console Tearsheet.
        """
        if not trade_log:
            log.warning("No trades generated during backtest. Report aborted.")
            return
            
        df = pd.DataFrame(trade_log)
        df['Exit_Time'] = pd.to_datetime(df['Exit_Time'])
        df = df.sort_values('Exit_Time')
        
        initial_balance = CONFIG['env'].get('initial_balance', 50000.0)
        df['Equity'] = initial_balance + df['Net_PnL'].cumsum()
        df['Drawdown'] = (df['Equity'] - df['Equity'].cummax()) / df['Equity'].cummax()
        
        total_pnl = df['Net_PnL'].sum()
        total_trades = len(df)
        win_rate = len(df[df['Net_PnL'] > 0]) / total_trades if total_trades > 0 else 0.0
        max_dd_pct = abs(df['Drawdown'].min()) * 100.0
        
        gross_profit = df[df['Net_PnL'] > 0]['Net_PnL'].sum()
        gross_loss = abs(df[df['Net_PnL'] <= 0]['Net_PnL'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = df[df['Net_PnL'] > 0]['Net_PnL'].mean() if len(df[df['Net_PnL'] > 0]) > 0 else 0.0
        avg_loss = df[df['Net_PnL'] <= 0]['Net_PnL'].mean() if len(df[df['Net_PnL'] <= 0]) > 0 else 0.0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # FTMO Logic Checks
        target_profit = initial_balance * 0.10
        max_loss_limit = 10.0 # 10%
        
        passed = (total_pnl >= target_profit) and (max_dd_pct < max_loss_limit)
        status_text = "✅ PASSED" if passed else "❌ FAILED"
        
        report = f"""
        =================================================================
        🦅 PROJECT PHOENIX V20.18 - FTMO TEARSHEET
        =================================================================
        Initial Balance:      ${initial_balance:,.2f}
        Final Equity:         ${initial_balance + total_pnl:,.2f}
        Net Profit:           ${total_pnl:,.2f} ({(total_pnl/initial_balance)*100:.2f}%)
        
        FTMO Target (10%):    ${target_profit:,.2f}
        Max Drawdown:         {max_dd_pct:.2f}% (Limit: 10.00%)
        FTMO Challenge:       {status_text}
        
        --- EXECUTION METRICS ---
        Total Trades:         {total_trades}
        Win Rate:             {win_rate*100:.2f}%
        Profit Factor:        {profit_factor:.2f}
        Expectancy per Trade: ${expectancy:.2f}
        Average Win:          ${avg_win:.2f}
        Average Loss:         ${avg_loss:.2f}
        =================================================================
        """
        
        print(report)
        log.info("FTMO Tearsheet generated.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(self.reports_dir / f"backtest_trades_{timestamp}.csv", index=False)
        log.info(f"Trades exported to reports/backtest_trades_{timestamp}.csv")

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