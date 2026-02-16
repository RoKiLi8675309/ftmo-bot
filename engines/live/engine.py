import logging
import time
import json
import threading
import signal
import sys
import os
import multiprocessing
import queue
import numpy as np
import math
import pytz
from datetime import datetime, timedelta, date, timezone
from collections import defaultdict, deque
from typing import Any, Optional, Dict, List, Set

# Third-Party NLP (Guarded)
try:
    from newspaper import Article
    from textblob import TextBlob
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    # print("WARNING: NLP libraries not found (newspaper3k, textblob). Sentiment features disabled.")

# Shared Imports
from shared import (
    CONFIG,
    setup_logging,
    RedisStreamManager,
    LogSymbols,
    FTMORiskMonitor,
    PortfolioRiskManager,
    SessionGuard,
    FTMOComplianceGuard,
    NewsEventMonitor,
    AdaptiveImbalanceBarGenerator, 
    VolumeBar,
    Trade,
    RiskManager,
    TradeContext,
    load_real_data 
)

# Local Engine Modules
from engines.live.dispatcher import TradeDispatcher
from engines.live.predictor import MultiAssetPredictor, Signal

setup_logging("LiveEngine")
logger = logging.getLogger("LiveEngine")

# CONSTANTS
MAX_TICK_LATENCY_SEC = 45.0 

# --- WORKER PROCESS FOR OFF-MAIN-THREAD COMPUTE ---
class LogicWorker(multiprocessing.Process):
    """
    Rec 1: Asynchronous Parallelism.
    Handles Feature Engineering, Bar Aggregation, and River Model Inference
    in a dedicated CPU process to prevent blocking the Redis Consumer.
    """
    def __init__(self, tick_queue: multiprocessing.Queue, signal_queue: multiprocessing.Queue, 
                 config: Dict, symbols: List[str]):
        super().__init__()
        self.tick_queue = tick_queue
        self.signal_queue = signal_queue
        self.config = config
        self.symbols = symbols
        self.running = True
        
        # Latency Params (Rec 2)
        self.max_latency_strict = 2.0
        self.max_latency_relaxed = 30.0
        self.vol_threshold = 0.0005 # ~5 pips volatility triggers strict mode

    def run(self):
        """
        Main Loop for the Logic Process.
        Owns the Predictor and Aggregators to ensure state isolation.
        """
        # Re-setup logging for this process
        setup_logging("LogicWorker")
        worker_log = logging.getLogger("LogicWorker")
        worker_log.info(f"{LogSymbols.ONLINE} Logic Worker Started (PID: {os.getpid()})")
        
        # 1. Initialize State (Local to Process)
        # We must calibrate/init aggregators here to avoid pickling issues across processes
        threshold_map = self._calibrate_thresholds(worker_log)
        
        aggregators = {}
        alpha = self.config['data'].get('imbalance_alpha', 0.05)
        
        for sym in self.symbols:
            # Use calibrated threshold or default
            thresh = threshold_map.get(sym, self.config['data'].get('volume_bar_threshold', 10.0))
            aggregators[sym] = AdaptiveImbalanceBarGenerator(
                symbol=sym,
                initial_threshold=thresh,
                alpha=alpha
            )
            
        # Initialize Predictor (Heavy Lift)
        # Rec 3: Predictor internally handles Volatility-Based Horizons via its config/labeler logic
        predictor = MultiAssetPredictor(self.symbols, threshold_map=threshold_map)
        
        # Context Cache (D1/H4 updates passed from Main)
        latest_context = defaultdict(dict)
        
        while self.running:
            try:
                # 1. Fetch Tick (Blocking with timeout for graceful shutdown)
                try:
                    tick_data = self.tick_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if tick_data == "STOP":
                    worker_log.info("Logic Worker received STOP signal.")
                    break

                symbol = tick_data.get('symbol')
                if symbol not in self.symbols: continue

                # 2. Hardened Latency Guard (Rec 2)
                # We check latency HERE, using the locally maintained Volatility state
                if not self._check_latency(tick_data, predictor, worker_log):
                    continue

                # 3. Update Context (if present in payload)
                if 'ctx_d1' in tick_data:
                    try: latest_context[symbol]['d1'] = json.loads(tick_data['ctx_d1'])
                    except: pass
                if 'ctx_h4' in tick_data:
                    try: latest_context[symbol]['h4'] = json.loads(tick_data['ctx_h4'])
                    except: pass

                # 4. Process Tick -> Bar
                # Parse data
                try:
                    price = float(tick_data.get('price', 0.0))
                    volume = float(tick_data.get('volume', 1.0))
                    ts = float(tick_data.get('time', 0.0))
                    if ts > 100_000_000_000: ts /= 1000.0 # Normalize ms
                    
                    bid_vol = float(tick_data.get('bid_vol', 0.0))
                    ask_vol = float(tick_data.get('ask_vol', 0.0))
                except (ValueError, TypeError):
                    continue

                # Aggregate
                bar = aggregators[symbol].process_tick(
                    price=price,
                    volume=volume,
                    timestamp=ts,
                    external_buy_vol=bid_vol,
                    external_sell_vol=ask_vol
                )

                # 5. Predict (if Bar closed)
                if bar:
                    # Prepare context
                    ctx = {
                        'd1': latest_context[symbol].get('d1', {}),
                        'h4': latest_context[symbol].get('h4', {}),
                        'positions': tick_data.get('positions', {}) # Passed from Main thread
                    }
                    
                    # Generate Signal
                    signal = predictor.process_bar(symbol, bar, context_data=ctx)
                    
                    if signal:
                        # Push to Execution Queue for Main Thread Dispatcher
                        self.signal_queue.put(signal)
                        worker_log.info(f"{LogSymbols.SIGNAL} Generated Signal: {signal.action} {symbol} ({signal.confidence:.2f})")

            except Exception as e:
                worker_log.error(f"Logic Worker Error: {e}", exc_info=True)
        
        # Cleanup
        predictor.save_state()
        worker_log.info("Logic Worker Shutdown Complete.")

    def _check_latency(self, tick_data: Dict, predictor: MultiAssetPredictor, log: logging.Logger) -> bool:
        """
        Rec 2: Dynamic Latency Guard.
        Checks volatility to determine acceptable latency.
        """
        try:
            ts = float(tick_data.get('time', 0.0))
            if ts > 100_000_000_000: ts /= 1000.0
            
            now = time.time()
            latency = abs(now - ts)
            symbol = tick_data.get('symbol')
            
            # Query Volatility Monitor (Thread-safe read from local predictor instance)
            current_vol = 0.001
            if symbol in predictor.feature_engineers:
                # Direct access to the monitor's getter
                current_vol = predictor.feature_engineers[symbol].vol_monitor.get()
            
            # Dynamic Limit: Strict if Volatility is High, Relaxed if Low
            limit = self.max_latency_strict if current_vol > self.vol_threshold else self.max_latency_relaxed
            
            if latency > limit:
                # Log only occasional warnings to avoid spam during lags
                if latency > 10.0:
                    log.warning(f"⚠️ Latency Guard: Dropped {symbol} (Lat: {latency:.2f}s > Limit: {limit:.1f}s | Vol: {current_vol:.5f})")
                return False
                
            return True
        except Exception:
            return True # Fail open if check fails

    def _calibrate_thresholds(self, log: logging.Logger) -> Dict[str, float]:
        """
        Internal calibration logic for the worker process.
        Reproduces the logic from the original Engine but inside the worker.
        """
        log.info(f"{LogSymbols.TRAINING} Logic Worker: Auto-Calibrating Thresholds...")
        threshold_map = {}
        alpha = self.config['data'].get('imbalance_alpha', 0.05)
        config_thresh = self.config['data'].get('volume_bar_threshold', 10.0)
        
        for sym in self.symbols:
            try:
                # Load Data (Last 30 Days)
                df = load_real_data(sym, n_candles=50000, days=30)
                
                if df is None or df.empty:
                    threshold_map[sym] = config_thresh
                    continue
                
                current_threshold = config_thresh
                attempts = 0
                max_attempts = 4
                
                while attempts < max_attempts:
                    gen = AdaptiveImbalanceBarGenerator(sym, initial_threshold=current_threshold, alpha=alpha)
                    bar_count = 0
                    
                    for row in df.itertuples():
                        price = getattr(row, 'price', getattr(row, 'close', None))
                        vol = getattr(row, 'volume', 1.0)
                        ts_val = getattr(row, 'Index', None).timestamp()
                        if price is None: continue
                        
                        b_vol = vol / 2
                        s_vol = vol / 2
                        if gen.process_tick(price, vol, ts_val, b_vol, s_vol):
                            bar_count += 1
                    
                    if bar_count >= 500:
                        threshold_map[sym] = current_threshold
                        break
                    else:
                        attempts += 1
                        current_threshold = max(5.0, current_threshold * 0.5)
                        threshold_map[sym] = current_threshold
                
                log.info(f"✅ {sym} Worker Calibrated: {threshold_map[sym]:.1f}")
            except Exception as e:
                log.error(f"Worker Calibration Error {sym}: {e}")
                threshold_map[sym] = config_thresh
        
        return threshold_map


class LiveTradingEngine:
    """
    The Central Orchestrator.
    1. Consumes Ticks from Redis (IO Bound).
    2. Feeds LogicWorker via Queue (Compute Bound).
    3. Consumes Signals from LogicWorker (Execution Bound).
    4. Dispatches Orders to Windows via Redis.
    """
    def __init__(self):
        self.shutdown_flag = False
        self.symbols = CONFIG['trading']['symbols']
        
        # 1. Infrastructure
        self.stream_mgr = RedisStreamManager(
            host=CONFIG['redis']['host'],
            port=CONFIG['redis']['port'],
            db=0
        )
        self.stream_mgr.ensure_group()

        # 2. Risk & Compliance
        initial_bal = CONFIG['env'].get('initial_balance', 100000.0)
        self.ftmo_guard = FTMORiskMonitor(
            initial_balance=initial_bal,
            max_daily_loss_pct=CONFIG['risk_management']['max_daily_loss_pct'],
            redis_client=self.stream_mgr.r
        )
        self.portfolio_mgr = PortfolioRiskManager(symbols=self.symbols)
        self.session_guard = SessionGuard()
        
        # --- COMPLIANCE SETUP ---
        self.news_monitor = NewsEventMonitor()
        # Initialize Guard with empty events first to avoid blocking startup
        self.compliance_guard = FTMOComplianceGuard([])
        
        # Start Background Calendar Sync
        self.calendar_thread = threading.Thread(target=self._calendar_sync_loop, daemon=True)
        self.calendar_thread.start()
        
        # 3. Execution Dispatcher
        self.pending_orders = {}  # Track orders to prevent dupes
        self.lock = threading.RLock()
        self.dispatcher = TradeDispatcher(
            stream_mgr=self.stream_mgr,
            pending_tracker=self.pending_orders,
            lock=self.lock
        )
        
        # EXECUTION THROTTLE STATE
        self.last_dispatch_time = defaultdict(float)
        
        # 4. Logic Worker Setup (Rec 1: Async Parallelism)
        # Use multiprocessing Queues to communicate with the worker process
        self.tick_queue = multiprocessing.Queue(maxsize=10000)
        self.signal_queue = multiprocessing.Queue(maxsize=1000)
        
        self.logic_worker = LogicWorker(
            tick_queue=self.tick_queue,
            signal_queue=self.signal_queue,
            config=CONFIG,
            symbols=self.symbols
        )
        
        # 5. Sentiment Engine
        self.global_sentiment = {} # {symbol: score} or {'GLOBAL': score}
        if NLP_AVAILABLE:
            self.news_thread = threading.Thread(target=self.fetch_news_loop, daemon=True)
            self.news_thread.start()

        # 6. Performance Monitor (Circuit Breaker & SQN)
        # THREAD SAFETY: Lock for statistics updated by background thread
        self.stats_lock = threading.Lock()

        # Tracks last 30 trades per symbol to calculate SQN
        self.performance_stats = defaultdict(lambda: deque(maxlen=30))
        
        # Server-Authority Stats
        # We store the "last reset timestamp" to detect day changes
        self.last_reset_ts = 0.0 
        
        # Daily Circuit Breaker State (Realized Execution)
        self.daily_execution_stats = defaultdict(lambda: {'losses': 0, 'pnl': 0.0, 'tickets': set()})
        
        # Increased to 5 to prevent early lockout in Aggressor Mode
        self.max_daily_losses_per_symbol = 5
        
        # --- TIMEZONE INIT ---
        tz_str = CONFIG['risk_management'].get('risk_timezone', 'Europe/Prague')
        try:
            self.server_tz = pytz.timezone(tz_str)
        except Exception:
            self.server_tz = pytz.timezone('Europe/Prague')

        # --- STATE RESTORATION (CRITICAL) ---
        self._restore_daily_state()
        # Restore Penalty Box to prevent Revenge Trading after restart
        self._restore_penalty_box_state()

        self.perf_thread = threading.Thread(target=self.fetch_performance_loop, daemon=True)
        self.perf_thread.start()
        
        # NEW: Signal Listener Thread (Rec 1)
        self.signal_thread = threading.Thread(target=self._signal_listener, daemon=True)
        self.signal_thread.start()

        # State
        self.is_warm = False
        self.last_corr_update = time.time()
        self.latest_prices = {}
        
        # Position Cache (for passing to LogicWorker)
        self.latest_positions = {}
        
        # --- INJECT FALLBACK PRICES ---
        self._inject_fallback_prices()
        
        self.liquidation_triggered_map = {sym: False for sym in CONFIG['trading']['symbols']}
        
        # --- TICK DEDUPLICATION & HEARTBEAT STATE ---
        self.processed_ticks = defaultdict(float)
        self.ticks_processed = 0 # Visual Heartbeat Counter

        # --- CONTEXT CACHE (D1/H4 from Windows) ---
        self.latest_context = defaultdict(dict) # {symbol: {'d1': {}, 'h4': {}}}

        # --- Volatility Gate Config ---
        self.vol_gate_conf = CONFIG['online_learning'].get('volatility_gate', {})
        self.use_vol_gate = self.vol_gate_conf.get('enabled', True)
        self.min_atr_spread_ratio = self.vol_gate_conf.get('min_atr_spread_ratio', 1.0) 
        self.spread_map = CONFIG.get('forensic_audit', {}).get('spread_pips', {})

        # --- SAFETY GAP STATE ---
        self.known_open_symbols: Set[str] = set()

        # Active Position Management Thread
        self.mgmt_thread = threading.Thread(target=self._manage_active_positions_loop, daemon=True)
        self.mgmt_thread.start()

    def _inject_fallback_prices(self):
        """
        Injects static fallback prices for conversion pairs to prevent
        'No Rate' errors before the first tick arrives.
        Updated for Golden Basket V16.30.
        """
        defaults = {
            # --- MAJORS (Primary Conversion) ---
            "USDJPY": 150.0, "GBPUSD": 1.25, "EURUSD": 1.08,
            "USDCAD": 1.35, "USDCHF": 0.90, "AUDUSD": 0.65, "NZDUSD": 0.60,
            # --- GOLDEN BASKET CROSSES (Fallbacks) ---
            "GBPJPY": 190.0, "EURJPY": 160.0, "AUDJPY": 95.0,
            # --- LEGACY (Low Priority) ---
            "GBPAUD": 1.95, "EURAUD": 1.65, "GBPNZD": 2.05
        }
        for sym, price in defaults.items():
            if sym not in self.latest_prices:
                self.latest_prices[sym] = price
        
        logger.info(f"💉 Injected {len(defaults)} fallback prices for PnL conversion.")

    def _get_producer_anchor_timestamp(self) -> float:
        """
        Polls Redis for the AUTHORITATIVE Midnight Timestamp set by Windows Producer.
        """
        try:
            # This key is set by windows_producer.py during midnight rollover
            reset_ts = self.stream_mgr.r.get("risk:last_reset_date")
            if reset_ts:
                return float(reset_ts)
        except:
            pass
        
        # Fallback: Midnight UTC today
        return datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    def _restore_daily_state(self):
        """
        Replays today's closed trades from Redis using the Producer's Timestamp Authority.
        """
        logger.info(f"{LogSymbols.DATABASE} Restoring Daily Execution State (Server Time Sync)...")
        try:
            stream_key = CONFIG['redis'].get('closed_trade_stream_key', 'stream:closed_trades')
            
            # 1. Get Authoritative Start Time
            self.last_reset_ts = self._get_producer_anchor_timestamp()
            
            # Fetch range from the Anchor point
            start_id = int(self.last_reset_ts * 1000)
            messages = self.stream_mgr.r.xrange(stream_key, min=start_id, max='+')
            
            restored_count = 0
            with self.stats_lock:
                # Clear stats on restore to ensure clean slate
                self.daily_execution_stats = defaultdict(lambda: {'losses': 0, 'pnl': 0.0, 'tickets': set()})
                
                for msg_id, data in messages:
                    symbol = data.get('symbol')
                    net_pnl = float(data.get('net_pnl', 0.0))
                    timestamp_raw = float(data.get('timestamp', 0))
                    ticket = str(data.get('ticket', '')) 
                    
                    # STRICT FILTER: Only count trades after the anchor
                    if timestamp_raw < self.last_reset_ts:
                        continue
                        
                    # Apply Logic
                    self.daily_execution_stats[symbol]['pnl'] += net_pnl
                    
                    # Deduplication
                    if ticket and ticket in self.daily_execution_stats[symbol]['tickets']:
                        continue
                    
                    if ticket:
                        self.daily_execution_stats[symbol]['tickets'].add(ticket)
                    
                    if net_pnl < 0:
                        self.daily_execution_stats[symbol]['losses'] += 1
                    
                    # Also restore SQN buffer
                    self.performance_stats[symbol].append(net_pnl)
                    restored_count += 1
            
            if restored_count > 0:
                logger.info(f"{LogSymbols.SUCCESS} Restored {restored_count} trades since Anchor ({self.last_reset_ts}).")
                for sym, stats in self.daily_execution_stats.items():
                    if stats['losses'] > 0 or stats['pnl'] != 0:
                        logger.info(f"   -> {sym}: PnL ${stats['pnl']:.2f} | Losses: {stats['losses']}")
            else:
                logger.info(f"{LogSymbols.INFO} No trades found for current session (Anchor: {self.last_reset_ts}).")
                
        except Exception as e:
            logger.error(f"{LogSymbols.ERROR} Failed to restore state: {e}")

    def _restore_penalty_box_state(self):
        """
        Restores Penalty Box state by scanning recent closed trades from Redis.
        Prevents "Revenge Trading" if the bot restarts immediately after a loss.
        """
        logger.info(f"{LogSymbols.DATABASE} Restoring Penalty Box State (Revenge Guard)...")
        cooldown_mins = CONFIG.get('risk_management', {}).get('loss_cooldown_minutes', 60)
        
        # Look back slightly longer than cooldown to be safe
        now_ts = datetime.now(timezone.utc).timestamp()
        start_ts = now_ts - (cooldown_mins * 60 + 600) # Buffer 10 mins
        start_id = int(start_ts * 1000)
        
        stream_key = CONFIG['redis'].get('closed_trade_stream_key', 'stream:closed_trades')
        
        try:
            # Fetch recent trades
            messages = self.stream_mgr.r.xrange(stream_key, min=start_id, max='+')
            
            restored_count = 0
            
            for _, data in messages:
                symbol = data.get('symbol')
                net_pnl = float(data.get('net_pnl', 0.0))
                close_ts = float(data.get('timestamp', 0.0))
                
                # Check if it was a loss
                if net_pnl < 0:
                    expiry = close_ts + (cooldown_mins * 60)
                    if expiry > now_ts:
                        # Trade implies active penalty
                        remaining = (expiry - now_ts) / 60.0
                        if remaining > 0:
                            self.portfolio_mgr.add_to_penalty_box(symbol, duration_minutes=int(remaining) + 1)
                            logger.warning(f"🚫 RESTORED PENALTY: {symbol} (Loss ${net_pnl:.2f}) -> Locked for {remaining:.1f}m")
                            restored_count += 1
            
            if restored_count == 0:
                logger.info("✅ No active penalties found in history.")
                
        except Exception as e:
            logger.error(f"Failed to restore penalty box: {e}")

    def _calendar_sync_loop(self):
        """
        Background thread to fetch Economic Calendar and update Compliance Guard.
        """
        logger.info(f"{LogSymbols.NEWS} Economic Calendar Sync Thread Started.")
        try:
            events = self.news_monitor.fetch_events()
            if events:
                self.compliance_guard = FTMOComplianceGuard(events)
                logger.info(f"{LogSymbols.NEWS} Initial Calendar Loaded: {len(events)} High Impact Events.")
        except Exception as e:
            logger.warning(f"Initial Calendar Fetch Failed: {e}")

        while not self.shutdown_flag:
            time.sleep(3600) 
            try:
                events = self.news_monitor.fetch_events()
                if events:
                    self.compliance_guard = FTMOComplianceGuard(events)
                    logger.info(f"{LogSymbols.NEWS} Calendar Synced: {len(events)} events active.")
            except Exception as e:
                logger.error(f"Calendar Sync Loop Error: {e}")

    def fetch_news_loop(self):
        """
        Background thread to fetch general news sentiment (NLP).
        """
        logger.info(f"{LogSymbols.NEWS} Sentiment Analysis Thread Started.")
        while not self.shutdown_flag:
            try:
                url = 'https://www.forexfactory.com/news'
                article = Article(url)
                article.download()
                article.parse()
                
                if article.text:
                    blob = TextBlob(article.text)
                    polarity = blob.sentiment.polarity
                    self.global_sentiment['GLOBAL'] = polarity
                    logger.info(f"{LogSymbols.NEWS} Market Sentiment Updated: {polarity:.3f}")
                
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"News Fetch Error: {e}")
                time.sleep(60)

    def fetch_performance_loop(self):
        """
        Background thread to listen for CLOSED trades.
        1. Updates SQN stats (performance sizing).
        2. Updates Daily Circuit Breaker (stops trading on bad days).
        3. Enforces MANDATORY COOLDOWN (prevents machine-gun trading).
        """
        logger.info(f"{LogSymbols.INFO} Performance Monitor (SQN & Circuit Breaker) Thread Started.")
        stream_key = CONFIG['redis'].get('closed_trade_stream_key', 'stream:closed_trades')
        
        # Start reading from current time ($)
        last_id = '$' 
        
        # Load cooldown duration from config
        cooldown_mins = CONFIG.get('risk_management', {}).get('loss_cooldown_minutes', 60)

        while not self.shutdown_flag:
            try:
                # Check if Day has changed (Anchor Update)
                current_anchor = self._get_producer_anchor_timestamp()
                if current_anchor > self.last_reset_ts:
                    with self.stats_lock:
                        logger.info(f"⚓ DAILY RESET DETECTED in Monitor (New Anchor: {current_anchor})")
                        self.last_reset_ts = current_anchor
                        self.daily_execution_stats.clear()
                
                # Block for 1 second waiting for new closed trades
                response = self.stream_mgr.r.xread({stream_key: last_id}, count=10, block=1000)
                
                if response:
                    for stream, messages in response:
                        for msg_id, data in messages:
                            last_id = msg_id
                            symbol = data.get('symbol')
                            net_pnl = float(data.get('net_pnl', 0.0))
                            ticket = str(data.get('ticket', ''))
                            ts_raw = float(data.get('timestamp', 0))
                            
                            # Filter old trades (Safety)
                            if ts_raw < self.last_reset_ts:
                                continue
                            
                            if symbol:
                                with self.stats_lock:
                                    # 1. SQN Window Update
                                    self.performance_stats[symbol].append(net_pnl)
                                    
                                    # 2. Update Daily Stats (Deduplicated)
                                    if ticket and ticket not in self.daily_execution_stats[symbol]['tickets']:
                                        self.daily_execution_stats[symbol]['pnl'] += net_pnl
                                        if ticket:
                                            self.daily_execution_stats[symbol]['tickets'].add(ticket)
                                        
                                        if net_pnl < 0:
                                            self.daily_execution_stats[symbol]['losses'] += 1
                                            logger.info(f"📉 LOSS DETECTED {symbol}: ${net_pnl:.2f} | Daily Losses: {self.daily_execution_stats[symbol]['losses']}")
                                        else:
                                            logger.info(f"💰 PROFIT DETECTED {symbol}: ${net_pnl:.2f}")

                                    # 3. GLOBAL MANDATORY COOLDOWN (Fixes "Machine Gun" Re-entry)
                                    # Apply penalty box to ALL closed trades (Win or Loss).
                                    # This forces the strategy to re-evaluate after a break, preventing tilt/overtrading.
                                    self.portfolio_mgr.add_to_penalty_box(symbol, duration_minutes=cooldown_mins)
                                    
                                    if net_pnl < 0:
                                        logger.warning(f"🚫 {symbol} REVENGE GUARD: Locked for {cooldown_mins}m (Loss)")
                                    else:
                                        logger.info(f"🛡️ {symbol} MANDATORY COOLDOWN: Locked for {cooldown_mins}m (Win/Break-even)")
            
            except Exception as e:
                logger.error(f"Performance Monitor Error: {e}")
                time.sleep(5)

    def _calculate_sqn(self, symbol: str) -> float:
        """
        Calculates the rolling System Quality Number (SQN) for a symbol.
        SQN = (Mean PnL / Std Dev PnL) * Sqrt(N)
        """
        with self.stats_lock:
            trades = list(self.performance_stats[symbol])
            
        if len(trades) < 5: 
            return 0.0 # Not enough data, neutral score
            
        avg_pnl = np.mean(trades)
        std_pnl = np.std(trades)
        
        if std_pnl < 1e-9:
            return 0.0
            
        sqn = (avg_pnl / std_pnl) * math.sqrt(len(trades))
        return sqn
        
    def _check_circuit_breaker(self, symbol: str) -> bool:
        """
        Checks if the symbol is locked out for the day due to losses.
        Returns TRUE if trading should be BLOCKED.
        """
        # Ensure we are using the latest Anchor
        current_anchor = self._get_producer_anchor_timestamp()
        
        with self.stats_lock:
            # Auto-reset if accessed on a new day (if loop hasn't caught it yet)
            if current_anchor > self.last_reset_ts:
                logger.info(f"⚓ LAZY DAILY RESET for {symbol} (Anchor: {current_anchor})")
                self.last_reset_ts = current_anchor
                self.daily_execution_stats.clear()
                return False
                
            stats = self.daily_execution_stats[symbol]
            losses = stats['losses']
            pnl = stats['pnl']

        # Rule 1: Max Losses per day
        if losses >= self.max_daily_losses_per_symbol:
            return True
            
        # Rule 2: Daily PnL < -1% of Equity
        # Estimate equity roughly from risk monitor
        current_equity = self.ftmo_guard.equity
        if current_equity > 0:
            limit = current_equity * 0.01
            if pnl < -limit:
                return True
                
        return False

    def _get_open_positions_from_redis(self) -> Dict[str, Dict]:
        """
        Fetches the current open positions synced by Windows Producer.
        """
        magic = CONFIG['trading']['magic_number']
        key = f"{CONFIG['redis']['position_state_key_prefix']}:{magic}"
        
        try:
            data = self.stream_mgr.r.get(key)
            if not data: return {}
            
            pos_list = json.loads(data)
            pos_map = {}
            for p in pos_list:
                sym = p.get('symbol')
                if sym:
                    pos_map[sym] = p
            
            # Cache for Worker Process injection
            self.latest_positions = pos_map
            return pos_map
        except Exception:
            return {}

    def _check_symbol_concurrency(self, symbol: str) -> bool:
        """
        V17.0 FIX: Enforces 1 Trade Per Pair, up to Global Max.
        UPDATED: Checks for Pyramiding configuration to allow adding to winners.
        """
        # 1. Fetch current positions
        open_positions = self._get_open_positions_from_redis()
        
        # 2. Check Global Limit (8)
        max_global = CONFIG['risk_management'].get('max_open_trades', 8)
        if len(open_positions) >= max_global:
            logger.warning(f"🚫 Global Trade Limit Reached ({len(open_positions)}/{max_global}).")
            return False

        # 3. Check if THIS symbol is already active
        if symbol in open_positions:
            # CHECK PYRAMIDING CONFIG
            is_pyramid_on = CONFIG['risk_management'].get('pyramiding', {}).get('enabled', False)
            if is_pyramid_on:
                # If Pyramiding is enabled, we DO NOT block here.
                # We allow the signal to proceed to RiskManager, which validates the specific "Add" criteria (Profitability, R-Multiple).
                return True
            else:
                logger.warning(f"🚫 {symbol} already active. Concurrency Limit (1 per pair).")
                return False
            
        return True

    def _signal_listener(self):
        """
        Consumes signals from LogicWorker and dispatches orders.
        This runs in the Main Process to handle Redis IO and Risk Checks.
        Rec 1 Implementation.
        """
        logger.info(f"{LogSymbols.SIGNAL} Signal Dispatcher Thread Active.")
        
        while not self.shutdown_flag:
            try:
                # 1. Pop Signal from Queue
                try:
                    signal = self.signal_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # --- FIX: IGNORE WARMUP SIGNALS ---
                # "WARMUP" signals are for model training only, not execution.
                # If we dispatch them, they default to "BUY" and cause invalid stops.
                if signal.action == "WARMUP":
                    continue
                # ----------------------------------
                
                symbol = signal.symbol
                
                # 2. Re-Validate Risk (State might have changed since Worker processing)
                if not self._check_risk_gates(symbol):
                    continue
                
                # 3. Execution Throttle
                ttl_sec = CONFIG['trading'].get('limit_order_ttl_seconds', 60)
                if time.time() - self.last_dispatch_time.get(symbol, 0) < ttl_sec:
                    continue
                
                # 4. CONCURRENCY CHECK (V17.0)
                if not self._check_symbol_concurrency(symbol):
                    continue

                # 5. Construct Trade Intent
                # NOTE: Re-calculate size locally to ensure Thread Safety on Balance
                
                open_positions = self._get_open_positions_from_redis()
                
                # Retrieve Meta Data
                current_atr = signal.meta_data.get('atr', 0.001)
                ker_val = signal.meta_data.get('ker', 1.0)
                volatility = signal.meta_data.get('volatility', 0.001)
                risk_percent_override = signal.meta_data.get('risk_percent_override')
                
                # Calculate SQN
                sqn_score = self._calculate_sqn(symbol)
                
                # Calculate Daily PnL Pct
                try:
                    start_eq = float(self.stream_mgr.r.get(CONFIG['redis']['risk_keys']['daily_starting_equity']) or 0.0)
                    if start_eq > 0:
                        daily_pnl_pct = (self.ftmo_guard.equity - start_eq) / start_eq
                    else:
                        daily_pnl_pct = 0.0
                except:
                    daily_pnl_pct = 0.0
                
                # Portfolio Heat
                active_corrs = self.portfolio_mgr.get_correlation_count(symbol)
                
                # Current Open Risk
                current_open_risk_pct = 0.0 # Simplified for listener, RiskManager handles margin checks
                
                # Get CURRENT PRICE for accurate calculation
                current_price = self.latest_prices.get(symbol, 0.0)
                if current_price <= 0:
                    logger.warning(f"⚠️ {symbol}: No price data for execution. Skipping.")
                    continue

                # --- V17.0 FIX: Fetch Margin Info for Leverage Guard ---
                estimated_free_margin = 100000.0 # Default fallback
                try:
                    acc_info = self.stream_mgr.r.hgetall(CONFIG['redis']['account_info_key'])
                    if acc_info:
                        estimated_free_margin = float(acc_info.get('free_margin', 100000.0))
                except Exception:
                    pass

                # Calculate Trade Size
                trade_intent, risk_usd = RiskManager.calculate_rck_size(
                    context=TradeContext(
                        symbol=symbol,
                        price=current_price, # Use actual live price
                        stop_loss_price=0.0,
                        account_equity=self.ftmo_guard.equity
                    ),
                    conf=signal.confidence,
                    volatility=volatility,
                    active_correlations=active_corrs,
                    market_prices=self.latest_prices,
                    atr=current_atr, 
                    ker=ker_val,
                    account_size=self.ftmo_guard.equity,
                    risk_percent_override=risk_percent_override,
                    performance_score=sqn_score,
                    daily_pnl_pct=daily_pnl_pct,
                    current_open_risk_pct=0.0, # Let RiskManager handle defaults or calc if needed
                    free_margin=estimated_free_margin # Pass explicit free margin
                )
                
                if trade_intent.volume <= 0: continue
                
                # Apply Signal Actions
                trade_intent.action = signal.action
                
                # --- MARKET EXECUTION LOGIC (FORCE ENTRY PRICE = CURRENT MARKET) ---
                
                # Apply Signal-Specific SL/TP Logic (Tighten Stops, Pyramid, etc.)
                tighten_stops = signal.meta_data.get('tighten_stops', False)
                optimized_rr = signal.meta_data.get('optimized_rr', 2.0)
                
                atr_mult_sl = float(CONFIG['risk_management'].get('stop_loss_atr_mult', 1.5))
                if tighten_stops:
                    atr_mult_sl = max(1.0, atr_mult_sl * 0.75)
                    trade_intent.comment += "|Tightened"
                
                # --- ROBUST PIP VALUE DETECTION ---
                pip_val, _ = RiskManager.get_pip_info(symbol)
                
                # FORCE JPY FIX: If symbol has JPY, pip MUST be 0.01
                if "JPY" in symbol and pip_val < 0.01:
                      pip_val = 0.01
                
                # Calculate Distances
                stop_dist = current_atr * atr_mult_sl
                
                # ENFORCE HARD FLOOR
                # Default to 15 pips if config missing or too small
                min_pips = float(CONFIG['risk_management'].get('min_stop_loss_pips', 15.0))
                hard_floor_dist = min_pips * pip_val
                
                # Log Geometry for Debugging
                logger.info(f"📉 SL GEOMETRY: ATR={current_atr:.5f} | Floor={hard_floor_dist:.5f} ({min_pips} pips) | Calc={stop_dist:.5f}")
                
                # Apply Floor
                stop_dist = max(stop_dist, hard_floor_dist)
                
                dynamic_tp_dist = stop_dist * optimized_rr
                
                # Calculate Absolute Prices relative to ENTRY PRICE (Current Market)
                entry_ref = current_price
                
                if signal.action == "BUY":
                    abs_sl = entry_ref - stop_dist
                    abs_tp = entry_ref + dynamic_tp_dist
                elif signal.action == "SELL":
                    abs_sl = entry_ref + stop_dist
                    abs_tp = entry_ref - dynamic_tp_dist
                else:
                    abs_sl = 0.0
                    abs_tp = 0.0

                # --- VALIDATION GATE ---
                if abs_sl <= 0 or abs_tp <= 0:
                    logger.error(f"🛑 INVALID SL/TP CALCULATION: {symbol} Action:{signal.action} Price:{entry_ref} SL:{abs_sl} TP:{abs_tp}. Trade Blocked.")
                    continue

                # Update Trade Object with Absolute Prices
                trade_intent.entry_price = entry_ref # Force Entry Price to Match Market Ref
                trade_intent.stop_loss = abs_sl
                trade_intent.take_profit = abs_tp
                
                # ------------------------------------------------
                
                # 6. Dispatch
                logger.info(f"📤 Dispatching MARKET {trade_intent.action} {symbol} ({trade_intent.volume} lots) @ ~{entry_ref:.5f} | SL: {abs_sl:.5f} | TP: {abs_tp:.5f}")
                self.dispatcher.send_order(trade_intent, risk_usd)
                self.last_dispatch_time[symbol] = time.time()
                
            except Exception as e:
                logger.error(f"Signal Listener Error: {e}", exc_info=True)

    def _manage_active_positions_loop(self):
        """
        Active Position Management Thread.
        Enforces:
        1. 4h Time Stop (Hard Exit) - Dynamic Config.
        2. 0.5R Trailing Stop.
        3. Daily Session Liquidation (Redundancy Check) - Uses authoritative NY Time.
        """
        logger.info(f"{LogSymbols.INFO} Active Position Manager Started (Trail: 0.5R, Time-Stop: Dynamic).")
        
        # Dynamic Horizon Fetch (Default 240m)
        tbm_conf = CONFIG.get('online_learning', {}).get('tbm', {})
        horizon_minutes = int(tbm_conf.get('horizon_minutes', 240))
        hard_stop_seconds = horizon_minutes * 60
        logger.info(f"{LogSymbols.TIME} Dynamic Time Stop Set to: {horizon_minutes} minutes ({horizon_minutes/60:.1f}h)")

        while not self.shutdown_flag:
            try:
                positions = self._get_open_positions_from_redis()
                if not positions:
                    time.sleep(5)
                    continue

                now_utc = datetime.now(pytz.utc)

                # SESSION CHECK (Redundancy)
                # Pass UTC time explicitly to ensure NY conversion works correctly within the guard
                session_liquidation = self.session_guard.should_liquidate(timestamp=now_utc)

                for sym, pos in positions.items():
                    exit_reason = None

                    # --- 1. SESSION HARD CLOSE ---
                    if session_liquidation:
                        exit_reason = "Session End Guard (3h pre-close)"
                        logger.warning(f"⌛ SESSION LIQUIDATION: {sym} closed by background manager.")

                    # --- 2. TIME BASED EXITS (DYNAMIC) ---
                    if not exit_reason:
                        entry_ts = float(pos.get('time', 0))
                        if entry_ts > 0:
                            entry_dt = datetime.fromtimestamp(entry_ts, pytz.utc)
                            duration = (now_utc - entry_dt).total_seconds()
                            
                            # Hard Time Stop (Configured Horizon)
                            if duration > hard_stop_seconds:
                                hours = int(horizon_minutes / 60)
                                exit_reason = f"Time Stop ({hours}h)"
                                logger.warning(f"⌛ TIME STOP: {sym} held for {duration/3600:.1f}h. Closing.")
                        
                    if exit_reason:
                        close_intent = Trade(
                            symbol=sym, 
                            action="CLOSE_ALL", 
                            volume=0.0, 
                            entry_price=0.0, 
                            stop_loss=0.0, 
                            take_profit=0.0, 
                            comment=exit_reason
                        )
                        self.dispatcher.send_order(close_intent, 0.0)
                        continue

                    # --- TRAILING STOP LOGIC (0.5R) ---
                    current_price = self.latest_prices.get(sym, 0.0)
                    if current_price <= 0: continue

                    entry_price = float(pos.get('entry_price', 0.0))
                    sl_price = float(pos.get('sl', 0.0))
                    tp_price = float(pos.get('tp', 0.0))
                    pos_type = pos.get('type') # "BUY" or "SELL"
                    ticket = pos.get('ticket')

                    if entry_price == 0 or sl_price == 0: continue

                    # Calculate Risk Distance (1R)
                    risk_dist = abs(entry_price - sl_price)
                    if risk_dist < 1e-5: continue

                    new_sl = None
                    
                    if pos_type == "BUY":
                        dist_pnl = current_price - entry_price
                        r_multiple = dist_pnl / risk_dist
                        
                        # Activate at 0.5R
                        if r_multiple >= 0.5:
                            # Move SL to Entry + 0.1R (Small lock-in)
                            target_sl = entry_price + (risk_dist * 0.1)
                            # Or if deep in profit (>1R), trail by 0.3R behind price
                            if r_multiple >= 1.0:
                                target_sl = current_price - (risk_dist * 0.3)
                            
                            # Only move UP
                            if target_sl > sl_price:
                                new_sl = target_sl

                    elif pos_type == "SELL":
                        dist_pnl = entry_price - current_price
                        r_multiple = dist_pnl / risk_dist
                        
                        # Activate at 0.5R
                        if r_multiple >= 0.5:
                            # Move SL to Entry - 0.1R
                            target_sl = entry_price - (risk_dist * 0.1)
                            # Trail if deep profit
                            if r_multiple >= 1.0:
                                target_sl = current_price + (risk_dist * 0.3)
                                
                            # Only move DOWN
                            if target_sl < sl_price:
                                new_sl = target_sl

                    if new_sl:
                        logger.info(f"🛡️ TRAILING STOP: {sym} (R={r_multiple:.2f}) -> New SL {new_sl:.5f}")
                        modify_intent = Trade(
                            symbol=sym,
                            action="MODIFY",
                            volume=0.0,
                            entry_price=0.0,
                            stop_loss=new_sl,
                            take_profit=tp_price, # Keep TP
                            ticket=ticket,
                            comment="Trail 0.5R"
                        )
                        self.dispatcher.send_order(modify_intent, 0.0)

            except Exception as e:
                logger.error(f"Position Manager Error: {e}")
            
            time.sleep(1) 

    def _reconcile_pending_orders(self, open_positions: Dict[str, Dict]):
        """
        Clears 'Pending' status if any position exists for the symbol.
        Robust against broken comment/UUID chains.
        """
        self.dispatcher.cleanup_stale_orders(ttl_seconds=600, open_positions=open_positions)

    def process_tick(self, tick_data: dict):
        """
        Main Thread Tick Handler.
        Simply unpacks and pushes to LogicWorker Queue.
        Minimal latency. Rec 1.
        """
        try:
            symbol = tick_data.get('symbol')
            if not symbol: return
            
            # --- CRITICAL: UPDATE LATEST PRICE ---
            bid = float(tick_data.get('bid', 0.0))
            ask = float(tick_data.get('ask', 0.0))
            if 'price' in tick_data:
                price = float(tick_data['price'])
            elif bid > 0 and ask > 0:
                price = (bid + ask) / 2.0
            else:
                price = 0.0
            
            if price > 0:
                self.latest_prices[symbol] = price

            if symbol not in self.symbols: return
            
            # Deduplication check
            ts = float(tick_data.get('time', 0.0))
            if ts <= self.processed_ticks[symbol]: return
            self.processed_ticks[symbol] = ts
            
            # Visual Heartbeat
            self.ticks_processed += 1
            if self.ticks_processed % 1000 == 0:
                logger.info(f"⚡ HEARTBEAT: Processed {self.ticks_processed} ticks... (Last: {symbol})")

            # Inject latest positions into tick data so Worker has context
            # This is critical for context-aware features without Redis IO in worker
            tick_data['positions'] = self.latest_positions.copy()
            
            # Non-blocking Push to LogicWorker
            try:
                self.tick_queue.put(tick_data, block=False)
            except queue.Full:
                logger.warning("⚠️ Tick Queue Full! Dropping tick to maintain real-time edge.")
                
        except Exception as e:
            logger.error(f"Main Process Tick Error: {e}")

    def _check_risk_gates(self, symbol: str) -> bool:
        """
        Runs the gauntlet of safety checks.
        Includes Margin Level Guard & Session Window Guard.
        """
        # 1. Midnight Freeze Check
        if self.stream_mgr.r.exists(CONFIG['redis']['risk_keys']['midnight_freeze']):
            logger.warning(f"{LogSymbols.FROZEN} Midnight Freeze Active. Holding for Daily Anchor.")
            return False

        # 2. Daily Anchor Existence Check
        if not self.stream_mgr.r.exists(CONFIG['redis']['risk_keys']['daily_starting_equity']):
            logger.warning(f"{LogSymbols.LOCK} Daily Anchor Missing. Waiting for Producer Sync.")
            return False

        # 3. FTMO Drawdown Guard
        if not self.ftmo_guard.can_trade():
            # Generate detailed audit log for the failure
            log_msg = self.ftmo_guard.check_circuit_breakers() if hasattr(self.ftmo_guard, 'check_circuit_breakers') else "Circuit Breaker Tripped"
            logger.warning(f"{LogSymbols.LOCK} FTMO Guard Halted: {log_msg}")
            return False

        # 4. General Market Hours & SESSION WINDOW
        # V17.5: Using internal time check which handles TZ correct
        if not self.session_guard.is_trading_allowed():
            # DIAGNOSTIC LOGGING FOR TIMEZONE ISSUES
            srv_time = datetime.now(self.session_guard.market_tz)
            logger.warning(f"{LogSymbols.LOCK} Session Guard Block: Server Time {srv_time.strftime('%H:%M')} (TZ: {self.session_guard.market_tz}) < Start Hour {self.session_guard.start_hour} or > End {self.session_guard.liq_hour}")
            return False
            
        # 5. Friday Entry Guard (No new trades after Noon NY Time)
        if self.session_guard.is_friday_afternoon():
            logger.warning(f"{LogSymbols.LOCK} Friday Guard Block: No new trades allowed late Friday (NY Noon).")
            return False

        # 6. Penalty Box (Correlation/Volatility penalty OR Revenge Guard)
        if self.portfolio_mgr.check_penalty_box(symbol):
            logger.warning(f"{LogSymbols.LOCK} {symbol} is in Penalty Box (Cool-down Active).")
            return False

        # 7. News Check
        if not self.compliance_guard.check_trade_permission(symbol):
             return False

        # --- 8. MARGIN LEVEL GUARD ---
        # Fetch current margin health from Redis
        try:
            acc_info = self.stream_mgr.r.hgetall(CONFIG['redis']['account_info_key'])
            if acc_info:
                equity = float(acc_info.get('equity', 0))
                margin = float(acc_info.get('margin', 0))
                
                # Margin Level = (Equity / Used Margin) * 100
                if margin > 0:
                    margin_level = (equity / margin) * 100.0
                    min_margin_level = CONFIG.get('risk_management', {}).get('min_margin_level_percent', 150.0)
                    
                    if margin_level < min_margin_level:
                        logger.critical(f"🛑 MARGIN LEVEL CRITICAL: {margin_level:.2f}% < {min_margin_level}%. BLOCKING TRADES.")
                        return False
        except Exception as e:
            logger.warning(f"Margin Check Failed: {e}")

        return True

    def run(self):
        """
        Main Event Loop.
        """
        logger.info(f"{LogSymbols.SUCCESS} Engine Loop Started. Waiting for Ticks on '{CONFIG['redis']['price_data_stream']}'...")
        self.is_warm = True
        
        # Start Logic Worker (Rec 1)
        self.logic_worker.start()
        
        stream_key = CONFIG['redis']['price_data_stream']
        group = self.stream_mgr.group_name
        consumer = f"engine-{CONFIG['trading']['magic_number']}"

        try:
            self.stream_mgr.r.xgroup_create(stream_key, group, id='0', mkstream=True)
        except Exception:
            pass

        while not self.shutdown_flag:
            try:
                # Check Worker Health
                if not self.logic_worker.is_alive():
                    logger.critical("🚨 Logic Worker DIED! Restarting...")
                    # Cleanup old queues if needed, or re-instantiate worker
                    self.logic_worker = LogicWorker(self.tick_queue, self.signal_queue, CONFIG, self.symbols)
                    self.logic_worker.start()

                # AUDIT FIX: PAUSE ON MIDNIGHT FREEZE
                # If the Producer is doing a rollover sync, we must pause tick processing
                if self.stream_mgr.r.exists(CONFIG['redis']['risk_keys']['midnight_freeze']):
                    logger.info("❄️ Midnight Freeze Detected. Pausing Tick Consumption...")
                    time.sleep(1)
                    continue

                response = self.stream_mgr.r.xreadgroup(
                    groupname=group,
                    consumername=consumer,
                    streams={stream_key: '>'},
                    count=50,
                    block=100
                )

                if response:
                    for stream, messages in response:
                        for message_id, data in messages:
                            self.process_tick(data)
                            self.stream_mgr.r.xack(stream_key, group, message_id)
                
                # --- SYNC RISK STATE FROM REDIS (PRODUCER AUTHORITY) ---
                try:
                    # 1. Update Current Equity (Fast update)
                    cached_eq = self.stream_mgr.r.get(CONFIG['redis']['risk_keys']['current_equity'])
                    if cached_eq and float(cached_eq) > 0:
                        self.ftmo_guard.update_equity(float(cached_eq))

                    # 2. Update Daily Anchor (The Fix for Drawdown False Positives)
                    cached_anchor = self.stream_mgr.r.get(CONFIG['redis']['risk_keys']['daily_starting_equity'])
                    if cached_anchor and float(cached_anchor) > 0:
                        self.ftmo_guard.starting_equity_of_day = float(cached_anchor)
                    
                    # 3. Update Account Size (Total Drawdown Base)
                    cached_size = self.stream_mgr.r.get("bot:account_size")
                    if cached_size and float(cached_size) > 0:
                        self.ftmo_guard.initial_balance = float(cached_size)
                        
                except Exception as e:
                    pass
                
            except Exception as e:
                logger.error(f"{LogSymbols.ERROR} Stream Read Error: {e}")
                time.sleep(1)

    def shutdown(self) -> None:
        logger.info(f"{LogSymbols.CLOSE} Engine Shutting Down...")
        self.shutdown_flag = True
        
        # Shutdown Worker
        self.tick_queue.put("STOP")
        self.logic_worker.join(timeout=5)
        if self.logic_worker.is_alive():
            self.logic_worker.terminate()
            
        logger.info("Logic Worker Terminated.")

if __name__ == "__main__":
    engine = LiveTradingEngine()
    try:
        engine.run()
    except KeyboardInterrupt:
        engine.shutdown()