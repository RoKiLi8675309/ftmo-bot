# =============================================================================
# FILENAME: engines/live/engine.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/live/engine.py
# DEPENDENCIES: shared, engines.live.dispatcher, engines.live.predictor
# DESCRIPTION: Core Event Loop. Ingests ticks, aggregates Tick Imbalance Bars (TIBs),
# and generates signals via the Golden Trio Predictor.
#
# PHOENIX V16.22 UPDATE (SESSION GUARD INTEGRATION):
# - LIQUIDATION: integrated `session_guard.should_liquidate()` to enforce
#   the 21:00 Server Time hard close (3h before NY Close).
# - TRADING WINDOW: Blocks entries outside London/NY hours via `_check_risk_gates`.
# =============================================================================
import logging
import time
import json
import threading
import signal
import sys
import numpy as np
import math
import pytz
from datetime import datetime, timedelta, date
from collections import defaultdict, deque
from typing import Any, Optional, Dict, List, Set

# Third-Party NLP (Guarded)
try:
    from newspaper import Article
    from textblob import TextBlob
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("WARNING: NLP libraries not found (newspaper3k, textblob). Sentiment features disabled.")

# Shared Imports (Modularized Architecture)
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
    load_real_data # Required for Calibration
)

# Local Engine Modules
from engines.live.dispatcher import TradeDispatcher
from engines.live.predictor import MultiAssetPredictor

setup_logging("LiveEngine")
logger = logging.getLogger("LiveEngine")

# CONSTANTS
MAX_TICK_LATENCY_SEC = 45.0  # RELAXED: Increased to accommodate network jitter

class LiveTradingEngine:
    """
    The Central Logic Unit for the Linux Consumer.
    1. Consumes Ticks from Redis.
    2. Aggregates Ticks into Adaptive Imbalance Bars (TIBs).
    3. Feeds Bars to Golden Trio Predictor (V16.0 Logic).
    4. Manages Active Positions (Time Stop / Trailing).
    5. Dispatches Orders to Windows.
    """
    def __init__(self):
        self.shutdown_flag = False
        
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
        self.portfolio_mgr = PortfolioRiskManager(symbols=CONFIG['trading']['symbols'])
        self.session_guard = SessionGuard()
        
        # --- COMPLIANCE SETUP ---
        self.news_monitor = NewsEventMonitor()
        # Initialize Guard with empty events first to avoid blocking startup
        self.compliance_guard = FTMOComplianceGuard([])
        
        # Start Background Calendar Sync
        self.calendar_thread = threading.Thread(target=self._calendar_sync_loop, daemon=True)
        self.calendar_thread.start()
        
        # 3. Data Aggregation & Calibration (V16.7 SYNC FIX)
        # We must calibrate thresholds BEFORE initializing aggregators/predictor
        self.threshold_map = self._calibrate_thresholds()
        
        self.aggregators = {}
        # Default alpha 0.05 if not in config
        alpha = CONFIG['data'].get('imbalance_alpha', 0.05) 

        for sym in CONFIG['trading']['symbols']:
            # Use the calibrated threshold if found, else config default
            thresh = self.threshold_map.get(sym, CONFIG['data'].get('volume_bar_threshold', 10.0))
            
            self.aggregators[sym] = AdaptiveImbalanceBarGenerator(
                symbol=sym,
                initial_threshold=thresh,
                alpha=alpha
            )
            logger.info(f"Initialized TIB Generator for {sym} (Start: {thresh:.1f}, Alpha: {alpha})")

        # 4. AI Predictor (Golden Trio / ARF)
        # V16.7: Pass the calibrated map so Predictor warms up with SAME data
        self.predictor = MultiAssetPredictor(
            symbols=CONFIG['trading']['symbols'],
            threshold_map=self.threshold_map
        )
        
        # 5. Execution Dispatcher
        self.pending_orders = {}  # Track orders to prevent dupes
        self.lock = threading.RLock()
        self.dispatcher = TradeDispatcher(
            stream_mgr=self.stream_mgr,
            pending_tracker=self.pending_orders,
            lock=self.lock
        )
        
        # V16.13: EXECUTION THROTTLE STATE
        self.last_dispatch_time = defaultdict(float)
        
        # DEBUG FIX: Force clear pending orders on startup to remove ghosts
        with self.lock:
            self.pending_orders.clear()

        # 6. Sentiment Engine
        self.global_sentiment = {} # {symbol: score} or {'GLOBAL': score}
        if NLP_AVAILABLE:
            self.news_thread = threading.Thread(target=self.fetch_news_loop, daemon=True)
            self.news_thread.start()

        # 7. Performance Monitor (Circuit Breaker & SQN)
        # THREAD SAFETY: Lock for statistics updated by background thread
        self.stats_lock = threading.Lock()

        # Tracks last 30 trades per symbol to calculate SQN
        self.performance_stats = defaultdict(lambda: deque(maxlen=30))
        
        # V16.24: Server-Authority Stats
        # We store the "last reset timestamp" to detect day changes
        self.last_reset_ts = 0.0 
        
        # V12.31: Daily Circuit Breaker State (Realized Execution)
        # Added 'tickets' set to track unique deal IDs and prevent double counting
        self.daily_execution_stats = defaultdict(lambda: {'losses': 0, 'pnl': 0.0, 'tickets': set()})
        
        # V14.0 UNSHACKLED: Increased to 5 to prevent early lockout in Aggressor Mode
        self.max_daily_losses_per_symbol = 5
        
        # --- TIMEZONE INIT ---
        tz_str = CONFIG['risk_management'].get('risk_timezone', 'Europe/Prague')
        try:
            self.server_tz = pytz.timezone(tz_str)
        except Exception:
            self.server_tz = pytz.timezone('Europe/Prague')

        # --- STATE RESTORATION (CRITICAL) ---
        self._restore_daily_state()

        self.perf_thread = threading.Thread(target=self.fetch_performance_loop, daemon=True)
        self.perf_thread.start()

        # State
        self.is_warm = False
        self.last_corr_update = time.time()
        self.latest_prices = {}
        
        # --- V16.4 FIX: INJECT FALLBACK PRICES ---
        # Ensures RiskManager has conversion rates before first tick arrives
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
        self.min_atr_spread_ratio = self.vol_gate_conf.get('min_atr_spread_ratio', 1.0) # Lowered for V14
        self.spread_map = CONFIG.get('forensic_audit', {}).get('spread_pips', {})

        # --- V16.22 FIX: SAFETY GAP STATE ---
        # Tracks known open positions to detect closures instantly
        self.known_open_symbols: Set[str] = set()

        # V11.1: Active Position Management Thread
        self.mgmt_thread = threading.Thread(target=self._manage_active_positions_loop, daemon=True)
        self.mgmt_thread.start()

    def _calibrate_thresholds(self) -> Dict[str, float]:
        """
        V16.7 NEW: Analyzes historical data to determine optimal bar thresholds.
        Replicates the Backtester's logic exactly to ensure the Live model sees
        the same data distribution it was trained on.
        """
        logger.info(f"{LogSymbols.TRAINING} Starting Threshold Auto-Calibration...")
        threshold_map = {}
        alpha = CONFIG['data'].get('imbalance_alpha', 0.05)
        config_thresh = CONFIG['data'].get('volume_bar_threshold', 10.0)
        
        min_bars_needed = 500
        
        for sym in CONFIG['trading']['symbols']:
            try:
                # 1. Load Data (Last 30 Days)
                df = load_real_data(sym, n_candles=50000, days=30)
                if df.empty:
                    logger.warning(f"⚠️ Calibration: No data for {sym}. Using default {config_thresh}.")
                    threshold_map[sym] = config_thresh
                    continue
                
                # 2. Calibration Loop
                current_threshold = config_thresh
                attempts = 0
                max_attempts = 4
                
                final_bars = 0
                
                while attempts < max_attempts:
                    gen = AdaptiveImbalanceBarGenerator(sym, initial_threshold=current_threshold, alpha=alpha)
                    bar_count = 0
                    
                    for row in df.itertuples():
                        price = getattr(row, 'price', getattr(row, 'close', None))
                        vol = getattr(row, 'volume', 1.0)
                        ts_val = getattr(row, 'Index', None).timestamp()
                        if price is None: continue
                        
                        # Synthetic flow for calibration estimate
                        b_vol = vol / 2
                        s_vol = vol / 2
                        
                        if gen.process_tick(price, vol, ts_val, b_vol, s_vol):
                            bar_count += 1
                    
                    if bar_count >= min_bars_needed:
                        # Found a good threshold
                        threshold_map[sym] = current_threshold
                        final_bars = bar_count
                        break
                    else:
                        # Too few bars -> Threshold too high -> Lower it
                        attempts += 1
                        new_threshold = max(5.0, current_threshold * 0.5)
                        logger.info(f"🔎 {sym}: Calibrating... {bar_count} bars (Threshold {current_threshold:.1f} -> {new_threshold:.1f})")
                        current_threshold = new_threshold
                        # Fallback if max attempts reached, use the lowest tested
                        threshold_map[sym] = current_threshold
                
                logger.info(f"✅ {sym} Calibrated: Threshold {threshold_map[sym]:.1f} (~{final_bars} bars/month)")
                
            except Exception as e:
                logger.error(f"❌ Calibration Error {sym}: {e}")
                threshold_map[sym] = config_thresh
        
        return threshold_map

    def _inject_fallback_prices(self):
        """
        Injects static fallback prices for conversion pairs to prevent
        'No Rate' errors before the first tick arrives.
        """
        defaults = {
            "NZDUSD": 0.60, "AUDUSD": 0.65, "GBPUSD": 1.25, "EURUSD": 1.08,
            "USDJPY": 150.0, "USDCAD": 1.35, "USDCHF": 0.90, "AUDJPY": 95.0,
            "EURJPY": 160.0, "GBPJPY": 190.0
        }
        for sym, price in defaults.items():
            if sym not in self.latest_prices:
                self.latest_prices[sym] = price
        
        logger.info(f"💉 Injected {len(defaults)} fallback prices for PnL conversion.")

    def _get_producer_anchor_timestamp(self) -> float:
        """
        V16.24 FIX: Polls Redis for the AUTHORITATIVE Midnight Timestamp set by Windows Producer.
        """
        try:
            reset_ts = self.stream_mgr.r.get("risk:last_reset_date")
            if reset_ts:
                return float(reset_ts)
        except:
            pass
        
        # Fallback: Midnight UTC today
        return datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    def _restore_daily_state(self):
        """
        CRITICAL: Replays today's closed trades from Redis using the Producer's Timestamp Authority.
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
        3. V16.4: Enforces REVENGE TRADING GUARD (Penalty Box).
        """
        logger.info(f"{LogSymbols.INFO} Performance Monitor (SQN & Circuit Breaker) Thread Started.")
        stream_key = CONFIG['redis'].get('closed_trade_stream_key', 'stream:closed_trades')
        
        # Start reading from current time ($)
        last_id = '$' 
        
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
                                    
                                    # 2. Circuit Breaker & Revenge Guard
                                    # Deduplication Check
                                    if ticket and ticket in self.daily_execution_stats[symbol]['tickets']:
                                        continue

                                    self.daily_execution_stats[symbol]['pnl'] += net_pnl
                                    
                                    if ticket:
                                        self.daily_execution_stats[symbol]['tickets'].add(ticket)
                                    
                                    if net_pnl < 0:
                                        self.daily_execution_stats[symbol]['losses'] += 1
                                        logger.info(f"📉 LOSS DETECTED {symbol}: ${net_pnl:.2f} | Daily Losses: {self.daily_execution_stats[symbol]['losses']}")
                                        
                                        # V16.4 FIX: REVENGE TRADING GUARD (PENALTY BOX)
                                        cooldown = CONFIG.get('risk_management', {}).get('loss_cooldown_minutes', 60)
                                        self.portfolio_mgr.add_to_penalty_box(symbol, duration_minutes=cooldown)
                                        logger.warning(f"🚫 {symbol} SENT TO PENALTY BOX for {cooldown}m (Revenge Guard)")
            
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
            return pos_map
        except Exception:
            return {}

    def _manage_active_positions_loop(self):
        """
        V16.0 UPDATE: Active Position Management Thread.
        Enforces:
        1. 8h Time Stop (Hard Exit) - Reduced from 24h for Scalping.
        2. 0.5R Trailing Stop.
        3. V16.22 UPDATE: Daily Session Liquidation (Redundancy Check).
        """
        logger.info(f"{LogSymbols.INFO} Active Position Manager Started (Time-Stop: 8h, Trail: 0.5R).")
        while not self.shutdown_flag:
            try:
                positions = self._get_open_positions_from_redis()
                if not positions:
                    time.sleep(5)
                    continue

                now_utc = datetime.now(pytz.utc)
                hard_stop_seconds = 28800 # 8 Hours (Intraday Focus)

                # V16.22 SESSION CHECK (Redundancy)
                # If the main loop missed the liquidation time, we catch it here.
                session_liquidation = self.session_guard.should_liquidate()

                for sym, pos in positions.items():
                    exit_reason = None

                    # --- 1. SESSION HARD CLOSE ---
                    if session_liquidation:
                        exit_reason = "Session End Guard (3h pre-close)"
                        logger.warning(f"⌛ SESSION LIQUIDATION: {sym} closed by background manager.")

                    # --- 2. TIME BASED EXITS ---
                    if not exit_reason:
                        entry_ts = float(pos.get('time', 0))
                        if entry_ts > 0:
                            entry_dt = datetime.fromtimestamp(entry_ts, pytz.utc)
                            duration = (now_utc - entry_dt).total_seconds()
                            
                            # Hard Time Stop (8h)
                            if duration > hard_stop_seconds:
                                exit_reason = "Time Stop (8h)"
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
            
            time.sleep(1) # V16.10: Faster checking (1s) to clear ghosts immediately

    def _reconcile_pending_orders(self, open_positions: Dict[str, Dict]):
        """
        V16.10 GHOST BUSTER: Clears 'Pending' status if any position exists for the symbol.
        Robust against broken comment/UUID chains.
        """
        self.dispatcher.cleanup_stale_orders(ttl_seconds=600, open_positions=open_positions)

    def process_tick(self, tick_data: dict):
        """
        Handles a single raw tick from Redis.
        Feeds the Adaptive Imbalance Bar Generator.
        Includes Strict Deduplication and VOLUME REPAIR.
        """
        try:
            symbol = tick_data.get('symbol')
            
            # --- CRITICAL FIX 16.4.1: UPDATE LATEST PRICE BEFORE BLOCKING ---
            # Explicit float casting for safety
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
            # -----------------------------------------------------------------

            if symbol not in self.aggregators: 
                # This return is now safe because we updated latest_prices first.
                # This ensures RiskManager has live prices for conversion pairs (NZDUSD)
                # even if we aren't actively trading them.
                return
            
            # --- TIMESTAMP DEDUPLICATION ---
            tick_ts = float(tick_data.get('time', 0.0))
            # Normalize ms to seconds if needed
            if tick_ts > 100_000_000_000:
                tick_ts /= 1000.0
            
            last_ts = self.processed_ticks.get(symbol, 0.0)
            
            # AUDIT FIX: Relaxed Deduplication (< instead of <=)
            # Allows same-millisecond bursts (HFT/Volume Aggregation)
            if tick_ts < last_ts:
                return 
            
            self.processed_ticks[symbol] = tick_ts

            # --- VISUAL HEARTBEAT ---
            self.ticks_processed += 1
            if self.ticks_processed % 1000 == 0:
                logger.info(f"⚡ HEARTBEAT: Processed {self.ticks_processed} ticks... (Last: {symbol})")

            # --- CRITICAL FIX: FORCE VOLUME FLOOR (Synthetic Tick Rule) ---
            raw_vol = float(tick_data.get('volume', 0.0))
            volume = raw_vol if raw_vol > 0 else 1.0

            bid_vol = float(tick_data.get('bid_vol', 0.0))
            ask_vol = float(tick_data.get('ask_vol', 0.0))

            # --- CONTEXT EXTRACTION (SAFETY PATCH) ---
            if 'ctx_d1' in tick_data:
                try: 
                    self.latest_context[symbol]['d1'] = json.loads(tick_data['ctx_d1'])
                except (json.JSONDecodeError, TypeError): 
                    pass
            
            if 'ctx_h4' in tick_data:
                try: 
                    self.latest_context[symbol]['h4'] = json.loads(tick_data['ctx_h4'])
                except (json.JSONDecodeError, TypeError): 
                    pass

            # 1. Feed Adaptive Aggregator
            bar = self.aggregators[symbol].process_tick(
                price=price, 
                volume=volume, 
                timestamp=tick_ts, 
                external_buy_vol=bid_vol, 
                external_sell_vol=ask_vol
            )
            
            # 2. If Bar Complete -> Predict
            if bar:
                self.on_bar_complete(bar, symbol)
                
        except Exception as e:
            logger.error(f"Tick Processing Error: {e}")

    def on_bar_complete(self, bar: VolumeBar, symbol: str):
        """
        Triggered when a Tick Imbalance Bar (TIB) is closed.
        Executes the Prediction and Dispatch logic.
        """
        # =====================================================================
        # V16.23 CRITICAL FIX: SAFETY GAP PRIORITY INVERSION
        # MOVED TO THE ABSOLUTE TOP TO PREVENT RACE CONDITIONS.
        # =====================================================================
        
        # 1. Fetch Latest Positions Snapshot from Redis (Producer is Authority)
        open_positions = self._get_open_positions_from_redis()
        current_open_symbols = set(open_positions.keys())
        
        # 2. Detect Closures (Diff between last tick known state vs current state)
        just_closed = self.known_open_symbols - current_open_symbols
        
        # Update state for next tick immediately
        self.known_open_symbols = current_open_symbols

        # 3. SAFETY GAP ENFORCEMENT
        # If the symbol we are processing just closed, we MUST ABORT immediately.
        # This covers the gap between "Trade Closed in MT5" and "Redis Performance Stream updated".
        if symbol in just_closed:
            logger.warning(f"🛑 REVENGE KILLER: {symbol} just closed. Applying 5m penalty & ABORTING tick.")
            # Apply 5-minute cooldown (Extended for safety) to bridge gap until 
            # Performance Monitor applies the full 60m penalty (if loss).
            self.portfolio_mgr.add_to_penalty_box(symbol, duration_minutes=5)
            return  # <--- CRITICAL: DO NOT PROCEED TO SIGNAL GENERATION

        # --- V16.13: EXECUTION THROTTLE (THE MACHINE GUN FIX) ---
        # Forces a hard wait time between orders for the same symbol.
        # This covers the critical "Redis Round Trip" latency window.
        ttl_sec = CONFIG['trading'].get('limit_order_ttl_seconds', 60)
        last_time = self.last_dispatch_time.get(symbol, 0)
        
        if time.time() - last_time < ttl_sec:
            logger.info(f"⏳ {symbol} in Execution Cool-down ({ttl_sec}s). Skipping.")
            return

        # --- V10.0 EXECUTION CIRCUIT BREAKER ---
        if self._check_circuit_breaker(symbol):
            return

        # 1. Update Portfolio Risk State
        try:
            ret = (bar.close - bar.open) / bar.open if bar.open > 0 else 0.0
            self.portfolio_mgr.update_returns(symbol, ret)
            
            if time.time() - self.last_corr_update > 60:
                self.portfolio_mgr.update_correlation_matrix()
                self.last_corr_update = time.time()
        except Exception as e:
            logger.error(f"Portfolio Update Error: {e}")
        
        # --- V16.22 SESSION GUARD & WEEKEND LIQUIDATION ---
        # This checks:
        # A. Is it Weekend?
        # B. Is it Friday Close time?
        # C. Is it Daily Session Liquidation time (3h before NY Close)?
        if self.session_guard.should_liquidate():
            if not self.liquidation_triggered_map[symbol]:
                logger.warning(f"{LogSymbols.CLOSE} SESSION END GUARD: Closing {symbol} (Hard Close 3h before NY End).")
                close_intent = Trade(
                    symbol=symbol, 
                    action="CLOSE_ALL", 
                    volume=0.0, 
                    entry_price=0.0, 
                    stop_loss=0.0, 
                    take_profit=0.0, 
                    comment="Session/Gap Liquidation"
                )
                self.dispatcher.send_order(close_intent, 0.0)
                self.liquidation_triggered_map[symbol] = True
            return # Stop processing signals
        else:
            # Reset trigger if we are back in safe hours (next day)
            if self.liquidation_triggered_map[symbol] and self.session_guard.is_trading_allowed():
                self.liquidation_triggered_map[symbol] = False
        
        # 2. Prepare Context
        current_sentiment = self.global_sentiment.get('GLOBAL', 0.0)
        mt5_context = self.latest_context.get(symbol, {})
        
        # --- V16.7 FIX: RECONCILE PENDING ORDERS ---
        # Check if any "Pending" orders are actually open now
        self._reconcile_pending_orders(open_positions)

        # =====================================================================
        # B. MAX TRADES GUARD (RACE CONDITION FIX - V16.25)
        # =====================================================================
        max_open_trades = CONFIG['risk_management'].get('max_open_trades', 1)
        
        # V16.25: Count In-Flight Orders (Strict Logic)
        # We must count ANY pending order that was dispatched recently (<60s)
        # as a "reserved slot" to prevent firing a second trade while the first is in transit.
        pending_count = 0
        now_ts = time.time()
        
        symbol_has_pending = False
        
        with self.lock:
            # Iterate copy to be safe
            for oid, p_data in list(self.pending_orders.items()):
                # Clean up ancient orders (TTL 60s)
                if now_ts - p_data.get('timestamp', 0) > 60:
                    del self.pending_orders[oid]
                    continue
                
                # Count valid pending orders
                pending_count += 1
                if p_data.get('symbol') == symbol:
                    symbol_has_pending = True

        total_utilization = len(open_positions) + pending_count

        # V16.25: BLOCK if Limit Reached
        if total_utilization >= max_open_trades:
            # Only allow processing if this symbol is already OPEN (for management/exit)
            # If it's pending, we also block to avoid duplicate signals for same symbol
            if symbol not in open_positions:
                if self.ticks_processed % 100 == 0:
                    logger.info(f"🛑 MAX TRADES LIMIT ({total_utilization}/{max_open_trades}) [Open:{len(open_positions)} Pending:{pending_count}]. Skipping {symbol}.")
                return
        
        # V16.25: BLOCK if symbol has pending order (Machine Gun Protection)
        if symbol_has_pending:
            logger.info(f"⚠️ Anti-Machine Gun: {symbol} has pending order. Skipping.")
            return
        # =====================================================================
        
        context_data = {
            'd1': mt5_context.get('d1', {}),
            'h4': mt5_context.get('h4', {}),
            'sentiment': current_sentiment,
            'positions': open_positions
        }
        
        # 3. Get Signal (Golden Trio / ARF)
        signal = self.predictor.process_bar(symbol, bar, context_data=context_data)
        
        if not signal: return

        # --- PHASE 2: WARM-UP GATE ---
        if signal.action == "WARMUP":
            return

        # 4. Validate Signal
        if signal.action == "HOLD":
            return
            
        logger.info(f"{LogSymbols.SIGNAL} SIGNAL: {signal.action} {symbol} (Conf: {signal.confidence:.2f})")

        # --- VOLATILITY GATE ---
        current_atr = signal.meta_data.get('atr', 0.0)
        if self.use_vol_gate:
            pip_size, _ = RiskManager.get_pip_info(symbol)
            spread_pips = self.spread_map.get(symbol, 1.5)
            spread_cost = spread_pips * pip_size
            
            if current_atr < (spread_cost * self.min_atr_spread_ratio):
                logger.warning(f"{LogSymbols.LOCK} Vol Gate: {symbol} Rejected. Low Volatility.")
                return

        # 5. Check Risk & Compliance Gates
        # DEBUG: Verbose logging for Gate Checks
        logger.info(f"🛡️ Checking Risk Gates for {symbol}...")
        if not self._check_risk_gates(symbol):
            logger.warning(f"🚫 Risk Gate BLOCKED {symbol}")
            return

        # 6. Calculate Size (Fixed Risk Mode)
        volatility = signal.meta_data.get('volatility', 0.001)
        active_corrs = self.portfolio_mgr.get_correlation_count(
            symbol, 
            threshold=CONFIG['risk_management']['correlation_penalty_threshold']
        )

        try:
            cached_size = self.stream_mgr.r.get("bot:account_size")
            account_size = float(cached_size) if cached_size else self.ftmo_guard.equity
        except:
            account_size = self.ftmo_guard.equity

        # Retrieve Risk Override from Predictor (Optimized)
        risk_percent_override = signal.meta_data.get('risk_percent_override')
        
        # Retrieve KER for Risk Scaling (Sniper Protocol)
        ker_val = signal.meta_data.get('ker', 1.0)
        
        # --- DYNAMIC SQN PERFORMANCE SCORE ---
        sqn_score = self._calculate_sqn(symbol)

        # Initial Context
        ctx = TradeContext(
            symbol=symbol,
            price=bar.close,
            stop_loss_price=0.0,
            account_equity=self.ftmo_guard.equity,
            account_currency="USD",
            win_rate=0.45, 
            risk_reward_ratio=2.0 
        )

        # --- V12.0: CALCULATE DAILY PNL FOR BUFFER SCALING ---
        try:
            start_eq = float(self.stream_mgr.r.get(CONFIG['redis']['risk_keys']['daily_starting_equity']) or 0.0)
            if start_eq > 0:
                daily_pnl_pct = (self.ftmo_guard.equity - start_eq) / start_eq
            else:
                daily_pnl_pct = 0.0
        except:
            daily_pnl_pct = 0.0

        # --- V13.0: CALCULATE TOTAL OPEN RISK % (LIVE) ---
        current_open_risk_usd = 0.0
        contract_size = 100000
        
        # --- V16.11 MARGIN SAFETY: Calculate Used Margin Locally ---
        # Fallback if Redis data is missing or stale.
        local_used_margin = 0.0
        
        for sym, pos in open_positions.items():
            entry = float(pos.get('entry_price', 0.0))
            sl = float(pos.get('sl', 0.0))
            vol = float(pos.get('volume', 0.0))
            
            # Risk Calculation
            if sl > 0 and entry > 0:
                price_dist = abs(entry - sl)
                curr_p = self.latest_prices.get(sym, entry)
                rate = RiskManager.get_conversion_rate(sym, curr_p, self.latest_prices)
                
                risk_val = price_dist * vol * contract_size * rate
                current_open_risk_usd += risk_val
                
                # Margin Calculation (Estimated)
                # IMPORTANT: We use conservative leverage logic here (RiskManager handles 1:30 fallback)
                margin_req = RiskManager.calculate_required_margin(
                    symbol=sym, lots=vol, price=curr_p, contract_size=contract_size, conversion_rate=rate
                )
                local_used_margin += margin_req
                
        equity = self.ftmo_guard.equity
        if equity > 0:
            current_open_risk_pct = (current_open_risk_usd / equity) * 100.0
        else:
            current_open_risk_pct = 0.0

        # --- V16.11: FETCH REAL FREE MARGIN WITH LOCAL FALLBACK ---
        real_free_margin = 0.0
        try:
            acc_info_raw = self.stream_mgr.r.hgetall(CONFIG['redis']['account_info_key'])
            if acc_info_raw and 'free_margin' in acc_info_raw:
                redis_free_margin = float(acc_info_raw['free_margin'])
                
                # If Redis reports suspiciously high margin (== Equity) while we know we have positions,
                # TRUST LOCAL CALCULATION. This happens if the Broker isn't sending Margin updates fast enough.
                if len(open_positions) > 0 and redis_free_margin >= (equity * 0.99):
                    logger.warning(f"⚠️ Suspicious Redis Free Margin ({redis_free_margin:.2f}). Using Local Estimate.")
                    real_free_margin = max(0.0, equity - local_used_margin)
                else:
                    real_free_margin = redis_free_margin
            else:
                real_free_margin = max(0.0, equity - local_used_margin)
        except:
            real_free_margin = max(0.0, equity - local_used_margin)

        # DEBUG: Verbose Sizing Log
        logger.info(f"📐 Calculating Size for {symbol}: Equity={account_size}, Margin={real_free_margin:.2f}, Risk%= {risk_percent_override}")

        # Get Base Trade Intent with Buffer Scaling
        trade_intent, risk_usd = RiskManager.calculate_rck_size(
            context=ctx,
            conf=signal.confidence,
            volatility=volatility,
            active_correlations=active_corrs,
            market_prices=self.latest_prices,
            atr=current_atr,
            account_size=account_size,
            contract_size_override=None,
            ker=ker_val,
            risk_percent_override=risk_percent_override,
            performance_score=sqn_score,
            daily_pnl_pct=daily_pnl_pct,
            current_open_risk_pct=current_open_risk_pct,
            free_margin=real_free_margin 
        )

        if trade_intent.volume <= 0:
            logger.warning(f"❌ Trade Size 0 for {symbol}: {trade_intent.comment}")
            return

        trade_intent.action = signal.action

        # --- V10.2: VOLATILITY TRIGGER (TIGHTEN STOPS) ---
        tighten = signal.meta_data.get('tighten_stops', False)
        
        # Current SL/TP are Distances
        sl_dist = trade_intent.stop_loss
        tp_dist = trade_intent.take_profit
        
        if tighten:
            sl_dist *= 0.75 # Compress Stop by 25%
            trade_intent.comment += "|Tightened"
            logger.info(f"🛡️ STOP TIGHTENED: {symbol} (High Volatility) -> SL Dist {sl_dist:.5f}")

        # --- V16.8: STOP DISTANCE CLAMP (THE FIX) ---
        # Ensure SL/TP are not inside the spread (min 5 pips)
        pip_size, _ = RiskManager.get_pip_info(symbol)
        min_pips = CONFIG['risk_management'].get('min_stop_loss_pips', 5.0)
        min_dist = min_pips * pip_size
        
        if sl_dist < min_dist:
            sl_dist = min_dist
            trade_intent.comment += "|MinSL"
            
        if tp_dist < min_dist:
            tp_dist = min_dist
            trade_intent.comment += "|MinTP"

        # --- DYNAMIC R:R OVERRIDE ---
        # V14.0 UPDATE: Check for Momentum Ignition reward boost
        optimized_rr = signal.meta_data.get('optimized_rr')
        if optimized_rr and optimized_rr > 0 and current_atr > 0:
            tp_dist = current_atr * optimized_rr
            # Re-check min dist after boost
            if tp_dist < min_dist: tp_dist = min_dist
            trade_intent.comment += f"|OptRR:{optimized_rr:.2f}"

        # --- CONVERT DISTANCES TO ABSOLUTE PRICES ---
        if trade_intent.action == "BUY":
            trade_intent.stop_loss = bar.close - sl_dist
            trade_intent.take_profit = bar.close + tp_dist
        elif trade_intent.action == "SELL":
            trade_intent.stop_loss = bar.close + sl_dist
            trade_intent.take_profit = bar.close - tp_dist

        # --- PYRAMIDING LOGIC ---
        is_pyramid = signal.meta_data.get('pyramid', False)
        if is_pyramid:
            original_volume = trade_intent.volume
            trade_intent.volume = round(original_volume * 0.5, 2)
            min_lot = CONFIG['risk_management'].get('min_lot_size', 0.01)
            if trade_intent.volume < min_lot: trade_intent.volume = min_lot
            trade_intent.comment += "|PYRAMID"
            logger.info(f"⚡ PYRAMID: Scaling In {trade_intent.volume} lots")

        # 7. Dispatch
        logger.info(f"📤 Dispatching Order: {trade_intent.action} {trade_intent.symbol} {trade_intent.volume} Lots")
        self.dispatcher.send_order(trade_intent, risk_usd)
        
        # V16.13: UPDATE DISPATCH TIMER
        self.last_dispatch_time[symbol] = time.time()

    def _check_risk_gates(self, symbol: str) -> bool:
        """
        Runs the gauntlet of safety checks.
        V16.11: Includes Margin Level Guard.
        V16.22: Includes Session Window Guard.
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

        # 4. General Market Hours & SESSION WINDOW (V16.22)
        if not self.session_guard.is_trading_allowed():
            logger.warning(f"{LogSymbols.LOCK} Session Guard Block: Outside Allowed Window (London/NY).")
            return False
            
        # 5. Friday Entry Guard (No new trades after 16:00)
        if self.session_guard.is_friday_afternoon():
            logger.warning(f"{LogSymbols.LOCK} Friday Guard Block: No new trades allowed late Friday.")
            return False

        # 6. Penalty Box (Correlation/Volatility penalty OR Revenge Guard)
        if self.portfolio_mgr.check_penalty_box(symbol):
            logger.warning(f"{LogSymbols.LOCK} {symbol} is in Penalty Box (Cool-down Active).")
            return False

        # 7. News Check
        if not self.compliance_guard.check_trade_permission(symbol):
             return False

        # --- 8. MARGIN LEVEL GUARD (V16.11) ---
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
        
        stream_key = CONFIG['redis']['price_data_stream']
        group = self.stream_mgr.group_name
        consumer = f"engine-{CONFIG['trading']['magic_number']}"

        try:
            self.stream_mgr.r.xgroup_create(stream_key, group, id='0', mkstream=True)
        except Exception:
            pass

        while not self.shutdown_flag:
            try:
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
                    # Transient Redis errors ignored in tight loop
                    pass
                
            except Exception as e:
                logger.error(f"{LogSymbols.ERROR} Stream Read Error: {e}")
                time.sleep(1)

    def shutdown(self) -> None:
        logger.info(f"{LogSymbols.CLOSE} Engine Shutting Down...")
        self.predictor.save_state()
        self.shutdown_flag = True

if __name__ == "__main__":
    engine = LiveTradingEngine()
    try:
        engine.run()
    except KeyboardInterrupt:
        engine.shutdown()