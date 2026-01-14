# =============================================================================
# FILENAME: engines/live/engine.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/live/engine.py
# DEPENDENCIES: shared, engines.live.dispatcher, engines.live.predictor
# DESCRIPTION: Core Event Loop. Ingests ticks, aggregates Tick Imbalance Bars (TIBs),
# and generates signals via the Golden Trio Predictor.
#
# PHOENIX V13.1 UPDATE (LEVERAGE HARMONY):
# 1. MARGIN AWARENESS: Fetches real-time 'free_margin' from Redis.
# 2. SAFETY CLAMP: Passes free margin to RiskManager to prevent "No Money" errors.
# 3. BUG FIX (V13.1.1): Fixed critical deduplication pass-through error.
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
from typing import Any, Optional, Dict, List

# Third-Party NLP (Guarded)
try:
    from newspaper import Article
    from textblob import TextBlob
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("WARNING: NLP libraries not found (newspaper3k, textblob). Sentiment features disabled.")

# Shared Modules (Modularized Architecture)
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
    TradeContext
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
    3. Feeds Bars to Golden Trio Predictor (V12.4 Logic).
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
        
        # 3. Data Aggregation (Adaptive Imbalance Bars)
        self.aggregators = {}
        
        # Use Volume Threshold config as a proxy for Initial Imbalance Threshold
        init_threshold = CONFIG['data'].get('volume_bar_threshold', 50) # V12.4 Default
        # Default alpha 0.05 if not in config
        alpha = CONFIG['data'].get('imbalance_alpha', 0.05) 

        for sym in CONFIG['trading']['symbols']:
            self.aggregators[sym] = AdaptiveImbalanceBarGenerator(
                symbol=sym,
                initial_threshold=init_threshold,
                alpha=alpha
            )
            logger.info(f"Initialized TIB Generator for {sym} (Start: {init_threshold}, Alpha: {alpha})")

        # 4. AI Predictor (Golden Trio / ARF)
        self.predictor = MultiAssetPredictor(symbols=CONFIG['trading']['symbols'])
        
        # 5. Execution Dispatcher
        self.pending_orders = {}  # Track orders to prevent dupes
        self.lock = threading.RLock()
        self.dispatcher = TradeDispatcher(
            stream_mgr=self.stream_mgr,
            pending_tracker=self.pending_orders,
            lock=self.lock
        )

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
        
        # V12.31: Daily Circuit Breaker State (Realized Execution)
        # Added 'tickets' set to track unique deal IDs and prevent double counting
        self.daily_execution_stats = defaultdict(lambda: {'date': None, 'losses': 0, 'pnl': 0.0, 'tickets': set()})
        
        # V12.7 UNSHACKLED: Increased from 2 to 4 to align with 1% Risk and 4% Daily Buffer
        self.max_daily_losses_per_symbol = 4 
        
        # --- TIMEZONE INIT (MOVED UP FOR V12.20 FIX) ---
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
        self.liquidation_triggered_map = {sym: False for sym in CONFIG['trading']['symbols']}
        
        # --- TICK DEDUPLICATION & HEARTBEAT STATE ---
        self.processed_ticks = defaultdict(float)
        self.ticks_processed = 0 # Visual Heartbeat Counter

        # --- CONTEXT CACHE (D1/H4 from Windows) ---
        self.latest_context = defaultdict(dict) # {symbol: {'d1': {}, 'h4': {}}}

        # --- Volatility Gate Config ---
        self.vol_gate_conf = CONFIG['online_learning'].get('volatility_gate', {})
        self.use_vol_gate = self.vol_gate_conf.get('enabled', True)
        self.min_atr_spread_ratio = self.vol_gate_conf.get('min_atr_spread_ratio', 1.5)
        self.spread_map = CONFIG.get('forensic_audit', {}).get('spread_pips', {})

        # V11.1: Active Position Management Thread
        self.mgmt_thread = threading.Thread(target=self._manage_active_positions_loop, daemon=True)
        self.mgmt_thread.start()

    def _restore_daily_state(self):
        """
        CRITICAL: Replays today's closed trades from Redis to reconstruct Daily PnL and Loss Counts.
        Prevents the bot from 'forgetting' it hit a daily limit if it restarts mid-day.
        V12.31: Implements Ticket Deduplication to fix double-counting.
        """
        logger.info(f"{LogSymbols.DATABASE} Restoring Daily Execution State...")
        try:
            stream_key = CONFIG['redis'].get('closed_trade_stream_key', 'stream:closed_trades')
            
            # Calculate start of day timestamp (approximate to cover timezones)
            now = datetime.now(self.server_tz)
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_ts_ms = int(start_of_day.timestamp() * 1000)
            
            # Fetch range from start of day until now
            # Use '0-0' if we want absolutely everything, but start_ts_ms is more efficient
            # xrange(key, min, max)
            messages = self.stream_mgr.r.xrange(stream_key, min=start_ts_ms, max='+')
            
            restored_count = 0
            with self.stats_lock:
                current_date = now.date()
                
                for msg_id, data in messages:
                    symbol = data.get('symbol')
                    net_pnl = float(data.get('net_pnl', 0.0))
                    timestamp_raw = float(data.get('timestamp', 0))
                    ticket = str(data.get('ticket', '')) # Ticket for dedup
                    
                    # Verify timestamp matches today (Server Time)
                    trade_dt = datetime.fromtimestamp(timestamp_raw, tz=self.server_tz)
                    if trade_dt.date() != current_date:
                        continue # Skip old trades if range was loose
                        
                    # Initialize if needed
                    if self.daily_execution_stats[symbol]['date'] != current_date:
                        self.daily_execution_stats[symbol] = {'date': current_date, 'losses': 0, 'pnl': 0.0, 'tickets': set()}
                    
                    # --- DEDUPLICATION CHECK ---
                    if ticket and ticket in self.daily_execution_stats[symbol]['tickets']:
                        # Skip processing if we already saw this ticket today
                        continue

                    # Apply Logic
                    self.daily_execution_stats[symbol]['pnl'] += net_pnl
                    if net_pnl < 0:
                        self.daily_execution_stats[symbol]['losses'] += 1
                    
                    # Mark ticket as seen
                    if ticket:
                        self.daily_execution_stats[symbol]['tickets'].add(ticket)
                    
                    # Also restore SQN buffer
                    self.performance_stats[symbol].append(net_pnl)
                    restored_count += 1
            
            if restored_count > 0:
                logger.info(f"{LogSymbols.SUCCESS} Restored {restored_count} unique trades for today.")
                for sym, stats in self.daily_execution_stats.items():
                    if stats['losses'] > 0 or stats['pnl'] != 0:
                        logger.info(f"   -> {sym}: PnL ${stats['pnl']:.2f} | Losses: {stats['losses']}")
            else:
                logger.info(f"{LogSymbols.INFO} No trades found for today (Fresh Start).")
                
        except Exception as e:
            logger.error(f"{LogSymbols.ERROR} Failed to restore state: {e}")

    def _calendar_sync_loop(self):
        """
        Background thread to fetch Economic Calendar and update Compliance Guard.
        Runs every hour to keep blackout windows fresh.
        """
        logger.info(f"{LogSymbols.NEWS} Economic Calendar Sync Thread Started.")
        
        # Initial fetch attempt
        try:
            events = self.news_monitor.fetch_events()
            if events:
                self.compliance_guard = FTMOComplianceGuard(events)
                logger.info(f"{LogSymbols.NEWS} Initial Calendar Loaded: {len(events)} High Impact Events.")
        except Exception as e:
            logger.warning(f"Initial Calendar Fetch Failed: {e}")

        while not self.shutdown_flag:
            time.sleep(3600) # Sleep first, then update
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
        V12.31: Applies deduplication for live monitoring as well.
        """
        logger.info(f"{LogSymbols.INFO} Performance Monitor (SQN & Circuit Breaker) Thread Started.")
        stream_key = CONFIG['redis'].get('closed_trade_stream_key', 'stream:closed_trades')
        
        # Start reading from current time ($) to avoid re-processing old trades (handled by restore)
        last_id = '$' 
        
        while not self.shutdown_flag:
            try:
                # Block for 1 second waiting for new closed trades
                response = self.stream_mgr.r.xread({stream_key: last_id}, count=10, block=1000)
                
                if response:
                    for stream, messages in response:
                        for msg_id, data in messages:
                            last_id = msg_id
                            symbol = data.get('symbol')
                            net_pnl = float(data.get('net_pnl', 0.0))
                            ticket = str(data.get('ticket', ''))
                            
                            if symbol:
                                with self.stats_lock:
                                    # 1. SQN Window Update
                                    self.performance_stats[symbol].append(net_pnl)
                                    
                                    # 2. V10.0 Circuit Breaker Update
                                    now_server = datetime.now(self.server_tz).date()
                                    
                                    # Reset if new day
                                    if self.daily_execution_stats[symbol]['date'] != now_server:
                                        self.daily_execution_stats[symbol] = {'date': now_server, 'losses': 0, 'pnl': 0.0, 'tickets': set()}
                                    
                                    # --- DEDUPLICATION CHECK ---
                                    if ticket and ticket in self.daily_execution_stats[symbol]['tickets']:
                                        logger.warning(f"Duplicate Trade Ignored in Monitor: {ticket}")
                                        continue

                                    self.daily_execution_stats[symbol]['pnl'] += net_pnl
                                    
                                    if net_pnl < 0:
                                        self.daily_execution_stats[symbol]['losses'] += 1
                                        logger.info(f"📉 LOSS DETECTED {symbol}: ${net_pnl:.2f} | Daily Losses: {self.daily_execution_stats[symbol]['losses']}")
                                    
                                    if ticket:
                                        self.daily_execution_stats[symbol]['tickets'].add(ticket)
                                
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
        V10.0: Checks if the symbol is locked out for the day due to losses.
        Returns TRUE if trading should be BLOCKED.
        """
        now_server = datetime.now(self.server_tz).date()
        
        with self.stats_lock:
            stats = self.daily_execution_stats[symbol]
            
            # Auto-reset if accessed on a new day before trade loop hits it
            if stats['date'] != now_server:
                self.daily_execution_stats[symbol] = {'date': now_server, 'losses': 0, 'pnl': 0.0, 'tickets': set()}
                return False
                
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
        V12.7 REFACTOR: Active Position Management Thread.
        Enforces:
        1. 24h Time Stop (Hard Exit).
        2. 0.5R Trailing Stop.
        """
        logger.info(f"{LogSymbols.INFO} Active Position Manager Started (Time-Stop: 24h, Trail: 0.5R).")
        while not self.shutdown_flag:
            try:
                positions = self._get_open_positions_from_redis()
                if not positions:
                    time.sleep(5)
                    continue

                now_utc = datetime.now(pytz.utc)
                hard_stop_seconds = 86400 # 24 hours

                for sym, pos in positions.items():
                    # --- TIME BASED EXITS ---
                    entry_ts = float(pos.get('time', 0))
                    if entry_ts > 0:
                        entry_dt = datetime.fromtimestamp(entry_ts, pytz.utc)
                        duration = (now_utc - entry_dt).total_seconds()
                        
                        exit_reason = None
                        
                        # 1. Hard Time Stop (24h)
                        if duration > hard_stop_seconds:
                            exit_reason = "Time Stop (24h)"
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
            
            time.sleep(5) # Check every 5 seconds

    def process_tick(self, tick_data: dict):
        """
        Handles a single raw tick from Redis.
        Feeds the Adaptive Imbalance Bar Generator.
        Includes Strict Deduplication and VOLUME REPAIR.
        """
        try:
            symbol = tick_data.get('symbol')
            if symbol not in self.aggregators: return
            
            # --- TIMESTAMP DEDUPLICATION ---
            tick_ts = float(tick_data.get('time', 0.0))
            if tick_ts > 100_000_000_000:
                tick_ts /= 1000.0
            
            last_ts = self.processed_ticks.get(symbol, 0.0)
            if tick_ts <= last_ts:
                # CRITICAL FIX (V13.1.1): Dropped ticks must RETURN, not PASS.
                # 'pass' allowed duplicates to corrupt volume aggregation.
                return 
            
            self.processed_ticks[symbol] = tick_ts

            # --- VISUAL HEARTBEAT ---
            # Pulse check every 10 ticks so user knows it's alive
            self.ticks_processed += 1
            if self.ticks_processed % 1000 == 0:
                logger.info(f"⚡ HEARTBEAT: Processed {self.ticks_processed} ticks... (Last: {symbol})")

            # --- DATA INTEGRITY ---
            bid = float(tick_data.get('bid', 0.0))
            ask = float(tick_data.get('ask', 0.0))
            
            if 'price' in tick_data:
                price = float(tick_data['price'])
            elif bid > 0 and ask > 0:
                price = (bid + ask) / 2.0
            else:
                price = 0.0
            
            if price <= 0: return

            # --- CRITICAL FIX: FORCE VOLUME FLOOR (Synthetic Tick Rule) ---
            # If MT5 sends 0 volume (common in Forex/CFD ticks), we default to 1.0.
            # This ensures bars are generated and VPIN/OFI math doesn't div-by-zero.
            raw_vol = float(tick_data.get('volume', 0.0))
            volume = raw_vol if raw_vol > 0 else 1.0

            bid_vol = float(tick_data.get('bid_vol', 0.0))
            ask_vol = float(tick_data.get('ask_vol', 0.0))

            # If volumes were 0 upstream, inject synthetic split
            if bid_vol == 0 and ask_vol == 0:
                bid_vol = volume / 2.0
                ask_vol = volume / 2.0

            self.latest_prices[symbol] = price
            
            # --- CONTEXT EXTRACTION ---
            if 'ctx_d1' in tick_data:
                try: self.latest_context[symbol]['d1'] = json.loads(tick_data['ctx_d1'])
                except: pass
            
            if 'ctx_h4' in tick_data:
                try: self.latest_context[symbol]['h4'] = json.loads(tick_data['ctx_h4'])
                except: pass

            # 1. Feed Adaptive Aggregator
            # Now guarantees non-zero volume/flow
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
        
        # --- V12.5 GAP-PROOF WEEKEND LIQUIDATION ---
        # Get Bar Timestamp in Server Time
        if bar.timestamp.tzinfo is None:
            bar_ts_aware = bar.timestamp.replace(tzinfo=pytz.utc)
        else:
            bar_ts_aware = bar.timestamp
            
        server_time = bar_ts_aware.astimezone(self.server_tz)
        
        # Check for Weekend or Late Friday
        is_weekend_hold = (server_time.weekday() > 4) # Sat=5, Sun=6
        
        # Get Friday Close Hour from Risk config (default 21)
        friday_close_hour = CONFIG.get('risk_management', {}).get('friday_liquidation_hour_server', 21)
        is_friday_close = (server_time.weekday() == 4 and server_time.hour >= friday_close_hour)

        if is_weekend_hold or is_friday_close:
            # If we are in the danger zone, do not generate signals.
            # Liquidation is handled by the Active Position loop, but we must ensure
            # we don't open *new* trades here.
            # Also, trigger a liquidation if not already done for this symbol today.
            if not self.liquidation_triggered_map[symbol]:
                logger.warning(f"{LogSymbols.CLOSE} WEEKEND/FRIDAY GAP GUARD: Closing {symbol}.")
                close_intent = Trade(
                    symbol=symbol, 
                    action="CLOSE_ALL", 
                    volume=0.0, 
                    entry_price=0.0, 
                    stop_loss=0.0, 
                    take_profit=0.0, 
                    comment="Gap-Proof Liquidation"
                )
                self.dispatcher.send_order(close_intent, 0.0)
                self.liquidation_triggered_map[symbol] = True
            return # Stop processing
        else:
            # Reset trigger if we are back in safe hours
            if self.liquidation_triggered_map[symbol]:
                self.liquidation_triggered_map[symbol] = False
        
        # 2. Prepare Context
        current_sentiment = self.global_sentiment.get('GLOBAL', 0.0)
        mt5_context = self.latest_context.get(symbol, {})
        open_positions = self._get_open_positions_from_redis()
        
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
        # V10.0: Includes strict Midnight Anchor and Freeze check
        if not self._check_risk_gates(symbol):
            return

        # --- V12.24 ANTI-STACKING GUARD ---
        # Ensure we don't fire if a trade is currently pending dispatch or waiting at broker
        is_pending_execution = False
        with self.lock:
            # self.pending_orders is {uuid: {symbol: 'EURUSD', ...}}
            for oid, p_data in self.pending_orders.items():
                if p_data.get('symbol') == symbol:
                    is_pending_execution = True
                    break
        
        if is_pending_execution:
            # Silently ignore to prevent log spam, or debug log
            # logger.debug(f"Skipping {symbol} signal - Order already pending.")
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
        
        for sym, pos in open_positions.items():
            entry = float(pos.get('entry_price', 0.0))
            sl = float(pos.get('sl', 0.0))
            vol = float(pos.get('volume', 0.0))
            
            if sl > 0 and entry > 0:
                price_dist = abs(entry - sl)
                curr_p = self.latest_prices.get(sym, entry)
                rate = RiskManager.get_conversion_rate(sym, curr_p, self.latest_prices)
                
                risk_val = price_dist * vol * contract_size * rate
                current_open_risk_usd += risk_val
                
        equity = self.ftmo_guard.equity
        if equity > 0:
            current_open_risk_pct = (current_open_risk_usd / equity) * 100.0
        else:
            current_open_risk_pct = 0.0

        # --- V13.1: FETCH REAL FREE MARGIN (LEVERAGE HARMONY) ---
        # We fetch this from Redis 'account_info' which is updated by Windows Producer every second.
        try:
            acc_info_raw = self.stream_mgr.r.hgetall(CONFIG['redis']['account_info_key'])
            if acc_info_raw and 'free_margin' in acc_info_raw:
                real_free_margin = float(acc_info_raw['free_margin'])
            else:
                # Fallback: Assume 95% of equity is free if data missing (Risk of rejection, but better than 0)
                real_free_margin = equity * 0.95
        except:
            real_free_margin = equity * 0.95

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
            free_margin=real_free_margin # V13.1 PASSED
        )

        if trade_intent.volume <= 0:
            logger.warning(f"Trade Size 0 for {symbol} (Risk Constraints: {trade_intent.comment}).")
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

        # --- DYNAMIC R:R OVERRIDE ---
        optimized_rr = signal.meta_data.get('optimized_rr')
        if optimized_rr and optimized_rr > 0 and current_atr > 0:
            tp_dist = current_atr * optimized_rr
            trade_intent.comment += f"|OptRR:{optimized_rr:.2f}"

        # --- CRITICAL FIX: CONVERT DISTANCES TO ABSOLUTE PRICES ---
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
        self.dispatcher.send_order(trade_intent, risk_usd)

    def _check_risk_gates(self, symbol: str) -> bool:
        """
        Runs the gauntlet of safety checks.
        V10.0: Includes strict Midnight Anchor and Freeze check.
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

        # 4. General Market Hours
        if not self.session_guard.is_trading_allowed():
            return False
            
        # 5. Friday Entry Guard (No new trades after 16:00)
        if self.session_guard.is_friday_afternoon():
            return False

        # 6. Penalty Box (Correlation/Volatility penalty)
        if self.portfolio_mgr.check_penalty_box(symbol):
            logger.warning(f"{LogSymbols.LOCK} {symbol} is in Penalty Box.")
            return False

        # 7. News Check
        if not self.compliance_guard.check_trade_permission(symbol):
             return False

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
                    # We must respect the Anchor calculated by the Producer at midnight/startup.
                    cached_anchor = self.stream_mgr.r.get(CONFIG['redis']['risk_keys']['daily_starting_equity'])
                    if cached_anchor and float(cached_anchor) > 0:
                        # Direct attribute sync to ensure logic consistency
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