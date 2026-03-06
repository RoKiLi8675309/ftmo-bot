import logging
import time
import json
import threading
import os
import multiprocessing
import queue
import numpy as np
import math
import pytz
import uuid
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from typing import Any, Optional, Dict, List, Set

# Third-Party NLP (Guarded)
try:
    from newspaper import Article
    from textblob import TextBlob
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

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
from engines.live.predictor import MultiAssetPredictor, Signal

setup_logging("LiveEngine")
logger = logging.getLogger("LiveEngine")

# CONSTANTS
MAX_TICK_LATENCY_SEC = 45.0 

# --- MT5 CONSTANTS (HARDCODED FOR LINUX) ---
MT5_ORDER_TYPE_BUY = 0
MT5_ORDER_TYPE_SELL = 1
MT5_TRADE_ACTION_DEAL = 1
MT5_TRADE_ACTION_PENDING = 5
MT5_TRADE_ACTION_SLTP = 6 
MT5_TRADE_ACTION_MODIFY = 6
MT5_TRADE_ACTION_REMOVE = 8

# MT5 TICK FLAGS (For True Institutional Order Flow)
TICK_FLAG_BID = 1
TICK_FLAG_ASK = 2
TICK_FLAG_LAST = 4
TICK_FLAG_VOLUME = 8
TICK_FLAG_BUY = 32
TICK_FLAG_SELL = 64

class TradeDispatcher:
    """
    Handles the reliable transmission of trade orders from the Logic Engine
    to the Execution Engine (Windows Producer) via Redis Streams.
    
    UPDATED: FORCE MARKET EXECUTION ONLY.
    V20.4 FIX: STRICT IMMUTABLE RISK PERSISTENCE FOR SWING TRADING.
    """
    def __init__(self, stream_mgr: RedisStreamManager, pending_tracker: dict[str, Any], lock: threading.RLock):
        self.stream_mgr = stream_mgr
        self.stream_key = CONFIG['redis']['trade_request_stream']
        self.magic_number = CONFIG['trading'].get('magic_number', 621001)
        self.pending_tracker = pending_tracker
        self.lock = lock

    def send_order(self, trade: Trade, estimated_risk_usd: float = 0.0) -> None:
        try:
            # 1. Generate IDs
            order_id = uuid.uuid4()
            short_id = str(order_id)[:8]
            
            # --- V20.4 FIX: PERSIST INITIAL RISK DISTANCE ---
            try:
                initial_risk_dist = abs(trade.entry_price - trade.stop_loss)
                if initial_risk_dist > 0:
                    self.stream_mgr.r.hset("bot:initial_risk", short_id, str(initial_risk_dist))
            except Exception as e:
                logger.error(f"Failed to persist initial risk to Redis for {short_id}: {e}")
            
            # 2. Register Pending (Thread-Safe)
            with self.lock:
                self.pending_tracker[str(order_id)] = {
                    "symbol": trade.symbol,
                    "action": trade.action,
                    "volume": trade.volume,
                    "timestamp": time.time(),
                    "status": "PENDING",
                    "risk_usd": estimated_risk_usd
                }

            # 3. Construct Payload (Strict Protocol Translation)
            if trade.action == "BUY":
                mt5_type = MT5_ORDER_TYPE_BUY
            elif trade.action == "SELL":
                mt5_type = MT5_ORDER_TYPE_SELL
            else:
                mt5_type = MT5_ORDER_TYPE_BUY 
            
            # B. FORCE MARKET EXECUTION
            if trade.action in ["BUY", "SELL"]:
                mt5_action = MT5_TRADE_ACTION_DEAL
                final_price = "0.0" 
            else:
                mt5_action = MT5_TRADE_ACTION_DEAL 
                final_price = "0.0"

            # Ensure Comment length compliance (MT5 limit)
            comment = f"Auto_{short_id}"
            if trade.comment:
                comment = f"{comment}_{trade.comment}"[:31]

            # Determine Action String for Producer Logic
            if trade.action in ["MODIFY", "CLOSE_ALL"]:
                action_payload = str(trade.action)
            else:
                action_payload = str(mt5_action)

            payload = {
                "id": str(order_id),
                "uuid": str(order_id),  
                "symbol": str(trade.symbol),
                
                # --- PROTOCOL TRANSLATION ---
                "action": action_payload,
                "type": str(mt5_type),
                
                # Explicit formatting to prevent float precision errors
                "volume": "{:.2f}".format(float(trade.volume)),
                
                # FORCE ZERO PRICE for Market Execution
                "entry_price": final_price,
                "price": final_price, 
                
                # CRITICAL MAPPING: Producer looks for 'sl' and 'tp', NOT 'stop_loss'
                "sl": "{:.5f}".format(trade.stop_loss),
                "tp": "{:.5f}".format(trade.take_profit),
                
                "magic_number": str(self.magic_number),
                "magic": str(self.magic_number), 
                
                "comment": comment,
                "timestamp": str(time.time()), 
                
                "ticket": str(trade.ticket) if trade.ticket else "0"
            }

            # 4. Transmit to Redis
            self.stream_mgr.r.xadd(self.stream_key, payload, maxlen=50000, approximate=True)
            
            logger.info(
                f"{LogSymbols.UPLOAD} DISPATCH SENT: {trade.action} {trade.symbol} "
                f"| Vol: {trade.volume:.2f} | Market Order (Price=0.0) | Risk: ${estimated_risk_usd:.2f}"
            )
        except Exception as e:
            logger.error(f"{LogSymbols.ERROR} Dispatch Failed for {trade.symbol}: {e}", exc_info=True)
            with self.lock:
                if str(order_id) in self.pending_tracker:
                    del self.pending_tracker[str(order_id)]

    def cleanup_stale_orders(self, ttl_seconds: int = 600, open_positions: Optional[Dict] = None):
        now = time.time()
        to_remove = []
        
        with self.lock:
            for oid, data in self.pending_tracker.items():
                if now - data['timestamp'] > ttl_seconds:
                    
                    is_zombie_match = False
                    if open_positions:
                        short_id = str(oid)[:8]
                        symbol = data['symbol']
                        
                        for pos in open_positions.values():
                            if pos.get('symbol') == symbol and short_id in pos.get('comment', ''):
                                is_zombie_match = True
                                break
                    
                    if is_zombie_match:
                        logger.warning(f"🧟 ZOMBIE MATCH: Clearing stale pending {oid} (Found matching position).")
                        to_remove.append(oid)
                    else:
                        logger.warning(f"🧹 Clearing Stale Pending Order: {oid} ({data['symbol']}) - > {ttl_seconds}s")
                        to_remove.append(oid)
            
            for oid in to_remove:
                if oid in self.pending_tracker:
                    try:
                        short_id = str(oid)[:8]
                        self.stream_mgr.r.hdel("bot:initial_risk", short_id)
                    except Exception:
                        pass
                        
                    del self.pending_tracker[oid]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} pending orders.")


# --- WORKER PROCESS FOR OFF-MAIN-THREAD COMPUTE ---
class LogicWorker(multiprocessing.Process):
    def __init__(self, tick_queue: multiprocessing.Queue, signal_queue: multiprocessing.Queue, 
                 config: Dict, symbols: List[str]):
        super().__init__()
        self.tick_queue = tick_queue
        self.signal_queue = signal_queue
        self.config = config
        self.symbols = symbols
        self.running = True
        
        self.max_latency_strict = 2.0
        self.max_latency_relaxed = 30.0
        self.vol_threshold = 0.0005 

    def run(self):
        setup_logging("LogicWorker")
        worker_log = logging.getLogger("LogicWorker")
        worker_log.info(f"{LogSymbols.ONLINE} Logic Worker Started (PID: {os.getpid()}) | V20.5 Unchoked Protocol Active")
        
        threshold_map = self._calibrate_thresholds(worker_log)
        
        aggregators = {}
        alpha = self.config['data'].get('imbalance_alpha', 0.05)
        
        for sym in self.symbols:
            thresh = threshold_map.get(sym, self.config['data'].get('volume_bar_threshold', 10.0)) 
            aggregators[sym] = AdaptiveImbalanceBarGenerator(
                symbol=sym,
                initial_threshold=thresh,
                alpha=alpha
            )
            
        predictor = MultiAssetPredictor(self.symbols, threshold_map=threshold_map)
        latest_context = defaultdict(dict)
        
        while self.running:
            try:
                try:
                    tick_data = self.tick_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if tick_data == "STOP":
                    worker_log.info("Logic Worker received STOP signal. Shutting down gracefully.")
                    break

                symbol = tick_data.get('symbol')
                if symbol not in self.symbols: 
                    continue

                if not self._check_latency(tick_data, predictor, worker_log):
                    continue

                if 'ctx_d1' in tick_data:
                    try: 
                        latest_context[symbol]['d1'] = json.loads(tick_data['ctx_d1'])
                    except Exception as e: 
                        worker_log.debug(f"Failed to parse D1 context for {symbol}: {e}")
                
                if 'ctx_h4' in tick_data:
                    try: 
                        latest_context[symbol]['h4'] = json.loads(tick_data['ctx_h4'])
                    except Exception as e: 
                        worker_log.debug(f"Failed to parse H4 context for {symbol}: {e}")

                try:
                    price = float(tick_data.get('price', 0.0))
                    volume = float(tick_data.get('volume', 1.0))
                    ts = float(tick_data.get('time', 0.0))
                    if ts > 100_000_000_000: 
                        ts /= 1000.0 
                    
                    bid_vol = float(tick_data.get('bid_vol', 0.0))
                    ask_vol = float(tick_data.get('ask_vol', 0.0))
                except (ValueError, TypeError) as e:
                    worker_log.debug(f"Malformed tick data received: {e}")
                    continue

                # Aggregate
                bar = aggregators[symbol].process_tick(
                    price=price,
                    volume=volume,
                    timestamp=ts,
                    external_buy_vol=bid_vol,
                    external_sell_vol=ask_vol
                )

                if bar:
                    ctx = {
                        'd1': latest_context[symbol].get('d1', {}),
                        'h4': latest_context[symbol].get('h4', {}),
                        'positions': tick_data.get('positions', {}) 
                    }
                    
                    signal = predictor.process_bar(symbol, bar, context_data=ctx)
                    
                    if signal:
                        self.signal_queue.put(signal)
                        if signal.action not in ["HOLD", "WARMUP"]:
                            worker_log.info(f"{LogSymbols.SIGNAL} Generated Signal: {signal.action} {symbol} ({signal.confidence:.2f})")

            except Exception as e:
                worker_log.error(f"Logic Worker Error: {e}", exc_info=True)
        
        worker_log.info("Saving Predictor State before shutdown...")
        predictor.save_state()
        worker_log.info("Logic Worker Shutdown Complete.")

    def _check_latency(self, tick_data: Dict, predictor: MultiAssetPredictor, log: logging.Logger) -> bool:
        try:
            ts = float(tick_data.get('time', 0.0))
            if ts > 100_000_000_000: 
                ts /= 1000.0
            
            now = time.time()
            latency = max(0.0, now - ts)
            symbol = tick_data.get('symbol')
            
            current_vol = 0.001
            if symbol in predictor.feature_engineers:
                current_vol = predictor.feature_engineers[symbol].vol_monitor.get()
            
            limit = self.max_latency_strict if current_vol > self.vol_threshold else self.max_latency_relaxed
            
            if latency > limit:
                if latency > 10.0:
                    log.warning(f"⚠️ Latency Guard: Dropped {symbol} (Lat: {latency:.2f}s > Limit: {limit:.1f}s | Vol: {current_vol:.5f})")
                return False
                
            return True
        except Exception as e:
            log.debug(f"Latency check failed: {e}")
            return True 

    def _calibrate_thresholds(self, log: logging.Logger) -> Dict[str, float]:
        log.info(f"{LogSymbols.TRAINING} Logic Worker: Auto-Calibrating Thresholds...")
        threshold_map = {}
        alpha = self.config['data'].get('imbalance_alpha', 0.05)
        config_thresh = self.config['data'].get('volume_bar_threshold', 10.0) 
        
        for sym in self.symbols:
            try:
                df = load_real_data(sym, n_candles=50000, days=30)
                
                if df is None or df.empty:
                    log.warning(f"No historical data for {sym}. Using default threshold: {config_thresh}")
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
                        
                        if price is None: 
                            continue
                        
                        b_vol = vol / 2
                        s_vol = vol / 2
                        if gen.process_tick(price, vol, ts_val, b_vol, s_vol):
                            bar_count += 1
                    
                    if bar_count >= 500:
                        threshold_map[sym] = current_threshold
                        break
                    else:
                        attempts += 1
                        current_threshold = max(10.0, current_threshold * 0.5)
                        threshold_map[sym] = current_threshold
                
                log.info(f"✅ {sym} Worker Calibrated: {threshold_map[sym]:.1f}")
            except Exception as e:
                log.error(f"Worker Calibration Error {sym}: {e}")
                threshold_map[sym] = config_thresh
        
        return threshold_map


class LiveTradingEngine:
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
        initial_bal = CONFIG['env'].get('initial_balance', 50000.0) 
        max_daily_loss = CONFIG.get('risk_management', {}).get('max_daily_loss_pct', 0.045)
        
        self.ftmo_guard = FTMORiskMonitor(
            initial_balance=initial_bal,
            max_daily_loss_pct=max_daily_loss,
            redis_client=self.stream_mgr.r
        )
        self.portfolio_mgr = PortfolioRiskManager(symbols=self.symbols)
        self.session_guard = SessionGuard()
        
        # --- COMPLIANCE SETUP ---
        self.news_monitor = NewsEventMonitor()
        self.compliance_guard = FTMOComplianceGuard([])
        
        self.calendar_thread = threading.Thread(target=self._calendar_sync_loop, daemon=True)
        self.calendar_thread.start()
        
        # 3. Execution Dispatcher
        self.pending_orders = {} 
        self.lock = threading.RLock()
        self.dispatcher = TradeDispatcher(
            stream_mgr=self.stream_mgr,
            pending_tracker=self.pending_orders,
            lock=self.lock
        )
        
        self.last_dispatch_time = defaultdict(float)
        
        # 4. Logic Worker Setup 
        self.tick_queue = multiprocessing.Queue(maxsize=10000)
        self.signal_queue = multiprocessing.Queue(maxsize=1000)
        
        self.logic_worker = LogicWorker(
            tick_queue=self.tick_queue,
            signal_queue=self.signal_queue,
            config=CONFIG,
            symbols=self.symbols
        )
        
        # 5. Sentiment Engine
        self.global_sentiment = {} 
        if NLP_AVAILABLE:
            self.news_thread = threading.Thread(target=self.fetch_news_loop, daemon=True)
            self.news_thread.start()

        # 6. Performance Monitor (Circuit Breaker & SQN)
        self.stats_lock = threading.Lock()
        self.performance_stats = defaultdict(lambda: deque(maxlen=30))
        self.last_reset_ts = 0.0 
        self.daily_execution_stats = defaultdict(lambda: {'losses': 0, 'pnl': 0.0, 'tickets': set()})
        
        self.max_daily_losses_per_symbol = 10
        
        tz_str = CONFIG['risk_management'].get('risk_timezone', 'Europe/Prague')
        try:
            self.server_tz = pytz.timezone(tz_str)
        except Exception:
            self.server_tz = pytz.timezone('Europe/Prague')

        self._restore_daily_state()
        self._restore_penalty_box_state()

        self.perf_thread = threading.Thread(target=self.fetch_performance_loop, daemon=True)
        self.perf_thread.start()
        
        self.signal_thread = threading.Thread(target=self._signal_listener, daemon=True)
        self.signal_thread.start()

        self.is_warm = False
        self.last_corr_update = time.time()
        self.latest_prices = {}
        
        self.latest_positions = defaultdict(list)
        
        self._inject_fallback_prices()
        
        self.liquidation_triggered_map = {sym: False for sym in CONFIG['trading']['symbols']}
        self.processed_ticks = defaultdict(str) 
        self.ticks_processed = 0 
        self.latest_context = defaultdict(dict) 

        self.vol_gate_conf = CONFIG['online_learning'].get('volatility_gate', {})
        self.use_vol_gate = self.vol_gate_conf.get('enabled', True)
        self.min_atr_spread_ratio = self.vol_gate_conf.get('min_atr_spread_ratio', 1.2)
        self.spread_map = CONFIG.get('forensic_audit', {}).get('spread_pips', {})

        self.known_open_symbols: Set[str] = set()

        self.mgmt_thread = threading.Thread(target=self._manage_active_positions_loop, daemon=True)
        self.mgmt_thread.start()

        # V20.5: Start MT5 Execution Logger Thread (PubSub from Windows)
        self.execution_log_thread = threading.Thread(target=self._execution_logger_loop, daemon=True)
        self.execution_log_thread.start()

    def _inject_fallback_prices(self):
        defaults = {
            "USDJPY": 150.0, "GBPUSD": 1.25, "EURUSD": 1.08,
            "USDCAD": 1.35, "USDCHF": 0.90, "AUDUSD": 0.65, "NZDUSD": 0.60,
            "GBPJPY": 190.0, "EURJPY": 160.0, "AUDJPY": 95.0,
            "GBPAUD": 1.95, "EURAUD": 1.65, "GBPNZD": 2.10
        }
        for sym, price in defaults.items():
            if sym not in self.latest_prices:
                self.latest_prices[sym] = price
        
        logger.info(f"💉 Injected {len(defaults)} fallback prices for PnL conversion.")

    def _get_producer_anchor_timestamp(self) -> float:
        try:
            reset_ts = self.stream_mgr.r.get("risk:last_reset_date")
            if reset_ts:
                return float(reset_ts)
        except Exception:
            pass
        return datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    def _restore_daily_state(self):
        logger.info(f"{LogSymbols.DATABASE} Restoring Daily Execution State (Server Time Sync)...")
        try:
            stream_key = CONFIG['redis'].get('closed_trade_stream_key', 'stream:closed_trades')
            self.last_reset_ts = self._get_producer_anchor_timestamp()
            start_id = int(self.last_reset_ts * 1000)
            messages = self.stream_mgr.r.xrange(stream_key, min=start_id, max='+')
            
            restored_count = 0
            with self.stats_lock:
                self.daily_execution_stats = defaultdict(lambda: {'losses': 0, 'pnl': 0.0, 'tickets': set()})
                
                for msg_id, data in messages:
                    symbol = data.get('symbol')
                    net_pnl = float(data.get('net_pnl', 0.0))
                    timestamp_raw = float(data.get('timestamp', 0))
                    ticket = str(data.get('ticket', '')) 
                    
                    if timestamp_raw < self.last_reset_ts:
                        continue
                        
                    self.daily_execution_stats[symbol]['pnl'] += net_pnl
                    
                    if ticket and ticket in self.daily_execution_stats[symbol]['tickets']:
                        continue
                    
                    if ticket:
                        self.daily_execution_stats[symbol]['tickets'].add(ticket)
                    
                    if net_pnl < 0:
                        self.daily_execution_stats[symbol]['losses'] += 1
                    
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
        logger.info(f"{LogSymbols.DATABASE} Restoring Penalty Box State (Revenge Guard)...")
        cooldown_mins = CONFIG.get('risk_management', {}).get('loss_cooldown_minutes', 60)
        
        now_ts = datetime.now(timezone.utc).timestamp()
        start_ts = now_ts - (cooldown_mins * 60 + 600) 
        start_id = int(start_ts * 1000)
        
        stream_key = CONFIG['redis'].get('closed_trade_stream_key', 'stream:closed_trades')
        
        try:
            messages = self.stream_mgr.r.xrange(stream_key, min=start_id, max='+')
            restored_count = 0
            
            for _, data in messages:
                symbol = data.get('symbol')
                net_pnl = float(data.get('net_pnl', 0.0))
                close_ts = float(data.get('timestamp', 0.0))
                
                if net_pnl < 0:
                    expiry = close_ts + (cooldown_mins * 60)
                    if expiry > now_ts:
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
        logger.info(f"{LogSymbols.INFO} Performance Monitor (SQN & Circuit Breaker) Thread Started.")
        stream_key = CONFIG['redis'].get('closed_trade_stream_key', 'stream:closed_trades')
        
        last_id = '$' 
        cooldown_mins = CONFIG.get('risk_management', {}).get('loss_cooldown_minutes', 60)

        while not self.shutdown_flag:
            try:
                current_anchor = self._get_producer_anchor_timestamp()
                if current_anchor > self.last_reset_ts:
                    with self.stats_lock:
                        logger.info(f"⚓ DAILY RESET DETECTED in Monitor (New Anchor: {current_anchor})")
                        self.last_reset_ts = current_anchor
                        self.daily_execution_stats.clear()
                
                response = self.stream_mgr.r.xread({stream_key: last_id}, count=10, block=1000)
                
                if response:
                    for stream, messages in response:
                        for msg_id, data in messages:
                            last_id = msg_id
                            symbol = data.get('symbol')
                            net_pnl = float(data.get('net_pnl', 0.0))
                            ticket = str(data.get('ticket', ''))
                            ts_raw = float(data.get('timestamp', 0))
                            
                            if ts_raw < self.last_reset_ts:
                                continue
                            
                            if symbol:
                                with self.stats_lock:
                                    self.performance_stats[symbol].append(net_pnl)
                                    
                                    if ticket and ticket not in self.daily_execution_stats[symbol]['tickets']:
                                        self.daily_execution_stats[symbol]['pnl'] += net_pnl
                                        if ticket:
                                            self.daily_execution_stats[symbol]['tickets'].add(ticket)
                                        
                                        if net_pnl < 0:
                                            self.daily_execution_stats[symbol]['losses'] += 1
                                            logger.info(f"📉 LOSS DETECTED {symbol}: ${net_pnl:.2f} | Daily Losses: {self.daily_execution_stats[symbol]['losses']}")
                                        else:
                                            logger.info(f"💰 PROFIT DETECTED {symbol}: ${net_pnl:.2f}")

                                    self.portfolio_mgr.add_to_penalty_box(symbol, duration_minutes=cooldown_mins)
                                    
                                    if net_pnl < 0:
                                        logger.warning(f"🚫 {symbol} REVENGE GUARD: Locked for {cooldown_mins}m (Loss)")
                                    else:
                                        logger.info(f"🛡️ {symbol} MANDATORY COOLDOWN: Locked for {cooldown_mins}m (Win/Break-even)")
            
            except Exception as e:
                logger.error(f"Performance Monitor Error: {e}")
                time.sleep(5)

    def _calculate_sqn(self, symbol: str) -> float:
        with self.stats_lock:
            trades = list(self.performance_stats[symbol])
            
        if len(trades) < 5: 
            return 0.0 
            
        avg_pnl = np.mean(trades)
        std_pnl = np.std(trades)
        
        if std_pnl < 1e-9:
            return 0.0
            
        sqn = (avg_pnl / std_pnl) * math.sqrt(len(trades))
        return sqn
        
    def _check_circuit_breaker(self, symbol: str) -> bool:
        current_anchor = self._get_producer_anchor_timestamp()
        
        with self.stats_lock:
            if current_anchor > self.last_reset_ts:
                logger.info(f"⚓ LAZY DAILY RESET for {symbol} (Anchor: {current_anchor})")
                self.last_reset_ts = current_anchor
                self.daily_execution_stats.clear()
                return False
                
            stats = self.daily_execution_stats[symbol]
            losses = stats['losses']
            pnl = stats['pnl']

        if losses >= self.max_daily_losses_per_symbol:
            return True
            
        current_equity = self.ftmo_guard.equity
        if current_equity > 0:
            limit = current_equity * 0.01
            if pnl < -limit:
                return True
                
        return False

    def _get_open_positions_from_redis(self) -> Dict[str, List[Dict]]:
        magic = CONFIG['trading']['magic_number']
        key = f"{CONFIG['redis']['position_state_key_prefix']}:{magic}"
        
        try:
            data = self.stream_mgr.r.get(key)
            if not data: return {}
            
            pos_list = json.loads(data)
            pos_map = defaultdict(list)
            
            for p in pos_list:
                sym = p.get('symbol')
                if sym:
                    pos_map[sym].append(p)
            
            self.latest_positions = dict(pos_map)
            return dict(pos_map)
        except Exception as e:
            logger.debug(f"Failed to fetch positions from redis: {e}")
            return {}

    def _check_symbol_concurrency(self, symbol: str) -> bool:
        magic = CONFIG['trading']['magic_number']
        key = f"{CONFIG['redis']['position_state_key_prefix']}:{magic}"
        
        total_open_tickets = 0
        symbol_open_tickets = 0
        
        try:
            data = self.stream_mgr.r.get(key)
            if data:
                pos_list = json.loads(data)
                total_open_tickets = len(pos_list)
                symbol_open_tickets = sum(1 for p in pos_list if p.get('symbol') == symbol)
        except Exception as e:
            logger.debug(f"Concurrency check redis fetch failed: {e}")

        with self.lock:
            local_pending_total = len(self.pending_orders)
            local_pending_symbol = sum(1 for o in self.pending_orders.values() if o.get('symbol') == symbol)
            
        total_open_tickets += local_pending_total
        symbol_open_tickets += local_pending_symbol

        # V20.5 FIX: Remove arbitrary global limits that choke scale. Fallback to 100.
        config_max = CONFIG.get('risk_management', {}).get('max_open_trades', 100)
        max_global = config_max

        if total_open_tickets >= max_global:
            logger.warning(f"🚫 Global Trade Limit Reached ({total_open_tickets}/{max_global}). Margin Protected.")
            return False

        # V20.5 FIX: STRICT 1 TRADE PER PAIR
        # We completely bypass the buggy pyramiding loop. If 1 trade exists, block new ones.
        if symbol_open_tickets >= 1:
            # Silently drop to avoid console spam during high-frequency signals on an open pair
            return False
            
        return True

    def _signal_listener(self):
        logger.info(f"{LogSymbols.SIGNAL} Signal Dispatcher Thread Active.")
        
        while not self.shutdown_flag:
            try:
                try:
                    signal = self.signal_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if signal.action in ["HOLD", "WARMUP"]:
                    continue
                    
                if signal.action not in ["BUY", "SELL"]:
                    continue
                
                symbol = signal.symbol
                
                if not self._check_risk_gates(symbol):
                    continue
                
                ttl_sec = CONFIG['trading'].get('limit_order_ttl_seconds', 60)
                if time.time() - self.last_dispatch_time.get(symbol, 0) < ttl_sec:
                    continue
                
                if not self._check_symbol_concurrency(symbol):
                    continue

                open_positions = self._get_open_positions_from_redis()
                
                current_atr = signal.meta_data.get('atr', 0.001)
                ker_val = signal.meta_data.get('ker', 1.0)
                volatility = signal.meta_data.get('volatility', 0.001)
                parkinson_vol = signal.meta_data.get('parkinson_vol', 0.0) 
                risk_percent_override = signal.meta_data.get('risk_percent_override')
                
                sqn_score = self._calculate_sqn(symbol)
                
                try:
                    start_eq = float(self.stream_mgr.r.get(CONFIG['redis']['risk_keys']['daily_starting_equity']) or 0.0)
                    if start_eq > 0:
                        daily_pnl_pct = (self.ftmo_guard.equity - start_eq) / start_eq
                    else:
                        daily_pnl_pct = 0.0
                except Exception as e:
                    logger.debug(f"Failed to calculate daily PnL pct: {e}")
                    daily_pnl_pct = 0.0
                
                active_corrs = self.portfolio_mgr.get_correlation_count(symbol)
                
                current_price = self.latest_prices.get(symbol, 0.0)
                if current_price <= 0:
                    logger.warning(f"⚠️ {symbol}: No price data for execution. Skipping.")
                    continue

                estimated_free_margin = 50000.0 
                try:
                    acc_info = self.stream_mgr.r.hgetall(CONFIG['redis']['account_info_key'])
                    if acc_info:
                        estimated_free_margin = float(acc_info.get('free_margin', 50000.0))
                except Exception as e:
                    logger.debug(f"Failed to fetch free margin: {e}")
                    pass

                trade_intent, risk_usd = RiskManager.calculate_rck_size(
                    context=TradeContext(
                        symbol=symbol,
                        price=current_price,
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
                    current_open_risk_pct=0.0, 
                    free_margin=estimated_free_margin, 
                    parkinson_vol=parkinson_vol 
                )
                
                if trade_intent.volume <= 0: 
                    continue
                
                trade_intent.action = signal.action
                entry_ref = current_price
                trade_intent.entry_price = entry_ref
                
                if signal.action == "BUY":
                    trade_intent.stop_loss = entry_ref - trade_intent.stop_loss
                    trade_intent.take_profit = entry_ref + trade_intent.take_profit
                else:  # SELL
                    trade_intent.stop_loss = entry_ref + trade_intent.stop_loss
                    trade_intent.take_profit = entry_ref - trade_intent.take_profit
                    
                if trade_intent.stop_loss <= 0 or trade_intent.take_profit <= 0:
                    logger.error(f"🛑 INVALID SL/TP CALCULATION: {symbol} Action:{signal.action} Price:{entry_ref} SL:{trade_intent.stop_loss} TP:{trade_intent.take_profit}. Trade Blocked.")
                    continue
                
                logger.info(f"📤 Dispatching MARKET {trade_intent.action} {symbol} ({trade_intent.volume} lots) @ ~{entry_ref:.5f} | SL: {trade_intent.stop_loss:.5f} | TP: {trade_intent.take_profit:.5f}")
                self.dispatcher.send_order(trade_intent, risk_usd)
                self.last_dispatch_time[symbol] = time.time()
                
            except Exception as e:
                logger.error(f"Signal Listener Error: {e}", exc_info=True)

    def _manage_active_positions_loop(self):
        logger.info(f"{LogSymbols.INFO} Active Position Manager Started.")
        
        tbm_conf = CONFIG.get('online_learning', {}).get('tbm', {})
        horizon_minutes = int(tbm_conf.get('horizon_minutes', 720)) 
        hard_stop_seconds = horizon_minutes * 60
        logger.info(f"{LogSymbols.TIME} Dynamic Time Stop Set to: {horizon_minutes} minutes ({horizon_minutes/60:.1f}h)")

        while not self.shutdown_flag:
            try:
                positions_map = self._get_open_positions_from_redis()
                if not positions_map:
                    time.sleep(5)
                    continue

                now_utc = datetime.now(pytz.utc)
                session_liquidation = self.session_guard.should_liquidate(timestamp=now_utc)

                for sym, pos_list in positions_map.items():
                    for pos in pos_list:
                        exit_reason = None

                        if session_liquidation:
                            exit_reason = "Session End Guard (3h pre-close)"
                            logger.warning(f"⌛ SESSION LIQUIDATION: {sym} closed by background manager.")

                        if not exit_reason:
                            entry_ts = float(pos.get('time', 0))
                            if entry_ts > 0:
                                entry_dt = datetime.fromtimestamp(entry_ts, pytz.utc)
                                duration = (now_utc - entry_dt).total_seconds()
                                
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

                        current_price = self.latest_prices.get(sym, 0.0)
                        if current_price <= 0: continue

                        entry_price = float(pos.get('entry_price', 0.0))
                        sl_price = float(pos.get('sl', 0.0))
                        tp_price = float(pos.get('tp', 0.0))
                        pos_type = pos.get('type') 
                        ticket = pos.get('ticket')
                        comment = str(pos.get('comment', ''))

                        if entry_price == 0 or sl_price == 0: continue

                        risk_dist = 0.0
                        try:
                            parts = comment.split('_')
                            if len(parts) >= 2 and parts[0] == "Auto":
                                short_id = parts[1]
                                stored_risk = self.stream_mgr.r.hget("bot:initial_risk", short_id)
                                if stored_risk:
                                    risk_dist = float(stored_risk)
                        except Exception as e:
                            logger.warning(f"Failed to fetch initial risk from Redis for {sym}: {e}")

                        if risk_dist <= 1e-5:
                            risk_dist = abs(entry_price - sl_price)
                            if risk_dist < 1e-5: continue
                        
                        trail_conf = CONFIG.get('risk_management', {}).get('trailing_stop', {})
                        activation_r = float(trail_conf.get('activation_r', 1.5))
                        trail_dist_r = float(trail_conf.get('trail_dist_r', 0.5))

                        new_sl = None
                        
                        if pos_type == "BUY":
                            dist_pnl = current_price - entry_price
                            r_multiple = dist_pnl / risk_dist
                            
                            if r_multiple >= activation_r:
                                trail_pips = risk_dist * trail_dist_r
                                target_sl = current_price - trail_pips
                                
                                min_lock = entry_price + (risk_dist * 0.1)
                                target_sl = max(target_sl, min_lock)
                                
                                if target_sl > sl_price:
                                    new_sl = target_sl

                        elif pos_type == "SELL":
                            dist_pnl = entry_price - current_price
                            r_multiple = dist_pnl / risk_dist
                            
                            if r_multiple >= activation_r:
                                trail_pips = risk_dist * trail_dist_r
                                target_sl = current_price + trail_pips
                                
                                min_lock = entry_price - (risk_dist * 0.1)
                                target_sl = min(target_sl, min_lock)
                                
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
                                take_profit=tp_price,
                                ticket=ticket,
                                comment=f"Trail {r_multiple:.1f}R"
                            )
                            self.dispatcher.send_order(modify_intent, 0.0)

            except Exception as e:
                logger.error(f"Position Manager Error: {e}", exc_info=True)
            
            time.sleep(1) 

    def _reconcile_pending_orders(self, open_positions_map: Dict[str, List[Dict]]):
        try:
            flat_positions = {}
            if open_positions_map:
                for sym, pos_list in open_positions_map.items():
                    for p in pos_list:
                        ticket = p.get('ticket', str(uuid.uuid4()))
                        flat_positions[str(ticket)] = p
            
            self.dispatcher.cleanup_stale_orders(ttl_seconds=600, open_positions=flat_positions)
        except Exception as e:
            logger.debug(f"Pending order reconciliation failed: {e}")

    def process_tick(self, tick_data: dict):
        try:
            symbol = tick_data.get('symbol')
            if not symbol: return
            
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
            
            ts = float(tick_data.get('time', 0.0))
            vol = float(tick_data.get('volume', 0.0))
            
            flags = int(tick_data.get('flags', 0))
            
            bid_vol = 0.0
            ask_vol = 0.0
            
            if flags > 0:
                if flags & TICK_FLAG_BUY:
                    bid_vol = vol
                elif flags & TICK_FLAG_SELL:
                    ask_vol = vol
                else:
                    if 'bid_vol' in tick_data and 'ask_vol' in tick_data:
                        bid_vol = float(tick_data['bid_vol'])
                        ask_vol = float(tick_data['ask_vol'])
            else:
                if 'bid_vol' in tick_data and 'ask_vol' in tick_data:
                    bid_vol = float(tick_data['bid_vol'])
                    ask_vol = float(tick_data['ask_vol'])

            tick_hash = f"{ts}_{price}_{vol}_{flags}"
            
            if tick_hash == self.processed_ticks[symbol]:
                return 
                
            self.processed_ticks[symbol] = tick_hash
            
            self.ticks_processed += 1
            if self.ticks_processed % 1000 == 0:
                logger.info(f"⚡ HEARTBEAT: Processed {self.ticks_processed} ticks... (Last: {symbol})")

            tick_data['bid_vol'] = bid_vol
            tick_data['ask_vol'] = ask_vol
            
            legacy_positions = {}
            for s, p_list in self.latest_positions.items():
                if p_list: legacy_positions[s] = p_list[-1]
            tick_data['positions'] = legacy_positions
            
            try:
                self.tick_queue.put(tick_data, block=False)
            except queue.Full:
                logger.warning("⚠️ Tick Queue Full! Dropping tick to maintain real-time edge.")
                
        except Exception as e:
            logger.error(f"Main Process Tick Error: {e}", exc_info=True)

    def _check_risk_gates(self, symbol: str) -> bool:
        try:
            if self.stream_mgr.r.exists(CONFIG['redis']['risk_keys']['midnight_freeze']):
                logger.warning(f"{LogSymbols.FROZEN} Midnight Freeze Active. Holding for Daily Anchor.")
                return False

            if not self.stream_mgr.r.exists(CONFIG['redis']['risk_keys']['daily_starting_equity']):
                logger.warning(f"{LogSymbols.LOCK} Daily Anchor Missing. Waiting for Producer Sync.")
                return False

            if not self.ftmo_guard.can_trade():
                log_msg = self.ftmo_guard.check_circuit_breakers() if hasattr(self.ftmo_guard, 'check_circuit_breakers') else "Circuit Breaker Tripped"
                logger.warning(f"{LogSymbols.LOCK} FTMO Guard Halted: {log_msg}")
                return False

            if not self.session_guard.is_trading_allowed():
                srv_time = datetime.now(self.session_guard.market_tz)
                logger.warning(f"{LogSymbols.LOCK} Session Guard Block: Server Time {srv_time.strftime('%H:%M')} (TZ: {self.session_guard.market_tz}) < Start Hour {self.session_guard.start_hour} or > End {self.session_guard.liq_hour}")
                return False
                
            if self.session_guard.is_friday_afternoon():
                logger.warning(f"{LogSymbols.LOCK} Friday Guard Block: No new trades allowed late Friday (NY Noon).")
                return False

            if self.portfolio_mgr.check_penalty_box(symbol):
                logger.warning(f"{LogSymbols.LOCK} {symbol} is in Penalty Box (Cool-down Active).")
                return False

            if not self.compliance_guard.check_trade_permission(symbol):
                 return False

            acc_info = self.stream_mgr.r.hgetall(CONFIG['redis']['account_info_key'])
            if acc_info:
                equity = float(acc_info.get('equity', 0))
                margin = float(acc_info.get('margin', 0))
                
                if margin > 0:
                    margin_level = (equity / margin) * 100.0
                    min_margin_level = CONFIG.get('risk_management', {}).get('min_margin_level_percent', 200.0)
                    
                    if margin_level < min_margin_level:
                        logger.critical(f"🛑 MARGIN LEVEL CRITICAL: {margin_level:.2f}% < {min_margin_level}%. BLOCKING TRADES.")
                        return False
            return True
        except Exception as e:
            logger.warning(f"Risk Gate Check Failed: {e}", exc_info=True)
            return False

    def run(self):
        logger.info(f"{LogSymbols.SUCCESS} Engine Loop Started. Waiting for Ticks on '{CONFIG['redis']['price_data_stream']}'...")
        self.is_warm = True
        
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
                if not self.logic_worker.is_alive():
                    logger.critical("🚨 Logic Worker DIED! Restarting...")
                    self.logic_worker = LogicWorker(self.tick_queue, self.signal_queue, CONFIG, self.symbols)
                    self.logic_worker.start()

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
                
                open_positions_map = self._get_open_positions_from_redis()
                self._reconcile_pending_orders(open_positions_map)
                
                try:
                    cached_eq = self.stream_mgr.r.get(CONFIG['redis']['risk_keys']['current_equity'])
                    if cached_eq and float(cached_eq) > 0:
                        self.ftmo_guard.update_equity(float(cached_eq))

                    cached_anchor = self.stream_mgr.r.get(CONFIG['redis']['risk_keys']['daily_starting_equity'])
                    if cached_anchor and float(cached_anchor) > 0:
                        self.ftmo_guard.starting_equity_of_day = float(cached_anchor)
                    
                    cached_size = self.stream_mgr.r.get("bot:account_size")
                    if cached_size and float(cached_size) > 0:
                        self.ftmo_guard.initial_balance = float(cached_size)
                        
                except Exception as e:
                    pass
                
            except Exception as e:
                logger.error(f"{LogSymbols.ERROR} Stream Read Error: {e}", exc_info=True)
                time.sleep(1)

    def shutdown(self) -> None:
        logger.info(f"{LogSymbols.CLOSE} Engine Shutting Down...")
        self.shutdown_flag = True
        
        try:
            self.tick_queue.put("STOP")
            
            if self.logic_worker.pid is not None:
                self.logic_worker.join(timeout=5)
                if self.logic_worker.is_alive():
                    self.logic_worker.terminate()
        except AssertionError:
            pass 
        except Exception as e:
            logger.error(f"Error terminating logic worker: {e}")
            
        logger.info("Logic Worker Terminated.")

    def _execution_logger_loop(self):
        """
        V20.5: Listens to Redis PubSub for real-time trade execution updates from Windows Producer.
        Brings MT5 terminal visibility directly into the Linux VS Code console.
        """
        logger.info(f"{LogSymbols.ONLINE} MT5 Execution Logger Thread Started.")
        try:
            pubsub = self.stream_mgr.r.pubsub()
            pubsub.subscribe(['order_filled_channel', 'order_failed_channel'])
            
            for message in pubsub.listen():
                if self.shutdown_flag:
                    break
                    
                if message['type'] == 'message':
                    try:
                        if isinstance(message['data'], bytes):
                            data_str = message['data'].decode('utf-8')
                        else:
                            data_str = message['data']
                            
                        data = json.loads(data_str)
                        channel = message['channel']
                        
                        if isinstance(channel, bytes):
                            channel = channel.decode('utf-8')
                        
                        if channel == 'order_filled_channel':
                            logger.info(f"{LogSymbols.SUCCESS} [MT5 TERMINAL] ORDER FILLED: {data.get('symbol')} | Ticket: {data.get('ticket')}")
                        elif channel == 'order_failed_channel':
                            logger.error(f"{LogSymbols.ERROR} [MT5 TERMINAL] ORDER FAILED: {data.get('symbol')} | Reason: {data.get('reason')}")
                    except Exception as e:
                        logger.debug(f"Failed to parse MT5 execution message: {e}")
        except Exception as e:
            logger.error(f"Execution Logger Loop Error: {e}")

if __name__ == "__main__":
    engine = LiveTradingEngine()
    try:
        engine.run()
    except KeyboardInterrupt:
        engine.shutdown()