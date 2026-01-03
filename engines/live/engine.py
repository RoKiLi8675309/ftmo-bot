# =============================================================================
# FILENAME: engines/live/engine.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/live/engine.py
# DEPENDENCIES: shared, engines.live.dispatcher, engines.live.predictor
# DESCRIPTION: Core Event Loop. Ingests ticks, aggregates Tick Imbalance Bars (TIBs),
# and generates signals via the Golden Trio Predictor.
#
# PHOENIX STRATEGY V12.4 (LIVE ENGINE - SNIPER MODE):
# 1. LOGIC: Added "Stalemate Exit" (4h) to active position management.
# 2. RISK: Strict adherence to V12.4 FTMO Limits and 0.50% base risk.
# 3. ASSETS: Optimized for High-Vol pairs (EURUSD, GBPUSD, JPY pairs).
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
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Any, Optional, Dict

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
MAX_TICK_LATENCY_SEC = 3.0  # Threshold for warning about NTP/Clock drift

class LiveTradingEngine:
    """
    The Central Logic Unit for the Linux Consumer.
    1. Consumes Ticks from Redis.
    2. Aggregates Ticks into Adaptive Imbalance Bars (TIBs).
    3. Feeds Bars to Golden Trio Predictor (V12.4 Logic).
    4. Manages Active Positions (Time Stop / Stalemate / Trailing).
    5. Dispatches Orders to Windows.
    """
    def __init__(self):
        self.shutdown_flag = False
        
        # 1. Infrastructure
        self.stream_mgr = RedisStreamManager(
            host=CONFIG['redis']['host'],
            port=CONFIG['redis']['port'],
            db=0,
            decode_responses=True
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
        self.news_monitor = NewsEventMonitor()
        
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
        # Tracks last 30 trades per symbol to calculate SQN
        self.performance_stats = defaultdict(lambda: deque(maxlen=30))
        
        # V10.0: Daily Circuit Breaker State (Realized Execution)
        # {symbol: {'date': date_obj, 'losses': int, 'pnl': float}}
        self.daily_execution_stats = defaultdict(lambda: {'date': None, 'losses': 0, 'pnl': 0.0})
        self.max_daily_losses_per_symbol = 2
        
        self.perf_thread = threading.Thread(target=self.fetch_performance_loop, daemon=True)
        self.perf_thread.start()

        # State
        self.is_warm = False
        self.last_corr_update = time.time()
        self.latest_prices = {}
        self.liquidation_triggered_map = {sym: False for sym in CONFIG['trading']['symbols']}
        
        # --- CONTEXT CACHE (D1/H4 from Windows) ---
        self.latest_context = defaultdict(dict) # {symbol: {'d1': {}, 'h4': {}}}

        # --- Volatility Gate Config ---
        self.vol_gate_conf = CONFIG['online_learning'].get('volatility_gate', {})
        self.use_vol_gate = self.vol_gate_conf.get('enabled', True)
        self.min_atr_spread_ratio = self.vol_gate_conf.get('min_atr_spread_ratio', 1.5)
        self.spread_map = CONFIG.get('forensic_audit', {}).get('spread_pips', {})
        
        # Timezone for Circuit Breaker Reset
        tz_str = CONFIG['risk_management'].get('risk_timezone', 'Europe/Prague')
        try:
            self.server_tz = pytz.timezone(tz_str)
        except Exception:
            self.server_tz = pytz.timezone('Europe/Prague')

        # V11.1: Active Position Management Thread
        self.mgmt_thread = threading.Thread(target=self._manage_active_positions_loop, daemon=True)
        self.mgmt_thread.start()

    def fetch_news_loop(self):
        """
        Background thread to fetch news and update sentiment scores.
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
        """
        logger.info(f"{LogSymbols.INFO} Performance Monitor (SQN & Circuit Breaker) Thread Started.")
        stream_key = CONFIG['redis'].get('closed_trade_stream_key', 'stream:closed_trades')
        
        # Start reading from current time ($)
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
                            
                            if symbol:
                                # 1. SQN Window Update
                                self.performance_stats[symbol].append(net_pnl)
                                
                                # 2. V10.0 Circuit Breaker Update
                                now_server = datetime.now(self.server_tz).date()
                                
                                # Reset if new day
                                if self.daily_execution_stats[symbol]['date'] != now_server:
                                    self.daily_execution_stats[symbol] = {'date': now_server, 'losses': 0, 'pnl': 0.0}
                                
                                self.daily_execution_stats[symbol]['pnl'] += net_pnl
                                
                                if net_pnl < 0:
                                    self.daily_execution_stats[symbol]['losses'] += 1
                                    logger.info(f"📉 LOSS DETECTED {symbol}: ${net_pnl:.2f} | Daily Losses: {self.daily_execution_stats[symbol]['losses']}")
                                
            except Exception as e:
                logger.error(f"Performance Monitor Error: {e}")
                time.sleep(5)

    def _calculate_sqn(self, symbol: str) -> float:
        """
        Calculates the rolling System Quality Number (SQN) for a symbol.
        SQN = (Mean PnL / Std Dev PnL) * Sqrt(N)
        """
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
        stats = self.daily_execution_stats[symbol]
        
        # Auto-reset if accessed on a new day before trade loop hits it
        if stats['date'] != now_server:
            self.daily_execution_stats[symbol] = {'date': now_server, 'losses': 0, 'pnl': 0.0}
            return False
            
        # Rule 1: Max 2 Losses per day
        if stats['losses'] >= self.max_daily_losses_per_symbol:
            return True
            
        # Rule 2: Daily PnL < -1% of Equity
        # Estimate equity roughly from risk monitor
        current_equity = self.ftmo_guard.equity
        if current_equity > 0:
            limit = current_equity * 0.01
            if stats['pnl'] < -limit:
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
        V12.4: Active Position Management Thread.
        Enforces:
        1. 24h Time Stop (Hard Exit).
        2. 4h Stalemate Exit (Exit if PnL between -0.5R and +0.5R).
        3. 0.5R Trailing Stop.
        """
        logger.info(f"{LogSymbols.INFO} Active Position Manager Started (Time-Stop: 24h, Stalemate: 4h, Trail: 0.5R).")
        while not self.shutdown_flag:
            try:
                positions = self._get_open_positions_from_redis()
                if not positions:
                    time.sleep(5)
                    continue

                now_utc = datetime.now(pytz.utc)
                hard_stop_seconds = 86400 # 24 hours
                stalemate_seconds = 14400 # 4 hours

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
                        
                        # 2. Stalemate Exit (4h)
                        elif duration > stalemate_seconds:
                            current_price = self.latest_prices.get(sym, 0.0)
                            entry_price = float(pos.get('entry_price', 0.0))
                            sl_price = float(pos.get('sl', 0.0))
                            
                            if current_price > 0 and entry_price > 0 and sl_price > 0:
                                risk_dist = abs(entry_price - sl_price)
                                if risk_dist > 1e-5:
                                    if pos.get('type') == "BUY":
                                        pnl_dist = current_price - entry_price
                                    else:
                                        pnl_dist = entry_price - current_price
                                    
                                    r_val = pnl_dist / risk_dist
                                    
                                    # If stuck between -0.5R and +0.5R -> Kill it
                                    if -0.5 <= r_val <= 0.5:
                                        exit_reason = "Stalemate (4h)"
                                        logger.info(f"⌛ STALEMATE: {sym} stuck at {r_val:.2f}R for {duration/3600:.1f}h. Freeing capital.")

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
        """
        try:
            symbol = tick_data.get('symbol')
            if symbol not in self.aggregators: return
            
            # --- CLOCK SKEW CHECK ---
            tick_ts = float(tick_data.get('time', 0.0))
            now_ts = time.time()
            latency = now_ts - tick_ts
            
            if abs(latency) > MAX_TICK_LATENCY_SEC:
                logger.warning(
                    f"{LogSymbols.TIME} CLOCK SKEW DETECTED: {symbol} Tick Drift {latency:.2f}s "
                    f"(Threshold: {MAX_TICK_LATENCY_SEC}s). Verify NTP sync."
                )

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

            volume = float(tick_data.get('volume', 1.0))
            bid_vol = float(tick_data.get('bid_vol', 0.0))
            ask_vol = float(tick_data.get('ask_vol', 0.0))

            self.latest_prices[symbol] = price
            
            # --- CONTEXT EXTRACTION ---
            if 'ctx_d1' in tick_data:
                try: self.latest_context[symbol]['d1'] = json.loads(tick_data['ctx_d1'])
                except: pass
            
            if 'ctx_h4' in tick_data:
                try: self.latest_context[symbol]['h4'] = json.loads(tick_data['ctx_h4'])
                except: pass

            # 1. Feed Adaptive Aggregator
            # Returns a VolumeBar (TIB) if imbalance threshold is met
            bar = self.aggregators[symbol].process_tick(
                price=price, 
                volume=volume, 
                timestamp=tick_ts, 
                external_buy_vol=bid_vol, 
                external_sell_vol=ask_vol
            )
            
            # 2. If Bar Complete -> Predict
            if bar:
                self.on_bar_complete(bar)
                
        except Exception as e:
            logger.error(f"Tick Processing Error: {e}")

    def on_bar_complete(self, bar: VolumeBar):
        """
        Triggered when a Tick Imbalance Bar (TIB) is closed.
        Executes the Prediction and Dispatch logic.
        """
        # --- V10.0 EXECUTION CIRCUIT BREAKER ---
        if self._check_circuit_breaker(bar.symbol):
            return

        # 1. Update Portfolio Risk State
        try:
            ret = (bar.close - bar.open) / bar.open if bar.open > 0 else 0.0
            self.portfolio_mgr.update_returns(bar.symbol, ret)
            
            if time.time() - self.last_corr_update > 60:
                self.portfolio_mgr.update_correlation_matrix()
                self.last_corr_update = time.time()
        except Exception as e:
            logger.error(f"Portfolio Update Error: {e}")
        
        # 2. Prepare Context
        current_sentiment = self.global_sentiment.get('GLOBAL', 0.0)
        mt5_context = self.latest_context.get(bar.symbol, {})
        open_positions = self._get_open_positions_from_redis()
        
        context_data = {
            'd1': mt5_context.get('d1', {}),
            'h4': mt5_context.get('h4', {}),
            'sentiment': current_sentiment,
            'positions': open_positions
        }
        
        # 3. Get Signal (Golden Trio / ARF)
        signal = self.predictor.process_bar(bar.symbol, bar, context_data=context_data)
        
        if not signal: return

        # --- SECTION 9: FRIDAY LIQUIDATION CHECK ---
        # Hard Close at 21:00 Server Time
        if self.session_guard.should_liquidate():
            if not self.liquidation_triggered_map[bar.symbol]:
                logger.warning(f"{LogSymbols.CLOSE} FRIDAY LIQUIDATION: Closing all {bar.symbol} trades.")
                close_intent = Trade(
                    symbol=bar.symbol, 
                    action="CLOSE_ALL", 
                    volume=0.0, 
                    entry_price=0.0, 
                    stop_loss=0.0, 
                    take_profit=0.0, 
                    comment="Friday Liquidation"
                )
                self.dispatcher.send_order(close_intent, 0.0)
                self.liquidation_triggered_map[bar.symbol] = True
            return 
        else:
            if self.liquidation_triggered_map[bar.symbol]:
                self.liquidation_triggered_map[bar.symbol] = False

        # --- PHASE 2: WARM-UP GATE ---
        if signal.action == "WARMUP":
            return

        # 4. Validate Signal
        if signal.action == "HOLD":
            return
            
        logger.info(f"{LogSymbols.SIGNAL} SIGNAL: {signal.action} {bar.symbol} (Conf: {signal.confidence:.2f})")

        # --- VOLATILITY GATE ---
        current_atr = signal.meta_data.get('atr', 0.0)
        if self.use_vol_gate:
            pip_size, _ = RiskManager.get_pip_info(bar.symbol)
            spread_pips = self.spread_map.get(bar.symbol, 1.5)
            spread_cost = spread_pips * pip_size
            
            if current_atr < (spread_cost * self.min_atr_spread_ratio):
                logger.warning(f"{LogSymbols.LOCK} Vol Gate: {bar.symbol} Rejected. Low Volatility.")
                return

        # 5. Check Risk & Compliance Gates
        # V10.0: Includes strict Midnight Anchor Check
        if not self._check_risk_gates(bar.symbol):
            return

        # 6. Calculate Size (Fixed Risk Mode)
        volatility = signal.meta_data.get('volatility', 0.001)
        active_corrs = self.portfolio_mgr.get_correlation_count(
            bar.symbol, 
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
        sqn_score = self._calculate_sqn(bar.symbol)

        # Initial Context
        ctx = TradeContext(
            symbol=bar.symbol,
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
            daily_pnl_pct=daily_pnl_pct # V12.0 Pass PnL
        )

        if trade_intent.volume <= 0:
            logger.warning(f"Trade Size 0 for {bar.symbol} (Risk Constraints: {trade_intent.comment}).")
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
            logger.info(f"🛡️ STOP TIGHTENED: {bar.symbol} (High Volatility) -> SL Dist {sl_dist:.5f}")

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
            logger.warning(f"{LogSymbols.LOCK} FTMO Guard: Trading Halted (Drawdown).")
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
        if not self.news_monitor.check_trade_permission(symbol):
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
                
                # Sync Equity for Risk Guard
                try:
                    cached_eq = self.stream_mgr.r.get(CONFIG['redis']['risk_keys']['current_equity'])
                    if cached_eq:
                        self.ftmo_guard.update_equity(float(cached_eq))
                except: pass
                
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