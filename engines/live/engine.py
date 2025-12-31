# =============================================================================
# FILENAME: engines/live/engine.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/live/engine.py
# DEPENDENCIES: shared, engines.live.dispatcher, engines.live.predictor
# DESCRIPTION: Core Event Loop. Ingests ticks, aggregates bars, generates signals.
#
# PHOENIX STRATEGY V7.0 (LIVE EXECUTION):
# 1. AGGRESSOR DATA: Ensures buy_vol/sell_vol are passed to Predictor.
# 2. FRIDAY GUARD: Enforces "No New Entries" after 16:00 (Section 8.2).
# 3. LIQUIDATION: Hard closes all positions at 21:00 (Section 9).
# =============================================================================
import logging
import time
import json
import threading
import signal
import sys
from datetime import datetime
from collections import defaultdict
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
    VolumeBarAggregator,
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
    2. Aggregates Ticks into Volume Bars.
    3. Fetches News & Calculates Sentiment.
    4. Feeds Bars to Predictor (Ensemble ARF) -> Gets Optimized R:R.
    5. Dispatches Orders to Windows with Precise Limits.
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
        
        # 3. Data Aggregation (Volume Bars)
        self.aggregators = {}
        for sym in CONFIG['trading']['symbols']:
            self.aggregators[sym] = VolumeBarAggregator(
                symbol=sym,
                threshold=CONFIG['data']['volume_bar_threshold']
            )

        # 4. AI Predictor (River Ensemble)
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

    def process_tick(self, tick_data: dict):
        """
        Handles a single raw tick from Redis.
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

            # 1. Feed Aggregator
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
        Triggered when a Volume Bar is closed.
        Executes the Prediction and Dispatch logic with Dynamic R:R.
        """
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
        
        # 3. Get Signal
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
        # Includes Section 8.2: Friday Entry Guard (checked inside SessionGuard logic)
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

        # Initial Context (R:R is informational here)
        ctx = TradeContext(
            symbol=bar.symbol,
            price=bar.close,
            stop_loss_price=0.0,
            account_equity=self.ftmo_guard.equity,
            account_currency="USD",
            win_rate=0.45,
            risk_reward_ratio=2.0 
        )

        # Get Base Trade Intent (Uses Config R:R by default)
        trade_intent, risk_usd = RiskManager.calculate_rck_size(
            context=ctx,
            conf=signal.confidence,
            volatility=volatility,
            active_correlations=active_corrs,
            market_prices=self.latest_prices,
            atr=current_atr,
            account_size=account_size,
            risk_percent_override=risk_percent_override
        )

        if trade_intent.volume <= 0:
            logger.warning(f"Trade Size 0 for {bar.symbol} (Risk Constraints).")
            return

        trade_intent.action = signal.action

        # --- CRITICAL UPDATE: DYNAMIC R:R OVERRIDE ---
        # Look for 'optimized_rr' in the signal (passed from Predictor)
        # and override the default Take Profit.
        optimized_rr = signal.meta_data.get('optimized_rr')
        
        if optimized_rr and optimized_rr > 0 and current_atr > 0:
            # Overwrite the default config-based TP
            new_tp_dist = current_atr * optimized_rr
            trade_intent.take_profit = new_tp_dist
            
            # Update comment for audit
            trade_intent.comment += f"|OptRR:{optimized_rr:.2f}"
        # ---------------------------------------------

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
        """
        if not self.ftmo_guard.can_trade():
            logger.warning(f"{LogSymbols.LOCK} FTMO Guard: Trading Halted (Drawdown).")
            return False

        # General Market Hours
        if not self.session_guard.is_trading_allowed():
            return False
            
        # Section 8.2: Friday Entry Guard (No new trades after 16:00)
        if self.session_guard.is_friday_afternoon():
            # Only block NEW entries, not exits (dispatches handle action type)
            # Since this flow is for Signal -> Entry, we block.
            return False

        if self.portfolio_mgr.check_penalty_box(symbol):
            logger.warning(f"{LogSymbols.LOCK} {symbol} is in Penalty Box.")
            return False

        if not self.news_monitor.check_trade_permission(symbol):
             return False

        return True

    def run(self):
        """
        Main Event Loop.
        """
        logger.info(f"{LogSymbols.SUCCESS} Engine Loop Started. Waiting for data on '{CONFIG['redis']['price_data_stream']}'...")
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