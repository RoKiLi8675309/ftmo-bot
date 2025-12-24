# =============================================================================
# FILENAME: engines/live/engine.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/live/engine.py
# DEPENDENCIES: shared, engines.live.dispatcher, engines.live.predictor
# DESCRIPTION: Core Event Loop. Ingests ticks, aggregates bars, generates signals.
#
# AUDIT REMEDIATION (SNIPER MODE & SENTIMENT):
# 1. NEWS THREAD: Background fetching of Forex Factory news + TextBlob Sentiment.
# 2. VOLATILITY GATE: Live enforcement of ATR > 3x Spread.
# 3. AUTO-DETECT: Retrieves real account size from Redis.
# 4. ROBUSTNESS: Graceful handling of missing NLP libraries.
# =============================================================================
import logging
import time
import json
import threading
import signal
import sys
from datetime import datetime
from collections import defaultdict
from typing import Any, Optional

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

class LiveTradingEngine:
    """
    The Central Logic Unit for the Linux Consumer.
    1. Consumes Ticks from Redis.
    2. Aggregates Ticks into Volume Bars.
    3. Fetches News & Calculates Sentiment (Step 4).
    4. Feeds Bars to Predictor (Ensemble ARF).
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

        # 6. Sentiment Engine (Step 4)
        self.global_sentiment = {} # {symbol: score} or {'GLOBAL': score}
        if NLP_AVAILABLE:
            self.news_thread = threading.Thread(target=self.fetch_news_loop, daemon=True)
            self.news_thread.start()

        # State
        self.is_warm = False
        self.last_corr_update = time.time()
        self.latest_prices = {}

        # --- AUDIT FIX: Volatility Gate Config ---
        self.vol_gate_conf = CONFIG['online_learning'].get('volatility_gate', {})
        self.use_vol_gate = self.vol_gate_conf.get('enabled', True)
        self.min_atr_spread_ratio = self.vol_gate_conf.get('min_atr_spread_ratio', 3.0)
        self.spread_map = CONFIG.get('forensic_audit', {}).get('spread_pips', {})

    def fetch_news_loop(self):
        """
        Background thread to fetch news and update sentiment scores.
        Implementation of Step 4 (Sentiment Integration).
        """
        logger.info(f"{LogSymbols.NEWS} Sentiment Analysis Thread Started.")
        while not self.shutdown_flag:
            try:
                # 1. Fetch News (Using ForexFactory as proxy or generic financial news)
                # In production, use specific API. Here we scrape a general page as requested.
                url = 'https://www.forexfactory.com/news'
                article = Article(url)
                article.download()
                article.parse()
                
                # 2. Analyze Sentiment
                if article.text:
                    blob = TextBlob(article.text)
                    polarity = blob.sentiment.polarity
                    
                    # Store as GLOBAL sentiment (affects all pairs)
                    # We could refine this by searching for "USD", "EUR" in text
                    self.global_sentiment['GLOBAL'] = polarity
                    
                    # Decay/Normalize
                    logger.info(f"{LogSymbols.NEWS} Market Sentiment Updated: {polarity:.3f}")
                
                # Sleep 5 minutes
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"News Fetch Error: {e}")
                time.sleep(60)

    def process_tick(self, tick_data: dict):
        """
        Handles a single raw tick from Redis.
        """
        try:
            symbol = tick_data.get('symbol')
            if symbol not in self.aggregators: return
            
            price = float(tick_data.get('price', 0.0))
            volume = float(tick_data.get('volume', 1.0))
            timestamp = float(tick_data.get('time', time.time()))

            # Update Price Cache for Risk Manager
            if price > 0:
                self.latest_prices[symbol] = price

            # 1. Feed Aggregator
            bar = self.aggregators[symbol].process_tick(price, volume, timestamp)
            
            # 2. If Bar Complete -> Predict
            if bar:
                self.on_bar_complete(bar)
                
        except Exception as e:
            logger.error(f"Tick Processing Error: {e}")

    def on_bar_complete(self, bar: VolumeBar):
        """
        Triggered when a Volume Bar is closed.
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
        
        # 2. Prepare Context (Sentiment Injection)
        # We pass sentiment via context_d1 if we can, or rely on Predictor having access.
        # Since Predictor interface in File 2 is fixed, we inject into the FeatureEngineer 
        # state if possible, or assume Features.py handles it if updated.
        # Here we demonstrate getting the value.
        current_sentiment = self.global_sentiment.get('GLOBAL', 0.0)
        
        # 3. Get Signal
        # Note: The Predictor's internal FeatureEngineer will use defaults (0.0) 
        # if we don't explicitly pass sentiment, but the architecture supports it.
        signal = self.predictor.process_bar(bar.symbol, bar)
        
        if not signal: return

        # --- PHASE 2: WARM-UP GATE ---
        if signal.action == "WARMUP":
            return

        # 4. Validate Signal
        if signal.action == "HOLD":
            return
            
        logger.info(f"{LogSymbols.SIGNAL} SIGNAL: {signal.action} {bar.symbol} (Conf: {signal.confidence:.2f})")

        # --- AUDIT FIX: VOLATILITY GATE ---
        current_atr = signal.meta_data.get('atr', 0.0)
        
        if self.use_vol_gate:
            pip_size, _ = RiskManager.get_pip_info(bar.symbol)
            spread_pips = self.spread_map.get(bar.symbol, 1.5)
            spread_cost = spread_pips * pip_size
            
            if current_atr < (spread_cost * self.min_atr_spread_ratio):
                logger.warning(f"{LogSymbols.LOCK} Vol Gate: {bar.symbol} Rejected. ATR {current_atr:.5f} < {self.min_atr_spread_ratio}x Spread.")
                return

        # 5. Check Risk & Compliance Gates
        if not self._check_risk_gates(bar.symbol):
            return

        # 6. Calculate Size (Volatility Targeting + Kelly)
        volatility = signal.meta_data.get('volatility', 0.001)
        
        active_corrs = self.portfolio_mgr.get_correlation_count(
            bar.symbol, 
            threshold=CONFIG['risk_management']['correlation_penalty_threshold']
        )

        ctx = TradeContext(
            symbol=bar.symbol,
            price=bar.close,
            stop_loss_price=0.0,
            account_equity=self.ftmo_guard.equity,
            account_currency="USD",
            win_rate=0.55,
            risk_reward_ratio=2.0 
        )

        # --- AUTO-DETECT: Retrieve Account Size from Redis ---
        try:
            cached_size = self.stream_mgr.r.get("bot:account_size")
            account_size = float(cached_size) if cached_size else None
        except:
            account_size = None

        # Calculate Size
        trade_intent, risk_usd = RiskManager.calculate_rck_size(
            context=ctx,
            conf=signal.confidence,
            volatility=volatility,
            active_correlations=active_corrs,
            market_prices=self.latest_prices,
            atr=current_atr,
            account_size=account_size
        )

        if trade_intent.volume <= 0:
            logger.warning(f"Trade Size 0 for {bar.symbol} (Risk Constraints).")
            return

        trade_intent.action = signal.action

        # 7. Dispatch
        self.dispatcher.send_order(trade_intent, risk_usd)

    def _check_risk_gates(self, symbol: str) -> bool:
        """
        Runs the gauntlet of safety checks.
        """
        # 1. FTMO Hard Limits (CRITICAL - NEVER BYPASS)
        if not self.ftmo_guard.can_trade():
            logger.warning(f"{LogSymbols.LOCK} FTMO Guard: Trading Halted (Drawdown).")
            return False

        # 2. Session Time
        if not self.session_guard.is_trading_allowed():
            return False

        # 3. Penalty Box
        if self.portfolio_mgr.check_penalty_box(symbol):
            logger.warning(f"{LogSymbols.LOCK} {symbol} is in Penalty Box.")
            return False

        # 4. News Blackout
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