# FILENAME: engines/live/engine.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/live/engine.py
# DEPENDENCIES: shared, engines.live.dispatcher, engines.live.predictor
# DESCRIPTION: Core Event Loop. Ingests ticks, aggregates bars, generates signals, checks risk, executes trades.
# COMPLIANCE: FTMO Rules, Session Guards, News Filters.
# REMEDIATION:
# - Fixed Portfolio Logic Gap (Updates Returns & Correlations).
# - Accurate Dollar Risk Logging.
# - CV-1: Injects Live Market Prices into RiskManager for precise conversion.
# =============================================================================
import logging
import time
import json
import threading
import signal
import sys
from collections import defaultdict
from typing import Any, Optional

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
    1. Consumes Ticks from Redis (pushed by Windows Producer).
    2. Aggregates Ticks into Volume Bars.
    3. Feeds Bars to Predictor (River ARF).
    4. Checks Risk & Compliance.
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
        self.stream_mgr.ensure_consumer_group()

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

        # 4. AI Predictor (River)
        self.predictor = MultiAssetPredictor(symbols=CONFIG['trading']['symbols'])

        # 5. Execution Dispatcher
        self.pending_orders = {}  # Track orders to prevent dupes
        self.lock = threading.RLock()
        self.dispatcher = TradeDispatcher(
            stream_mgr=self.stream_mgr,
            pending_tracker=self.pending_orders,
            lock=self.lock
        )

        # State
        self.is_warm = False
        # Throttle correlation updates (heavy calc)
        self.last_corr_update = time.time()
        
        # AUDIT FIX: Live Price Cache for Risk Calculations
        self.latest_prices = {}

    def process_tick(self, tick_data: dict):
        """
        Handles a single raw tick from Redis.
        """
        try:
            symbol = tick_data.get('symbol')
            if symbol not in self.aggregators: return
            price = float(tick_data.get('price', 0.0))
            # Fallback to 1.0 volume if missing (common in some feeds)
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
        logger.info(f"{LogSymbols.VPIN} New Bar: {bar.symbol} | Vol: {bar.volume:.0f} | Close: {bar.close}")

        # 1. Update Portfolio Risk State (Returns)
        # AUDIT FIX: Update returns and matrix for HRP/Correlation logic
        try:
            # Simple return calculation: (Close - Open) / Open
            # NOTE: For volume bars, Open/Close are reliable snapshots
            ret = (bar.close - bar.open) / bar.open if bar.open > 0 else 0.0
            self.portfolio_mgr.update_returns(bar.symbol, ret)
           
            # Periodically rebuild correlation matrix (e.g., every 60 seconds)
            if time.time() - self.last_corr_update > 60:
                self.portfolio_mgr.update_correlation_matrix()
                self.last_corr_update = time.time()
        except Exception as e:
            logger.error(f"Portfolio Update Error: {e}")
       
        # 2. Get Signal
        signal = self.predictor.process_bar(bar.symbol, bar)
        if not signal: return

        # 3. Validate Signal
        if signal.action == "HOLD":
            return

        logger.info(f"{LogSymbols.SIGNAL} SIGNAL: {signal.action} {bar.symbol} (Conf: {signal.confidence:.2f})")

        # 4. Check Risk & Compliance Gates
        if not self._check_risk_gates(bar.symbol):
            return

        # 5. Calculate Size (RCK + CPPI)
        # Need current volatility from features
        volatility = signal.meta_data.get('volatility', 0.001)

        # Count correlations
        active_corrs = self.portfolio_mgr.get_correlation_count(
            bar.symbol,
            threshold=CONFIG['risk_management']['correlation_penalty_threshold']
        )

        # Context object for sizing
        ctx = TradeContext(
            symbol=bar.symbol,
            price=bar.close,
            stop_loss_price=0.0,  # Calculated inside RCK
            account_equity=self.ftmo_guard.equity,
            account_currency="USD",
            win_rate=0.55,
            risk_reward_ratio=1.5
        )

        # AUDIT FIX: Pass latest_prices to RiskManager for precise conversion
        trade_intent, risk_usd = RiskManager.calculate_rck_size(
            context=ctx,
            conf=signal.confidence,
            volatility=volatility,
            active_correlations=active_corrs,
            market_prices=self.latest_prices
        )

        if trade_intent.volume <= 0:
            logger.warning(f"Trade Size 0 for {bar.symbol} (Risk Constraints).")
            return

        trade_intent.action = signal.action

        # 6. Dispatch
        self.dispatcher.send_order(trade_intent, risk_usd)

    def _check_risk_gates(self, symbol: str) -> bool:
        """
        Runs the gauntlet of safety checks.
        """
        # 1. FTMO Hard Limits
        if not self.ftmo_guard.can_trade():
            logger.warning(f"{LogSymbols.LOCK} FTMO Guard: Trading Halted (Drawdown).")
            return False

        # 2. Session Time
        if not self.session_guard.is_trading_allowed():
            # logger.debug("Session Guard: Market Closed/Rollover.")
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
       
        # Redis XREAD configuration
        stream_key = CONFIG['redis']['price_data_stream']
        group = self.stream_mgr.group_name
        consumer = f"engine-{CONFIG['trading']['magic_number']}"

        # Create group if not exists
        try:
            self.stream_mgr.r.xgroup_create(stream_key, group, id='0', mkstream=True)
        except Exception:
            pass  # Exists

        while not self.shutdown_flag:
            try:
                # Read from Consumer Group
                response = self.stream_mgr.r.xreadgroup(
                    groupname=group,
                    consumername=consumer,
                    streams={stream_key: '>'},  # '>' means new messages never delivered to this consumer
                    count=50,
                    block=100
                )

                if response:
                    for stream, messages in response:
                        for message_id, data in messages:
                            self.process_tick(data)
                            # Ack immediately
                            self.stream_mgr.r.xack(stream_key, group, message_id)
               
                # Update local equity cache periodically for accurate RCK sizing
                # (This assumes Producer is updating the Redis key)
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
    # If run directly, allow a simple test boot
    engine = LiveTradingEngine()
    try:
        engine.run()
    except KeyboardInterrupt:
        engine.shutdown()