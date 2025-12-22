# =============================================================================
# FILENAME: engines/live/engine.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/live/engine.py
# DEPENDENCIES: shared, engines.live.dispatcher, engines.live.predictor
# DESCRIPTION: Core Event Loop. Ingests ticks, aggregates bars, generates signals, checks risk, executes trades.
# COMPLIANCE: FTMO Rules, Session Guards, News Filters.
# AUDIT REMEDIATION (PHASE 2):
# - WARM-UP GATE: Enforces strict burn-in period (signals 'WARMUP' are ignored).
# - PORTFOLIO: Maintains correlation matrix updates even during warm-up.
# - RISK: Passes 'atr' from signal metadata to RiskManager for adaptive stops.
# - FORCE UPDATE: Implements 'Aggressive Mode' to force 1 trade per day per pair.
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
        
        # Live Price Cache for Risk Calculations
        self.latest_prices = {}

        # FORCE TRADE LOGIC: Track trades per day to enforce 1/day/pair
        self.daily_trades = defaultdict(int)
        self.last_reset_date = datetime.now().date()

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
        # 0. Daily Reset Logic for Forced Trades
        current_date = datetime.fromtimestamp(bar.timestamp).date()
        if current_date > self.last_reset_date:
            self.daily_trades.clear()
            self.last_reset_date = current_date
            logger.info(f"{LogSymbols.RESET} Daily trade counters reset for {current_date}")

        # 1. Update Portfolio Risk State (Returns)
        try:
            # Simple return calculation: (Close - Open) / Open
            ret = (bar.close - bar.open) / bar.open if bar.open > 0 else 0.0
            self.portfolio_mgr.update_returns(bar.symbol, ret)
            
            # Periodically rebuild correlation matrix (e.g., every 60 seconds)
            if time.time() - self.last_corr_update > 60:
                self.portfolio_mgr.update_correlation_matrix()
                self.last_corr_update = time.time()
        except Exception as e:
            logger.error(f"Portfolio Update Error: {e}")
        
        # 2. Determine Aggression (Force Trade)
        # If we haven't traded this symbol today, we activate Aggressive Mode
        aggressive_mode = (self.daily_trades[bar.symbol] == 0)
        
        if aggressive_mode:
            # We log this sparingly to avoid spam, but distinct enough to know why it traded
            # logger.debug(f"⚡ AGGRESSIVE MODE ACTIVE for {bar.symbol} (0 trades today)")
            pass

        # 3. Get Signal (Pass Aggressive Flag)
        signal = self.predictor.process_bar(bar.symbol, bar, aggressive=aggressive_mode)
        if not signal: return

        # --- PHASE 2: WARM-UP GATE ---
        # Strictly ignore signals during burn-in.
        if signal.action == "WARMUP":
            return

        # 4. Validate Signal
        if signal.action == "HOLD":
            return

        logger.info(f"{LogSymbols.SIGNAL} SIGNAL: {signal.action} {bar.symbol} (Conf: {signal.confidence:.2f}) | Forced: {aggressive_mode}")

        # 5. Check Risk & Compliance Gates (Pass Aggressive Flag)
        if not self._check_risk_gates(bar.symbol, aggressive=aggressive_mode):
            return

        # 6. Calculate Size (Kelly-Vol)
        # Need current volatility from features (passed in signal metadata)
        volatility = signal.meta_data.get('volatility', 0.001)
        # ATR is now critical for Phase 2 sizing
        current_atr = signal.meta_data.get('atr', 0.001) 

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
            risk_reward_ratio=2.0 # Updated to 2.0 per config
        )

        # Calculate Size
        trade_intent, risk_usd = RiskManager.calculate_rck_size(
            context=ctx,
            conf=signal.confidence,
            volatility=volatility, # Used for Kelly
            active_correlations=active_corrs,
            market_prices=self.latest_prices,
            atr=current_atr # Phase 2: Explicitly pass ATR for stop distances
        )

        if trade_intent.volume <= 0:
            logger.warning(f"Trade Size 0 for {bar.symbol} (Risk Constraints).")
            return

        trade_intent.action = signal.action

        # 7. Dispatch
        if self.dispatcher.send_order(trade_intent, risk_usd):
            # Only increment daily counter if dispatch was successful
            self.daily_trades[bar.symbol] += 1

    def _check_risk_gates(self, symbol: str, aggressive: bool = False) -> bool:
        """
        Runs the gauntlet of safety checks.
        If 'aggressive' is True, we bypass non-critical gates to ensure daily volume.
        """
        # 1. FTMO Hard Limits (CRITICAL - NEVER BYPASS)
        # Bypassing this would lose the funded account.
        if not self.ftmo_guard.can_trade():
            logger.warning(f"{LogSymbols.LOCK} FTMO Guard: Trading Halted (Drawdown).")
            return False

        # If Aggressive (Forced Trade), bypass optional filters
        if aggressive:
            return True

        # 2. Session Time (Optional)
        if not self.session_guard.is_trading_allowed():
            # logger.debug("Session Guard: Market Closed/Rollover.")
            return False

        # 3. Penalty Box (Optional)
        if self.portfolio_mgr.check_penalty_box(symbol):
            logger.warning(f"{LogSymbols.LOCK} {symbol} is in Penalty Box.")
            return False

        # 4. News Blackout (Optional)
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