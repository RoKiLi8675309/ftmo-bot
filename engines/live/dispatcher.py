# =============================================================================
# FILENAME: engines/live/dispatcher.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/live/dispatcher.py
# DEPENDENCIES: shared
# DESCRIPTION: Outbound Trade Router. Formats and pushes execution requests to Redis.
# CRITICAL: Ensures strict compatibility with Windows Producer (Py3.9).
# 
# PHOENIX V12.39 FIX (MARKET ORDER SAFETY):
# 1. FIX: Force 'price' to "0.0" if type is "MARKET".
# 2. REASON: Prevents Producer from misinterpreting Market orders as Pending/Limit
#    orders when a snapshot price is inadvertently passed.
# =============================================================================
import logging
import json
import uuid
import threading
import time
from typing import Any, Dict

# Shared Imports
from shared import CONFIG, LogSymbols, RedisStreamManager, Trade

logger = logging.getLogger("TradeDispatcher")

class TradeDispatcher:
    """
    Handles the reliable transmission of trade orders from the Logic Engine
    to the Execution Engine (Windows Producer).
    """
    def __init__(self, stream_mgr: RedisStreamManager, pending_tracker: dict[str, Any], lock: threading.RLock):
        """
        :param stream_mgr: Active RedisStreamManager instance.
        :param pending_tracker: Reference to the Engine's pending order dictionary.
        :param lock: Thread lock to ensure thread-safe updates to the tracker.
        """
        self.stream_mgr = stream_mgr
        self.stream_key = CONFIG['redis']['trade_request_stream']
        self.magic_number = CONFIG['trading'].get('magic_number', 621001)
        self.pending_tracker = pending_tracker
        self.lock = lock

    def send_order(self, trade: Trade, estimated_risk_usd: float = 0.0) -> None:
        """
        Formats and dispatches a Trade object to Redis.
        """
        try:
            # 1. Generate IDs
            order_id = uuid.uuid4()
            short_id = str(order_id)[:8]
            
            # 2. Register Pending (Thread-Safe)
            with self.lock:
                self.pending_tracker[str(order_id)] = {
                    "symbol": trade.symbol,
                    "action": trade.action,
                    "volume": trade.volume,
                    "timestamp": time.time(),
                    "status": "PENDING"
                }

            # 3. Construct Payload (Strict String Typing for Redis)
            # Ensure Comment length compliance (MT5 limit)
            # Shorten UUID for comment: "Auto_1234abcd"
            comment = f"Auto_{short_id}"
            if trade.comment:
                comment = f"{comment}_{trade.comment}"[:31]

            # V12.39 FIX: MARKET ORDER PRICE SAFETY
            # If it's a MARKET order, we MUST send price="0.0" to force Instant Execution.
            # Sending a specific price (e.g. 1.0500) might trigger Pending Limit logic in Producer.
            entry_type = str(trade.entry_type).upper()
            final_price = str(trade.entry_price)
            
            if "MARKET" in entry_type:
                final_price = "0.0"

            payload = {
                "id": str(order_id),
                "uuid": str(order_id),  # CRITICAL for Producer Deduplication
                "symbol": str(trade.symbol),
                "action": str(trade.action),  # "BUY" or "SELL"
                "volume": "{:.2f}".format(float(trade.volume)),  # Explicit float formatting
                
                # V12.19 FIX: Send both legacy 'entry_price' and new 'price' keys
                # The Producer looks for 'price' to enable Limit Order logic.
                "entry_price": final_price,
                "price": final_price, 
                
                "stop_loss": str(trade.stop_loss),
                "take_profit": str(trade.take_profit),
                "magic_number": str(self.magic_number),
                "comment": comment,
                "timestamp": str(time.time()),  # ZOMBIE CHECK: High precision time
                
                # V12.19 FIX: Respect trade.entry_type (LIMIT vs MARKET)
                "type": entry_type
            }

            # 4. Transmit to Redis
            # AUDIT FIX: Enforce maxlen to prevent stream from growing indefinitely
            self.stream_mgr.r.xadd(self.stream_key, payload, maxlen=10000, approximate=True)
            
            logger.info(
                f"{LogSymbols.UPLOAD} DISPATCH SENT: {trade.action} {trade.symbol} "
                f"| Vol: {trade.volume:.2f} | Price: {final_price} | Type: {entry_type} | Risk: ${estimated_risk_usd:.2f}"
            )
        except Exception as e:
            logger.error(f"{LogSymbols.ERROR} Dispatch Failed for {trade.symbol}: {e}")
            # Rollback tracker if send failed
            with self.lock:
                if str(order_id) in self.pending_tracker:
                    del self.pending_tracker[str(order_id)]

    def cleanup_stale_orders(self, ttl_seconds: int = 300):
        """
        Removes orders that have been stuck in 'PENDING' for too long.
        Prevents memory leaks in the tracker.
        """
        now = time.time()
        to_remove = []
        with self.lock:
            for oid, data in self.pending_tracker.items():
                if now - data['timestamp'] > ttl_seconds:
                    to_remove.append(oid)
            
            for oid in to_remove:
                del self.pending_tracker[oid]
        
        if to_remove:
            logger.warning(f"Cleaned up {len(to_remove)} stale pending orders.")