# =============================================================================
# FILENAME: engines/live/dispatcher.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/live/dispatcher.py
# DEPENDENCIES: shared
# DESCRIPTION: Outbound Trade Router. Formats and pushes execution requests to Redis.
# CRITICAL: Ensures strict compatibility with Windows Producer (Py3.9).
# 
# PHOENIX V16.20 AUDIT FIX (THE JAMMER CURE):
# 1. RACE CONDITION FIX: 'cleanup_stale_orders' now cross-references Open Positions
#    before deleting keys. If a position exists, it assumes the order filled but
#    lost its UUID link (Zombie Match), preventing duplicate firing.
# 2. TTL EXTENSION: Increased pending order TTL to 600s to survive network lag.
# =============================================================================
import logging
import json
import uuid
import threading
import time
from typing import Any, Dict, Optional

# Shared Imports
from shared import CONFIG, LogSymbols, RedisStreamManager, Trade

logger = logging.getLogger("TradeDispatcher")

# --- MT5 CONSTANTS (HARDCODED FOR LINUX) ---
# Since we cannot import MetaTrader5 on Linux, we define the standard constants here.
MT5_ORDER_TYPE_BUY = 0
MT5_ORDER_TYPE_SELL = 1
MT5_TRADE_ACTION_DEAL = 1
MT5_TRADE_ACTION_PENDING = 5

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
                    "status": "PENDING",
                    "risk_usd": estimated_risk_usd
                }

            # 3. Construct Payload (Strict Protocol Translation)
            
            # A. Translate Action (BUY/SELL -> 0/1) to 'type' field
            # Windows Producer expects 'type' to be the direction.
            if trade.action == "BUY":
                mt5_type = MT5_ORDER_TYPE_BUY
            elif trade.action == "SELL":
                mt5_type = MT5_ORDER_TYPE_SELL
            else:
                # Fallback for "CLOSE_ALL" or "MODIFY" which handle their own logic in Producer
                # But for standard entry, this is required.
                mt5_type = MT5_ORDER_TYPE_BUY # Default dummy

            # B. Translate Entry Type (MARKET -> DEAL, LIMIT -> PENDING) to 'action' field
            # Windows Producer expects 'action' to be the execution method.
            entry_type_str = str(trade.entry_type).upper()
            final_price = str(trade.entry_price)
            
            if "MARKET" in entry_type_str:
                mt5_action = MT5_TRADE_ACTION_DEAL
                final_price = "0.0" # Force zero for Market Execution
            else:
                mt5_action = MT5_TRADE_ACTION_PENDING

            # Handling Special Actions (MODIFY / CLOSE_ALL)
            # These override the standard mapping logic
            if trade.action == "MODIFY":
                pass 
            
            # Ensure Comment length compliance (MT5 limit)
            # Shorten UUID for comment: "Auto_1234abcd"
            comment = f"Auto_{short_id}"
            if trade.comment:
                # V16.20: Truncate carefully to preserve key info
                comment = f"{comment}_{trade.comment}"[:31]

            payload = {
                "id": str(order_id),
                "uuid": str(order_id),  # CRITICAL for Producer Deduplication
                "symbol": str(trade.symbol),
                
                # --- PROTOCOL TRANSLATION START ---
                # Producer: request['action'] (Deal/Pending)
                # Producer: request['type'] (Buy/Sell)
                # We send them as strings, Producer casts to int.
                
                # Special Case: MODIFY/CLOSE_ALL are high-level commands trapped by Producer before casting
                "action": str(trade.action) if trade.action in ["MODIFY", "CLOSE_ALL"] else str(mt5_action),
                "type": str(mt5_type),
                # ----------------------------------

                "volume": "{:.2f}".format(float(trade.volume)),
                
                # V12.19 FIX: Send both legacy 'entry_price' and new 'price' keys
                "entry_price": final_price,
                "price": final_price, 
                
                # CRITICAL MAPPING: Producer looks for 'sl' and 'tp', NOT 'stop_loss'
                # V16.20: Enforce string format to prevent scientific notation on small pips
                "sl": "{:.5f}".format(trade.stop_loss),
                "tp": "{:.5f}".format(trade.take_profit),
                
                "magic_number": str(self.magic_number),
                "magic": str(self.magic_number), # Map to 'magic' for Producer
                
                "comment": comment,
                "timestamp": str(time.time()),  # ZOMBIE CHECK: High precision time
                
                # Ticket required for MODIFY
                "ticket": str(trade.ticket) if trade.ticket else "0"
            }

            # 4. Transmit to Redis
            # AUDIT FIX: Enforce maxlen to prevent stream from growing indefinitely
            # 50,000 to match Producer capacity
            self.stream_mgr.r.xadd(self.stream_key, payload, maxlen=50000, approximate=True)
            
            logger.info(
                f"{LogSymbols.UPLOAD} DISPATCH SENT: {trade.action} {trade.symbol} "
                f"| Vol: {trade.volume:.2f} | Price: {final_price} | Type: {mt5_type} (MT5) | Risk: ${estimated_risk_usd:.2f}"
            )
        except Exception as e:
            logger.error(f"{LogSymbols.ERROR} Dispatch Failed for {trade.symbol}: {e}")
            # Rollback tracker if send failed
            with self.lock:
                if str(order_id) in self.pending_tracker:
                    del self.pending_tracker[str(order_id)]

    def cleanup_stale_orders(self, ttl_seconds: int = 600, open_positions: Optional[Dict] = None):
        """
        Removes orders that have been stuck in 'PENDING' for too long.
        V16.20 FIX: Cross-reference with open_positions (Zombie Check) to avoid
        deleting a key for a trade that actually filled but lost connection.
        
        :param ttl_seconds: Max age of a pending order (Default 600s / 10m).
        :param open_positions: Current live positions to check against.
        """
        now = time.time()
        to_remove = []
        
        with self.lock:
            for oid, data in self.pending_tracker.items():
                # 1. Time Check
                if now - data['timestamp'] > ttl_seconds:
                    
                    # 2. Zombie Check (If positions provided)
                    # If we find a position for this symbol that looks like it belongs to us,
                    # we assume it filled and just clear the pending flag safely.
                    is_zombie_match = False
                    if open_positions:
                        short_id = str(oid)[:8]
                        symbol = data['symbol']
                        
                        for pos in open_positions.values():
                            # Check Symbol AND Comment match
                            if pos.get('symbol') == symbol and short_id in pos.get('comment', ''):
                                is_zombie_match = True
                                break
                    
                    if is_zombie_match:
                        logger.warning(f"🧟 ZOMBIE MATCH: Clearing stale pending {oid} (Found matching position).")
                        to_remove.append(oid)
                    else:
                        # Genuine Timeout - Order likely rejected or lost
                        logger.warning(f"🧹 Clearing Stale Pending Order: {oid} ({data['symbol']}) - > {ttl_seconds}s")
                        to_remove.append(oid)
            
            for oid in to_remove:
                if oid in self.pending_tracker:
                    del self.pending_tracker[oid]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} pending orders.")