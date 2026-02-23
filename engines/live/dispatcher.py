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
# These MUST match the Windows Producer's expectation.
MT5_ORDER_TYPE_BUY = 0
MT5_ORDER_TYPE_SELL = 1
MT5_TRADE_ACTION_DEAL = 1
MT5_TRADE_ACTION_PENDING = 5
MT5_TRADE_ACTION_SLTP = 6 
MT5_TRADE_ACTION_MODIFY = 6
MT5_TRADE_ACTION_REMOVE = 8

class TradeDispatcher:
    """
    Handles the reliable transmission of trade orders from the Logic Engine
    to the Execution Engine (Windows Producer) via Redis Streams.
    
    UPDATED: FORCE MARKET EXECUTION ONLY.
    V17.1 FIX: INJECTS STATIC INITIAL RISK TO REDIS TO PREVENT R-MULTIPLE HALLUCINATION.
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
        Strictly enforces Market Execution (Deal) protocol.
        """
        try:
            # 1. Generate IDs
            order_id = uuid.uuid4()
            short_id = str(order_id)[:8]
            
            # --- V17.1 FIX: PERSIST INITIAL RISK DISTANCE ---
            # We calculate the absolute price distance between the entry reference 
            # and the initial stop loss. This is saved to a Redis hash so the 
            # Trailing Stop and Pyramiding logic don't hallucinate shrinking 
            # R-multiples as the SL physically moves closer to price.
            try:
                initial_risk_dist = abs(trade.entry_price - trade.stop_loss)
                if initial_risk_dist > 0:
                    self.stream_mgr.r.hset("bot:initial_risk", short_id, str(initial_risk_dist))
            except Exception as e:
                logger.error(f"Failed to persist initial risk to Redis for {short_id}: {e}")
            # ------------------------------------------------
            
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
            
            # B. FORCE MARKET EXECUTION
            # We ignore trade.entry_type and force TRADE_ACTION_DEAL (1)
            # Price must be 0.0 for Instant/Market Execution in MT5
            if trade.action in ["BUY", "SELL"]:
                mt5_action = MT5_TRADE_ACTION_DEAL
                final_price = "0.0" 
            else:
                # MODIFY / CLOSE_ALL are handled specifically by Producer strings
                # But strictly speaking, they don't map to standard DEAL/PENDING here
                # We pass the string action through, Producer handles the switch
                mt5_action = MT5_TRADE_ACTION_DEAL 
                final_price = "0.0"

            # Ensure Comment length compliance (MT5 limit)
            # Shorten UUID for comment: "Auto_1234abcd"
            comment = f"Auto_{short_id}"
            if trade.comment:
                # Truncate carefully to preserve key info within 31 chars
                comment = f"{comment}_{trade.comment}"[:31]

            # Determine Action String for Producer Logic
            # If standard trade, send integer. If special command, send string.
            if trade.action in ["MODIFY", "CLOSE_ALL"]:
                action_payload = str(trade.action)
            else:
                action_payload = str(mt5_action)

            payload = {
                "id": str(order_id),
                "uuid": str(order_id),  # CRITICAL for Producer Deduplication
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
                # Enforce string format to prevent scientific notation (e.g. 1e-5)
                "sl": "{:.5f}".format(trade.stop_loss),
                "tp": "{:.5f}".format(trade.take_profit),
                
                "magic_number": str(self.magic_number),
                "magic": str(self.magic_number), 
                
                "comment": comment,
                "timestamp": str(time.time()),  # ZOMBIE CHECK: High precision time
                
                # Ticket required for MODIFY
                "ticket": str(trade.ticket) if trade.ticket else "0"
            }

            # 4. Transmit to Redis
            # Enforce maxlen to prevent stream from growing indefinitely
            self.stream_mgr.r.xadd(self.stream_key, payload, maxlen=50000, approximate=True)
            
            logger.info(
                f"{LogSymbols.UPLOAD} DISPATCH SENT: {trade.action} {trade.symbol} "
                f"| Vol: {trade.volume:.2f} | Market Order (Price=0.0) | Risk: ${estimated_risk_usd:.2f}"
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
        Cross-reference with open_positions (Zombie Check) to avoid
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
                    # Clean up the corresponding initial risk from Redis to prevent memory leaks over months
                    try:
                        short_id = str(oid)[:8]
                        self.stream_mgr.r.hdel("bot:initial_risk", short_id)
                    except Exception:
                        pass
                        
                    del self.pending_tracker[oid]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} pending orders.")