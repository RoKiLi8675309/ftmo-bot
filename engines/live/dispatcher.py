import logging
import time
import math
import uuid
import threading
from typing import Any, Dict, Optional

# Shared Imports
from shared import CONFIG, LogSymbols, RedisStreamManager, Trade

logger = logging.getLogger("TradeDispatcher")

# --- MT5 CONSTANTS (HARDCODED FOR LINUX / REDIS PROTOCOL) ---
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
    V20.18 DUAL-MODEL UPGRADE: Fully supports independent directional EV processing.
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
            
            # --- V20.18 FIX: STRICT R:R ENFORCEMENT AT THE DISPATCH LAYER ---
            if trade.action in ["BUY", "SELL"] and trade.stop_loss > 0 and trade.take_profit > 0:
                risk_dist = abs(trade.entry_price - trade.stop_loss)
                reward_dist = abs(trade.take_profit - trade.entry_price)
                if risk_dist > 0:
                    rr_ratio = reward_dist / risk_dist
                    # Target >= 1.5. Blocked at 1.40 to permit minor spread slippage variance.
                    if rr_ratio < 1.40:
                        logger.error(f"🛑 REJECTED BY DISPATCHER: {trade.symbol} {trade.action} R:R Ratio is {rr_ratio:.2f} (Target >= 1.5). Blocked to prevent spread bleed.")
                        return  # Abort dispatch immediately

            # --- PERSIST INITIAL RISK DISTANCE (WITH GLOBAL TTL) ---
            try:
                initial_risk_dist = abs(trade.entry_price - trade.stop_loss)
                if initial_risk_dist > 0:
                    self.stream_mgr.r.hset("bot:initial_risk", short_id, str(initial_risk_dist))
                    # V20.18.2 FIX: Decouple lifecycle by setting a global 48-hour TTL on the hash.
                    # This prevents memory leaks while ensuring active trades always have their R-multiple data.
                    self.stream_mgr.r.expire("bot:initial_risk", 172800) 
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
            
            # FORCE MARKET EXECUTION
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

            # Sanitize floats
            safe_sl = float(trade.stop_loss) if math.isfinite(trade.stop_loss) else 0.0
            safe_tp = float(trade.take_profit) if math.isfinite(trade.take_profit) else 0.0
            safe_vol = float(trade.volume) if math.isfinite(trade.volume) else 0.01

            payload = {
                "id": str(order_id),
                "uuid": str(order_id),  
                "symbol": str(trade.symbol),
                
                # --- PROTOCOL TRANSLATION ---
                "action": action_payload,
                "type": str(mt5_type),
                
                "volume": "{:.2f}".format(safe_vol),
                
                # FORCE ZERO PRICE for Market Execution
                "entry_price": final_price,
                "price": final_price, 
                
                "intended_price": "{:.5f}".format(trade.entry_price),
                
                "sl": "{:.5f}".format(safe_sl),
                "tp": "{:.5f}".format(safe_tp),
                
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
                            comment = str(pos.get('comment', ''))
                            if pos.get('symbol') == symbol and short_id in comment:
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
                    # V20.18.2 FIX: Removed hdel call. 
                    # We NO LONGER delete the initial_risk hash key here. We rely on the global 48h TTL
                    # to clean it up so that active trades correctly resolve their R-multiples for trailing stops.
                    del self.pending_tracker[oid]