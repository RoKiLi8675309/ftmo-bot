# =============================================================================
# FILENAME: dashboard.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: dashboard.py
# DEPENDENCIES: streamlit, pandas, redis, shared
# DESCRIPTION: Real-time GUI for monitoring the Algorithmic Trading System.
#
# PHOENIX V12.8 UPDATE (DASHBOARD FIXES):
# 1. VISUALIZATION: Replaced raw stream tail with "Market Watch" (Unique Symbol Snapshot).
# 2. DEDUPLICATION: Added logic to filter duplicate consecutive ticks in the Tape view.
# 3. ROBUSTNESS: Enhanced Redis error handling and "Hard Kill" visibility.
# =============================================================================
import streamlit as st
import pandas as pd
import json
import time
import sys
import os
import redis
from datetime import datetime
import plotly.graph_objects as go

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from shared import CONFIG, get_redis_connection, LogSymbols
except ImportError as e:
    st.error(f"Failed to import shared modules: {e}")
    st.stop()

# --- CONFIGURATION ---
st.set_page_config(
    page_title="FTMO Forensic Command",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHED RESOURCES ---
@st.cache_resource
def get_db_connection():
    """Establishes persistent Redis connection."""
    return get_redis_connection(
        host=CONFIG['redis']['host'],
        port=CONFIG['redis']['port'],
        decode_responses=True
    )

try:
    r = get_db_connection()
except Exception as e:
    st.error(f"Redis Connection Failed: {e}")
    st.stop()

# --- HELPER FUNCTIONS ---
def get_system_health():
    """Checks Redis ping and Producer Heartbeat."""
    try:
        redis_alive = r.ping()
        # Producer updates 'producer:heartbeat' timestamp
        last_beat = r.get(CONFIG.get('producer', {}).get('heartbeat_key', 'producer:heartbeat'))
        
        producer_status = "OFFLINE"
        if last_beat:
            delta = time.time() - float(last_beat)
            if delta < 15:
                producer_status = "ONLINE"
            else:
                producer_status = f"STALE ({int(delta)}s)"
        
        return redis_alive, producer_status
    except Exception:
        return False, "ERROR"

def get_risk_metrics():
    """Fetches equity and drawdown info from Redis keys."""
    try:
        start_eq = float(r.get(CONFIG['redis']['risk_keys']['daily_starting_equity']) or 0.0)
        curr_eq = float(r.get(CONFIG['redis']['risk_keys']['current_equity']) or 0.0)
        hwm = float(r.get("risk:high_water_mark") or 0.0)
        
        if start_eq > 0:
            daily_dd_pct = (start_eq - curr_eq) / start_eq
            daily_dd_val = start_eq - curr_eq
        else:
            daily_dd_pct = 0.0
            daily_dd_val = 0.0
        return start_eq, curr_eq, hwm, daily_dd_pct, daily_dd_val
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0

def toggle_kill_switch():
    """Toggles the global trading suspension flag (Logic)."""
    key = CONFIG['redis']['risk_keys']['kill_switch_active']
    if r.exists(key):
        r.delete(key)
    else:
        r.set(key, "1")

# --- UI LAYOUT ---
# 1. SIDEBAR
with st.sidebar:
    st.title("🦅 Control Panel")
    
    redis_ok, producer_stat = get_system_health()
    
    st.markdown("### System Status")
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Redis", "OK" if redis_ok else "ERR")
    col_s2.metric("Producer", producer_stat)
    
    st.divider()
    
    st.markdown("### Emergency Controls")
      
    # LOGIC SWITCH (Redis)
    kill_switch_key = CONFIG['redis']['risk_keys']['kill_switch_active']
    is_killed = r.exists(kill_switch_key)
    
    if is_killed:
        st.error("🛑 KILL SWITCH ACTIVE (Logic)")
        if st.button("RESUME TRADING"):
            toggle_kill_switch()
            st.rerun()
    else:
        st.success("🟢 SYSTEM ARMED")
        if st.button("ACTIVATE KILL SWITCH (Logic)", type="primary"):
            toggle_kill_switch()
            st.rerun()
      
    st.markdown("---")
      
    # HARD KILL SWITCH (File)
    KILL_SWITCH_FILE = "kill_switch.lock"
    file_killed = os.path.exists(KILL_SWITCH_FILE)
      
    if file_killed:
        st.error("💀 HARD KILL ACTIVE (File)")
        st.caption("Windows Producer should be dead.")
        if st.button("RESET HARD KILL (Delete File)"):
            try:
                os.remove(KILL_SWITCH_FILE)
                st.success("File deleted.")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to delete file: {e}")
    else:
        if st.button("ACTIVATE HARD KILL (File)", type="secondary"):
            try:
                with open(KILL_SWITCH_FILE, "w") as f:
                    f.write("KILL")
                st.error("Hard Kill File Created!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create file: {e}")
            
    st.divider()
    refresh_rate = st.slider("Refresh Rate (s)", 1, 60, 2)

# 2. MAIN HEADER
st.title("🦅 FTMO Forensic Command")
st.markdown(f"**Environment:** {sys.platform} | **Python:** {sys.version.split()[0]}")

# 3. RISK COCKPIT (Top Row)
start_eq, curr_eq, hwm, daily_dd_pct, daily_dd_val = get_risk_metrics()
limit_pct = CONFIG['risk_management']['max_daily_loss_pct']

# Progress Bar for Daily Limit
st.markdown("#### Daily Risk Utilization")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Equity", f"${curr_eq:,.2f}", delta=f"{curr_eq - start_eq:,.2f}")
with col2:
    st.metric("Daily Drawdown", f"{daily_dd_pct:.2%}", delta=f"-{daily_dd_val:,.2f}", delta_color="inverse")
with col3:
    st.metric("Daily Limit", f"{limit_pct:.1%}", f"Buffer: {(limit_pct - daily_dd_pct):.2%}")
with col4:
    st.metric("High Water Mark", f"${hwm:,.2f}")

progress = min(1.0, max(0.0, daily_dd_pct / limit_pct)) if limit_pct > 0 else 0
st.progress(progress)
if progress > 0.8:
    st.warning(f"⚠️ Warning: {progress:.1%} of Daily Risk Budget consumed!")

# 4. LIVE MARKET DATA (Tabs)
tab_market, tab_trades, tab_logs = st.tabs(["📊 Live Market", "⚖️ Trade Log", "📜 System Logs"])

with tab_market:
    # --- MARKET SNAPSHOT (DEDUPLICATED) ---
    st.subheader("Market Watch (Latest State)")
    
    stream_key = CONFIG['redis']['price_data_stream']
    try:
        # Read larger chunk to ensure we get all symbols, but deduplicate by ID
        # xrevrange gives Newest -> Oldest
        raw_data = r.xrevrange(stream_key, count=1000)
        
        snapshot_data = {}
        tape_data = []
        seen_symbols = set()
        
        for msg_id, payload in raw_data:
            sym = payload.get('symbol')
            ts_raw = payload.get('time', 0)
            
            # Format Timestamp
            try:
                ts_val = float(ts_raw)
                if ts_val > 1e10: ts_val /= 1000 # Handle ms
                dt_str = datetime.fromtimestamp(ts_val).strftime('%H:%M:%S')
            except:
                dt_str = str(ts_raw)

            # SNAPSHOT LOGIC: Only take the first (latest) occurrence of each symbol
            if sym and sym not in seen_symbols:
                snapshot_data[sym] = {
                    "Symbol": sym,
                    "Time": dt_str,
                    "Bid": float(payload.get('bid', 0)),
                    "Ask": float(payload.get('ask', 0)),
                    "Spread": (float(payload.get('ask', 0)) - float(payload.get('bid', 0))) * 10000 if 'JPY' in sym else (float(payload.get('ask', 0)) - float(payload.get('bid', 0))) * 10000, # Rough pip calc
                    "Vol": float(payload.get('volume', 0)),
                    "Bid Vol": float(payload.get('bid_vol', 0)),
                    "Ask Vol": float(payload.get('ask_vol', 0)),
                }
                seen_symbols.add(sym)
            
            # TAPE LOGIC: Add to tape list (we can limit this later)
            tape_row = {
                "ID": msg_id,
                "Time": dt_str,
                "Symbol": sym,
                "Bid": payload.get('bid'),
                "Ask": payload.get('ask'),
                "Vol": payload.get('volume')
            }
            tape_data.append(tape_row)
        
        # Display Snapshot Table
        if snapshot_data:
            df_snap = pd.DataFrame(list(snapshot_data.values()))
            # Sort by Symbol for stability
            df_snap.sort_values('Symbol', inplace=True)
            st.dataframe(df_snap, use_container_width=True, hide_index=True)
        else:
            st.info("Waiting for tick data...")

        # --- RAW TAPE (EXPANDER) ---
        with st.expander("Raw Tick Tape (Last 50)"):
            if tape_data:
                df_tape = pd.DataFrame(tape_data[:50]) # Show top 50
                st.dataframe(df_tape, use_container_width=True)
            else:
                st.write("No tape data.")
                
    except Exception as e:
        st.error(f"Error reading stream: {e}")

with tab_trades:
    st.subheader("Execution Stream")
    trade_stream = CONFIG['redis']['trade_request_stream']
    try:
        raw_trades = r.xrevrange(trade_stream, count=50)
        trade_list = []
        for msg_id, payload in raw_trades:
            # Clean up payload for display
            display_payload = payload.copy()
            # Timestamp formatting
            if 'timestamp' in display_payload:
                try:
                    ts = float(display_payload['timestamp'])
                    display_payload['time'] = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                except: pass
            
            trade_list.append(display_payload)
        
        if trade_list:
            df_trades = pd.DataFrame(trade_list)
            st.dataframe(df_trades, use_container_width=True)
        else:
            st.info("No recent trades dispatched.")
    except Exception as e:
        st.error(f"Error reading trade stream: {e}")

with tab_logs:
    st.subheader("System Event Log (Simulated Tail)")
    log_file = "logs/ftmo_bot.log"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            # Read last 50 lines efficiently
            try:
                # Seek to end and read backwards approx 50 lines
                f.seek(0, 2)
                file_size = f.tell()
                # Read last 8KB of logs
                read_size = min(8192, file_size)
                f.seek(file_size - read_size)
                lines = f.readlines()
                # If we read halfway through a line, drop the first one
                if len(lines) > 1 and read_size < file_size:
                    lines = lines[1:]
                
                # Show last 20 reversed
                for line in reversed(lines[-20:]):
                    st.text(line.strip())
            except Exception as e:
                st.error(f"Log read error: {e}")
    else:
        st.warning(f"Log file not found at {log_file}")

# Auto-Refresh Logic
time.sleep(refresh_rate)
st.rerun()