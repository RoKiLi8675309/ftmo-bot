# =============================================================================
# FILENAME: dashboard.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: dashboard.py
# DEPENDENCIES: streamlit, pandas, redis, plotly, shared
# DESCRIPTION: Enhanced Real-time GUI for the Algorithmic Trading System.
#
# PHOENIX V16.5 UI OVERHAUL:
# 1. UX UPGRADE: Added Active Positions Table, Drawdown Gauge, and Better Metrics.
# 2. VISIBILITY: Explicitly tracking Free Margin and PnL coloring.
# 3. SAFETY: Clearer Kill Switch status indicators.
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
    page_title="Phoenix Command",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        div[data-testid="metric-container"] {
            background-color: #1E1E1E;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #FF4B4B;
        }
        div[data-testid="stDataFrame"] { width: 100%; }
    </style>
""", unsafe_allow_html=True)

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

# --- DATA FETCHING FUNCTIONS ---

def get_system_health():
    """Checks Redis ping and Producer Heartbeat."""
    try:
        redis_alive = r.ping()
        # Producer updates 'producer:heartbeat' timestamp
        last_beat = r.get(CONFIG.get('producer', {}).get('heartbeat_key', 'producer:heartbeat'))
        
        producer_status = "OFFLINE"
        status_color = "red"
        
        if last_beat:
            delta = time.time() - float(last_beat)
            if delta < 15:
                producer_status = "ONLINE"
                status_color = "green"
            elif delta < 60:
                producer_status = f"LAGGING ({int(delta)}s)"
                status_color = "orange"
            else:
                producer_status = f"STALE ({int(delta)}s)"
        
        return redis_alive, producer_status, status_color
    except Exception:
        return False, "ERROR", "red"

def get_risk_metrics():
    """Fetches equity and drawdown info from Redis keys."""
    try:
        start_eq = float(r.get(CONFIG['redis']['risk_keys']['daily_starting_equity']) or 0.0)
        curr_eq = float(r.get(CONFIG['redis']['risk_keys']['current_equity']) or 0.0)
        
        # Calculate daily PnL
        daily_pnl = curr_eq - start_eq
        
        # Calculate Daily Drawdown % (Only relevant if PnL is negative)
        if start_eq > 0:
            if daily_pnl < 0:
                daily_dd_pct = abs(daily_pnl) / start_eq
            else:
                daily_dd_pct = 0.0
        else:
            daily_dd_pct = 0.0
            
        return start_eq, curr_eq, daily_pnl, daily_dd_pct
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def get_account_margin_info():
    """Fetches Margin info for Leverage Guard visibility."""
    try:
        key = CONFIG['redis'].get('account_info_key', 'account:info')
        info = r.hgetall(key)
        if not info:
            return 0.0, 0.0, 0.0
        
        equity = float(info.get('equity', 0.0))
        margin = float(info.get('margin', 0.0))
        free_margin = float(info.get('free_margin', 0.0))
        
        # Calculate Margin Level
        margin_level = (equity / margin * 100) if margin > 0 else 9999.0
        
        return free_margin, margin, margin_level
    except Exception:
        return 0.0, 0.0, 0.0

def get_open_positions():
    """Fetches the actual open positions from Redis."""
    try:
        magic = CONFIG['trading']['magic_number']
        key = f"{CONFIG['redis']['position_state_key_prefix']}:{magic}"
        data = r.get(key)
        if data:
            return json.loads(data)
        return []
    except Exception:
        return []

def get_recent_signals(limit=10):
    """Fetches recent signals dispatched to the execution stream."""
    stream_key = CONFIG['redis']['trade_request_stream']
    try:
        raw_trades = r.xrevrange(stream_key, count=limit)
        trade_list = []
        for msg_id, payload in raw_trades:
            item = payload.copy()
            # Timestamp formatting
            if 'timestamp' in item:
                try:
                    ts = float(item['timestamp'])
                    item['time'] = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                except: pass
            trade_list.append(item)
        return trade_list
    except Exception:
        return []

def toggle_kill_switch():
    """Toggles the global trading suspension flag (Logic)."""
    key = CONFIG['redis']['risk_keys']['kill_switch_active']
    if r.exists(key):
        r.delete(key)
    else:
        r.set(key, "1")

# --- UI COMPONENTS ---

def render_gauge(current_val, max_val, title):
    """Renders a Plotly Gauge chart for Drawdown."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_val * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, max_val * 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#FF4B4B"}, # Red bar for drawdown
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_val * 0.5 * 100], 'color': "#1E3D59"}, # Safe zone
                {'range': [max_val * 0.5 * 100, max_val * 0.8 * 100], 'color': "#FFC107"}, # Warning
                {'range': [max_val * 0.8 * 100, max_val * 100], 'color': "#FF4B4B"}], # Danger
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 100}
        }
    ))
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

# --- MAIN LAYOUT ---

# 1. SIDEBAR
with st.sidebar:
    st.header("🦅 Control Tower")
    
    redis_ok, producer_stat, prod_color = get_system_health()
    
    st.markdown(f"**Redis:** {'✅' if redis_ok else '❌'}")
    # FIX: variable producer_status -> producer_stat
    st.markdown(f"**Producer:** :{prod_color}[{producer_stat}]")
    
    st.divider()
    
    # KILL SWITCH LOGIC
    kill_switch_key = CONFIG['redis']['risk_keys']['kill_switch_active']
    is_killed = r.exists(kill_switch_key)
    
    if is_killed:
        st.error("⛔ BOT PAUSED")
        if st.button("RESUME TRADING", use_container_width=True):
            toggle_kill_switch()
            st.rerun()
    else:
        st.success("🟢 BOT ACTIVE")
        if st.button("PAUSE TRADING", type="primary", use_container_width=True):
            toggle_kill_switch()
            st.rerun()
            
    st.markdown("---")
    refresh_rate = st.select_slider("Refresh Rate", options=[1, 2, 5, 10, 30], value=2)
    
    st.info(f"Bot Version: 16.5\nMode: {CONFIG['env']['mode']}")

# 2. KPI HEADER
st.title("Phoenix Algo Dashboard")

start_eq, curr_eq, daily_pnl, daily_dd_pct = get_risk_metrics()
free_margin, margin, margin_level = get_account_margin_info()
daily_limit_pct = CONFIG['risk_management']['max_daily_loss_pct']

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Equity", f"${curr_eq:,.2f}", f"${daily_pnl:,.2f} Today")

with kpi2:
    # Color logic for PnL
    pnl_color = "normal" if daily_pnl == 0 else ("off" if daily_pnl < 0 else "inverse")
    st.metric("Daily Return", f"{(daily_pnl/start_eq if start_eq else 0):.2%}", delta_color=pnl_color)

with kpi3:
    st.metric("Free Margin", f"${free_margin:,.0f}", f"Level: {margin_level:.0f}%")

with kpi4:
    # Drawdown metric
    dd_status = "Safe" if daily_dd_pct < daily_limit_pct * 0.5 else "Warning"
    st.metric("Daily Drawdown", f"{daily_dd_pct:.2%}", f"Limit: {daily_limit_pct:.1%} ({dd_status})", delta_color="inverse")

# 3. VISUALIZATIONS & ACTIVE TRADES
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Active Positions")
    positions = get_open_positions()
    
    if positions:
        df_pos = pd.DataFrame(positions)
        # Select and Rename columns for display
        cols_to_show = ['symbol', 'type', 'volume', 'entry_price', 'profit', 'comment']
        # Filter existing columns
        df_pos = df_pos[[c for c in cols_to_show if c in df_pos.columns]]
        
        st.dataframe(
            df_pos,
            use_container_width=True,
            column_config={
                "symbol": "Symbol",
                "type": "Side",
                "volume": "Lots",
                "entry_price": st.column_config.NumberColumn("Entry", format="$%.5f"),
                "profit": st.column_config.NumberColumn("PnL", format="$%.2f"),
                "comment": "Tag"
            },
            hide_index=True
        )
    else:
        st.info("💤 No active positions. Scanning markets...")

    st.subheader("Recent Signals")
    signals = get_recent_signals()
    if signals:
        df_sig = pd.DataFrame(signals)
        cols_sig = ['time', 'symbol', 'action', 'volume', 'price', 'confidence']
        df_sig = df_sig[[c for c in cols_sig if c in df_sig.columns]]
        st.dataframe(df_sig, use_container_width=True, hide_index=True, height=200)

with col_right:
    st.subheader("Risk Gauge")
    # Gauge Chart for Drawdown
    fig_gauge = render_gauge(daily_dd_pct, daily_limit_pct, "Daily Drawdown Utilization")
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.subheader("Market Pulse")
    # Quick Ticker View
    stream_key = CONFIG['redis']['price_data_stream']
    try:
        raw_ticks = r.xrevrange(stream_key, count=20)
        tick_data = {}
        for _, payload in raw_ticks:
            sym = payload.get('symbol')
            if sym not in tick_data:
                tick_data[sym] = {
                    'Price': float(payload.get('price', 0)),
                    'Vol': float(payload.get('volume', 0))
                }
        
        if tick_data:
            df_pulse = pd.DataFrame.from_dict(tick_data, orient='index').reset_index()
            df_pulse.columns = ['Symbol', 'Price', 'Volume']
            st.dataframe(
                df_pulse, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="%.3f")
                }
            )
    except Exception:
        st.caption("Waiting for market data...")

# 4. LOGS & SYSTEM
with st.expander("📜 System Logs & Audit Trail", expanded=False):
    log_file = "logs/ftmo_bot.log"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
            # Show last 20 lines
            for line in reversed(lines[-20:]):
                st.text(line.strip())
    else:
        st.warning(f"Log file not found at {log_file}")

# Auto-Refresh Logic
time.sleep(refresh_rate)
st.rerun()