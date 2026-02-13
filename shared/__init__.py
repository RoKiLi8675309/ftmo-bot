"""
shared/__init__.py
Exposes all shared components.
COMPATIBILITY: Python 3.9 (Windows) & 3.11 (Linux)
AUDIT FIX: Removed circular imports from 'engines'. Shared is a foundation library.
"""

from shared.core.config import CONFIG
from shared.core.logging_setup import setup_logging, LogSymbols
from shared.core.infrastructure import RedisStreamManager, get_redis_connection
from shared.domain.models import Trade, TradeContext, VolumeBar, NewsEvent

# --- CORE UTILS (Safe) ---
from shared.utils import PrecisionGuard, SystemDiagnose

# --- FINANCIAL (Safe) ---
from shared.financial.risk import (
    RiskManager, 
    PortfolioRiskManager, 
    FTMORiskMonitor,
    HierarchicalRiskParity,
    SessionGuard
)

from shared.financial.features import (
    OnlineFeatureEngineer,
    StreamingTripleBarrier, # Kept for legacy compatibility if needed
    AdaptiveTripleBarrier,  # PHASE 2: New Volatility-Adaptive Labeler
    StreamingIndicators,    # PHASE 2: New Recursive Indicators
    ProbabilityCalibrator,
    VPINMonitor,
    enrich_with_d1_data,
    calculate_hurst,
    IncrementalFracDiff,
    VolatilityMonitor,
    MetaLabeler             # PHASE 2: Exposed for Research pipeline
)

# --- TRANSFORMERS (Safe) ---
from shared.financial.transformer import (
    TimeFeatureTransformer,
    ClusterContextBuilder
)

# --- COMPLIANCE (Safe) ---
from shared.compliance import (
    NewsEventMonitor,
    FTMOComplianceGuard
)

# --- DATA (Safe - Psycopg2 guarded internally) ---
from shared.data import (
    load_real_data,
    batch_generate_volume_bars,
    VolumeBarAggregator,
    TemporalPipeline,
    AdaptiveImbalanceBarGenerator # V10.0: Added TIB Generator
)

# --- OPTIONAL DEPENDENCY CHECKS ---

# SCIPY CHECK (Required for HRP/Cluster, optional for basic Risk)
try:
    import scipy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# MT5 CHECK (Windows Only)
try:
    import MetaTrader5 as mt5 # type: ignore
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# RIVER CHECK (Linux Only - ML)
try:
    import river
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False