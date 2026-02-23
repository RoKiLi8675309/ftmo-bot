# =============================================================================
# FILENAME: shared/core/config.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/core/config.py
# DEPENDENCIES: pyyaml, python-dotenv
# DESCRIPTION: Loads configuration from YAML and injects Environment secrets.
# Supports path resolution for both Monolithic and Modular layouts.
# CRITICAL: Python 3.9 Compatible (No '|' unions).
#
# UPDATES (V17.4 - Loky Conda Sync Fix):
# 1. ENVIRONMENT LOCK: Enforced LOKY_PYTHON to prevent multiprocessing workers
#    from dropping the Conda path and failing psycopg2 imports.
# 2. PYLANCE FIX: Removed manual psycopg2 imports to clear source resolution errors.
# =============================================================================

import os
import sys

# --- CRITICAL MULTIPROCESSING FIX ---
# Forces joblib/loky to use the exact Conda python executable for all worker processes.
# This prevents the workers from hallucinating that packages like psycopg2 don't exist.
os.environ["LOKY_PYTHON"] = sys.executable
os.environ["PYTHONEXECUTABLE"] = sys.executable

import yaml
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional

def load_environment_variables() -> None:
    """
    Loads .env file from the project root.
    Traverses up from shared/core/config.py to find the root.
    """
    try:
        # shared/core/config.py -> shared/core -> shared -> root
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        env_path = project_root / '.env'
        
        if env_path.exists():
            load_dotenv(dotenv_path=str(env_path))
        else:
            # Fallback for Docker or different CWD
            load_dotenv()
    except Exception as e:
        print(f"Warning: Failed to load .env file: {e}")

# Load env vars immediately on module import
load_environment_variables()

def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Injects default values for new features and enforces Safety Protocols.
    """
    # 1. Ensure online_learning exists
    if 'online_learning' not in config:
        config['online_learning'] = {}

    # 2. Triple Barrier Defaults (Rec 3: Volatility Horizon)
    if 'tbm' not in config['online_learning']:
        config['online_learning']['tbm'] = {}
    
    tbm = config['online_learning']['tbm']
    # Default to TIME for safety, but enable code to see VOLUME/VOLATILITY
    if 'horizon_type' not in tbm:
        tbm['horizon_type'] = 'TIME' 
    if 'horizon_threshold' not in tbm:
        tbm['horizon_threshold'] = 0.0 # 0.0 implies auto-calculation or config default

    # 3. Latency Guard Defaults (Rec 2)
    # We inject these into the 'producer' section or a new 'system' section
    if 'producer' not in config:
        config['producer'] = {}
    
    # These can be used by the Engine/Worker to tune the dynamic guard
    if 'latency_guard' not in config['producer']:
        config['producer']['latency_guard'] = {
            'strict_limit': 2.0,
            'relaxed_limit': 30.0,
            'volatility_threshold': 0.0005
        }

    # 4. SURVIVAL MODE: Aggressor Protocol Risk Clamp (Rec 4)
    if 'risk_management' not in config:
        config['risk_management'] = {}
        
    # TIMEZONE FIX: Enforce Europe/Athens (EET) to match Broker Server
    # This ensures 10:00 AM matches Server Time, not Local Time
    config['risk_management']['risk_timezone'] = "Europe/Athens"

    # RISK FIX: Enforce 0.25% Base Risk (Allows ~20 losses before daily limit)
    config['risk_management']['base_risk_per_trade_percent'] = 0.0025
    
    # Enforce 0.50% Scaled Risk (Hot Hand Cap)
    config['risk_management']['scaled_risk_percent'] = 0.005

    return config

def get_config() -> Optional[Dict[str, Any]]:
    """
    Locates config.yaml, loads it, overrides sensitive values 
    with Environment Variables, and sanitizes new feature flags.
    """
    # 1. Determine Path to config.yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Priority list for config file location
    potential_paths = [
        os.getenv('CONFIG_PATH', 'config.yaml'), # Env Override
        os.path.join(os.getcwd(), 'config.yaml'), # CWD
        os.path.join(current_dir, 'config.yaml'), # Same dir
        os.path.join(current_dir, '..', '..', 'config.yaml'), # Project Root (from shared/core)
        os.path.join(current_dir, '..', '..', '..', 'config.yaml') # Fallback
    ]

    config_path = None
    for p in potential_paths:
        if os.path.exists(p):
            config_path = p
            break
    
    if not config_path:
        print("CRITICAL ERROR: 'config.yaml' not found in search paths.")
        print(f"Searched: {potential_paths}")
        return None

    # 2. Load YAML
    try:
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"FATAL: Error parsing config.yaml: {e}")
        return None
    
    if config is None:
        return None

    # 3. Inject Secrets from Environment (12-Factor App Pattern)
    # This ensures secrets stored in .env take precedence over YAML defaults

    # --- Postgres ---
    if 'postgres' not in config:
        config['postgres'] = {}
        
    config['postgres']['user'] = os.getenv('POSTGRES_USER', config['postgres'].get('user', 'postgres'))
    config['postgres']['password'] = os.getenv('POSTGRES_PASSWORD', config['postgres'].get('password', 'password'))
    config['postgres']['db'] = os.getenv('POSTGRES_DB', config['postgres'].get('db', 'trading_db'))
    config['postgres']['host'] = os.getenv('POSTGRES_HOST', config['postgres'].get('host', 'localhost'))
    config['postgres']['port'] = int(os.getenv('POSTGRES_PORT', config['postgres'].get('port', 5432)))
    
    # Construct Safe DSN for SQLAlchemy (escaped password)
    safe_pass = urllib.parse.quote_plus(config['postgres']['password'])
    
    # Async DSN (for raw psycopg2 usage if needed)
    config['postgres']['dsn'] = (
        f"dbname='{config['postgres']['db']}' "
        f"user='{config['postgres']['user']}' "
        f"password='{config['postgres']['password']}' "
        f"host='{config['postgres']['host']}' "
        f"port={config['postgres']['port']}"
    )
    
    # SQLAlchemy URL (used by WFO/Optuna)
    if 'wfo' not in config:
        config['wfo'] = {}

    # --- V17.4 MULTIPROCESSING SYNC FIX ---
    # Removed dynamic driver loading that caused Pylance errors and worker mismatch.
    # SQLAlchemy automatically defaults to psycopg2 when using the 'postgresql://' prefix.
    config['wfo']['db_url'] = (
        f"postgresql://{config['postgres']['user']}:"
        f"{safe_pass}@"
        f"{config['postgres']['host']}:{config['postgres']['port']}/"
        f"{config['postgres']['db']}"
    )

    # --- Redis ---
    if 'redis' not in config:
        config['redis'] = {}

    config['redis']['host'] = os.getenv('REDIS_HOST', config['redis'].get('host', 'localhost'))
    config['redis']['port'] = int(os.getenv('REDIS_PORT', config['redis'].get('port', 6379)))
    # Redis DB usually remains static (0), but can be overridden if needed

    # --- MetaTrader 5 (Windows Only) ---
    if 'mt5' not in config:
        config['mt5'] = {}
    
    config['mt5']['login'] = os.getenv('MT5_LOGIN', config['mt5'].get('login'))
    config['mt5']['password'] = os.getenv('MT5_PASSWORD', config['mt5'].get('password'))
    config['mt5']['server'] = os.getenv('MT5_SERVER', config['mt5'].get('server'))
    
    # Path override (e.g. if installed in a custom location)
    env_mt5_path = os.getenv('MT5_PATH')
    if env_mt5_path:
        config['mt5']['path'] = env_mt5_path

    # 4. Sanitize and Polyfill Defaults (Rec 2 & 3 Support)
    config = _sanitize_config(config)

    return config

# 5. Initialize Global Singleton
CONFIG: Optional[Dict[str, Any]] = get_config()

if CONFIG is None:
    # We raise SystemExit here because the app cannot function without config.
    # However, we print first to ensure the error is seen.
    print("Failed to initialize configuration. Exiting.")
    sys.exit(1)