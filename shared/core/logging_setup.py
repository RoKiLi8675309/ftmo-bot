# =============================================================================
# FILENAME: shared/core/logging_setup.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/core/logging_setup.py
# DEPENDENCIES: shared.core.config
# DESCRIPTION: Central logging configuration. Defines Emojis and Filters.
# CRITICAL: Python 3.9 Compatible. Unmutes Optuna for Research Transparency.
# AUDIT REMEDIATION:
#   - CLEANUP: Removed initialization print statement to reduce console spam.
#   - SILENCE: Added 'hmmlearn' to noisy_libs and suppressed Python warnings.
#   - FIX: Removed 'Logging Initialized' console log to stop worker spam.
# =============================================================================

import logging
import logging.handlers
import sys
import os
import warnings
from typing import Optional
from .config import CONFIG

class LogSymbols:
    """
    Emoji constants for human-readable console logs.
    Used across Windows Producer and Linux Consumer.
    """
    # Standard Levels
    INFO = "ℹ️"
    WARN = "⚠️"
    WARNING = "⚠️"
    ERROR = "❌"
    CRITICAL = "🚨"
    SUCCESS = "✅"
    
    # Trading Actions
    TRADE_BUY = "🟢"
    TRADE_SELL = "🔴"
    HOLD = "🛡️"
    PROFIT = "💰"
    LOSS = "💸"
    
    # System Status
    ONLINE = "🟢"
    OFFLINE = "⚫"
    NETWORK = "🌐"
    DATABASE = "💾"
    FROZEN = "🧊"
    SLEEP = "💤"
    CLOSE = "🏁"
    
    # Logic/Events
    NEWS = "📰"
    TIME = "⏰"
    SIGNAL = "📡"
    VPIN = "📊"
    LOCK = "🔒"
    UNLOCK = "🔓"
    UPLOAD = "📤"
    DOWNLOAD = "📥"
    
    # Sessions
    SESSION_LDN = "🇬🇧"
    SESSION_NY = "🇺🇸"
    SESSION_TOK = "🇯🇵"

class ConsoleSpamFilter(logging.Filter):
    """
    Filters out high-frequency noise from the Console output
    while allowing it to pass to the File log for debugging.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        
        # High-frequency rejection messages to suppress on console
        if "REJECT" in msg: return False
        if "Low ER" in msg: return False
        if "High VPIN" in msg: return False
        if "Low Prob" in msg: return False
        if "Ignoring" in msg: return False
        if "No tick data" in msg: return False
        
        return True

def setup_logging(component_name: str = "FTMO_Bot", log_level_override: Optional[str] = None) -> None:
    """
    Configures the root logger with:
    1. RotatingFileHandler (Full Verbosity)
    2. StreamHandler (Console, Filtered)
    """
    # Global Warning Suppression for libraries that don't use standard logging
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=UserWarning)
    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.simplefilter("ignore", category=ConvergenceWarning)
    except ImportError:
        pass

    # 1. Determine Paths and Levels
    log_dir = "logs"
    try:
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    except OSError:
        pass

    # Fetch Level from Config or Override
    if log_level_override:
        config_log_level_str = log_level_override.upper()
    else:
        config_log_level_str = CONFIG.get('logging', {}).get('level', 'INFO').upper()
    
    file_log_level = getattr(logging, config_log_level_str, logging.INFO)

    # 2. Configure Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Default to INFO

    # Clear existing handlers to prevent duplicates on reload
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)sZ [%(levelname)-8s] %(message)s (%(name)s:%(lineno)d)',
        datefmt='%Y-%m-%dT%H:%M:%S'
    )

    # 3. File Handler (Rotating)
    # Saves to logs/ftmo_bot.log (or component specific)
    log_filename = CONFIG.get('logging', {}).get('file_path', 'ftmo_bot.log')
    if component_name != "FTMO_Bot":
        # Prefix component name if distinct (e.g., research_ftmo_bot.log)
        log_filename = f"{component_name.lower()}_{log_filename}"

    log_file_path = os.path.join(log_dir, log_filename)

    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=CONFIG.get('logging', {}).get('max_size_mb', 10) * 1024 * 1024,
            backupCount=CONFIG.get('logging', {}).get('backup_count', 5),
            encoding='utf-8',
            delay=False # Ensure file is created immediately
        )
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # AUDIT FIX: Removed print statement entirely
        
    except Exception as e:
        print(f"Logging setup failed (File Handler): {e}")

    # 4. Console Handler (Standard Output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # Console usually stays at INFO/WARN
    console_handler.setFormatter(formatter)
    
    # Apply Spam Filter to Console Only
    console_handler.addFilter(ConsoleSpamFilter())
    
    root_logger.addHandler(console_handler)

    # 5. Silence Noisy Third-Party Libraries
    # These libraries are very chatty at DEBUG/INFO levels
    # AUDIT FIX: Added 'hmmlearn' and others to suppress noise
    noisy_libs = [
        "optuna", 
        "stable_baselines3", 
        "urllib3", 
        "matplotlib", 
        "requests", 
        "cloudscraper", 
        "numba",
        "parso",
        "asyncio",
        "faker",
        "PIL",
        "websockets",
        "hmmlearn",
        "river"
    ]
    
    for lib in noisy_libs:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # AUDIT FIX: Removed the "Logging Initialized" info log to prevent worker spam
    # logging.info(f"{LogSymbols.SUCCESS} Logging Initialized for {component_name} (Level: {config_log_level_str})")