# =============================================================================
# FILENAME: engines/live/main_live.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: engines/live/main_live.py
# DEPENDENCIES: engines.live.engine
# DESCRIPTION: Entry point for the Live Trading Consumer.
# CRITICAL: Python 3.11 Compatible. Handles Graceful Shutdowns.
# =============================================================================
import sys
import os
import signal
import logging
import time

# Ensure the project root is in sys.path to resolve 'shared' and 'engines' modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after path fix
try:
    from engines.live.engine import LiveTradingEngine
    from shared import LogSymbols
except ImportError as e:
    print(f"CRITICAL: Failed to import dependencies. Ensure you are running from the project root or 'shared' is accessible.\nError: {e}")
    sys.exit(1)

# Setup basic logging for the entry point before Engine takes over
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("Main")

def main():
    """
    Bootstraps the Live Trading Engine.
    """
    log.info(f"{LogSymbols.ONLINE} Initializing Live Trading Consumer (Linux/WSL2)...")
    log.info(f"{LogSymbols.INFO} Python Version: {sys.version}")

    # Initialize Engine
    try:
        engine = LiveTradingEngine()
    except Exception as e:
        log.critical(f"{LogSymbols.CRITICAL} Failed to initialize Engine: {e}", exc_info=True)
        sys.exit(1)

    # Signal Handler for Graceful Shutdown (Ctrl+C or Docker Stop)
    def signal_handler(sig, frame):
        print("\n")  # Newline for CLI cleanliness
        log.info(f"{LogSymbols.CLOSE} Shutdown Signal Received ({sig}). Stopping Engine...")
        engine.shutdown()
        # Allow engine loop to exit naturally, but force exit if it hangs
        time.sleep(2)
        sys.exit(0)

    # Register Signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start Main Loop
    try:
        engine.run()
    except KeyboardInterrupt:
        # Catch explicit keyboard interrupt if signal handler doesn't catch it first
        engine.shutdown()
    except Exception as e:
        log.critical(f"{LogSymbols.CRITICAL} Engine Crash: {e}", exc_info=True)
        engine.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()