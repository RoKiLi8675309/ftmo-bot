#!/bin/bash
# =============================================================================
# FILENAME: run_pipeline.sh
# ARCHITECTURE: Linux/WSL2 Orchestrator
# PURPOSE: Enforces Causality & Microstructure Checks BEFORE Online Optimization.
# =============================================================================
set -e

# --- CONSOLE FEEDBACK (UNFREEZE INDICATOR) ---
echo "================================================================="
echo " 🚀 STARTING FTMO FORENSIC PIPELINE (V13.1 SURVIVAL MODE)"
echo " 🕒 $(date)"
echo "================================================================="

# --- HELPERS ---
function show_help {
    echo "Usage: ./run_pipeline.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --fresh-start    Purge existing Optuna studies and models (Starts from scratch)."
    echo "  --test-only      Run only the Pre-Flight Diagnostics and exit."
    echo "  --wfo            Run Walk-Forward Optimization (Rolling Window)."
    echo "  --help           Show this message."
    echo ""
}

# --- PARSE ARGUMENTS ---
FRESH_START_FLAG=""
TEST_ONLY_MODE=false
WFO_FLAG=""

for arg in "$@"
do
    case $arg in
        --fresh-start)
            echo "!!! WARNING: --fresh-start flag detected. !!!"
            echo "This will PURGE all previous Optuna studies and models."
            read -p "Are you sure? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Aborting."
                exit 1
            fi
            FRESH_START_FLAG="--fresh-start"
            # Clean up local models directory (DB cleaned by Python script)
            rm -f models/*.pkl models/*.json
            echo "Artifacts purged."
            ;;
        --test-only)
            TEST_ONLY_MODE=true
            ;;
        --wfo)
            WFO_FLAG="--wfo"
            echo ">>> WALK-FORWARD OPTIMIZATION MODE ENABLED <<<"
            ;;
        --help)
            show_help
            exit 0
            ;;
    esac
done

# --- ENVIRONMENT SETUP (CRITICAL FIX FOR CONDA) ---
# Smart detection of Python interpreter
if [ -n "$CONDA_PREFIX" ]; then
    TARGET_PYTHON="$CONDA_PREFIX/bin/python" # CRITICAL FIX: Use active Conda env
elif [ -n "$VIRTUAL_ENV" ]; then
    TARGET_PYTHON="python" # Use active venv
elif [ -f "venv/bin/python" ]; then
    TARGET_PYTHON="venv/bin/python"
elif [ -f "anaconda3/envs/algo_env_stable/bin/python" ]; then
    TARGET_PYTHON="anaconda3/envs/algo_env_stable/bin/python"
elif command -v python3 &> /dev/null; then
    TARGET_PYTHON="python3"
else
    TARGET_PYTHON="python"
fi

echo "Using Python Interpreter: $TARGET_PYTHON"

# --- V17.4 PRE-FLIGHT C-LIBRARY CHECK ---
echo "Verifying PostgreSQL C-Libraries..."
if ! "$TARGET_PYTHON" -c "import psycopg2" &> /dev/null; then
    echo ""
    echo "❌ CRITICAL: psycopg2 C-extensions failed to load."
    echo "   Pip installed the wrapper, but WSL2 is missing the underlying C-libraries (libpq)."
    echo ""
    echo "💡 THE FIX: Run this exact command to let Conda inject the missing C-libraries:"
    echo "   conda install -y -c conda-forge psycopg2"
    echo ""
    echo "Aborting pipeline to prevent data corruption and worker crashes."
    exit 1
fi
echo "✅ Database Drivers OK."

# --- STAGE 1: PRE-FLIGHT DIAGNOSTICS (THE FORENSIC GATE) ---
echo ""
echo "================================================================="
echo " EXECUTING STAGE 1: Forensic Diagnostics (Math & Causality)      "
echo "================================================================="
# Runs unit tests for Entropy, VPIN, OFI, and TBM
"$TARGET_PYTHON" diagnose.py

if [ $? -ne 0 ]; then
    echo "❌ CRITICAL FAILURE: Pre-Flight Diagnostics Failed."
    echo "   The pipeline has been halted to prevent model corruption."
    echo "   Check the logs above for specific math errors."
    exit 1
fi

echo "--- STAGE 1 COMPLETED SUCCESSFULLY: Math is Sound. ---"

if [ "$TEST_ONLY_MODE" = true ]; then
    echo "Test-only mode requested. Exiting."
    exit 0
fi

# --- STAGE 2: ADAPTIVE LEARNING (TRAINING / OPTIMIZATION / WFO) ---
echo ""
echo "================================================================="
echo " EXECUTING STAGE 2: Online Adaptive Learning (River)             "
echo "================================================================="

if [ -n "$WFO_FLAG" ]; then
    # Run Walk-Forward Optimization
    echo "   Running Walk-Forward Analysis (Rolling Window)..."
    "$TARGET_PYTHON" engines/research/main_research.py --wfo
else
    # Run Standard Optimization/Training
    echo "   Running Global Optimization (Full History)..."
    # Pass flags explicitly
    if [ -n "$FRESH_START_FLAG" ]; then
        "$TARGET_PYTHON" engines/research/main_research.py --train --fresh-start
    else
        "$TARGET_PYTHON" engines/research/main_research.py --train
    fi
fi

if [ $? -ne 0 ]; then
    echo "❌ TRAINING FAILED. Check logs above."
    exit 1
fi

echo "--- STAGE 2 COMPLETED SUCCESSFULLY ---"
echo ""

# --- STAGE 3: FINAL BACKTEST (VERIFICATION) ---
# Only run if not in WFO mode (WFO does its own reporting)
if [ -z "$WFO_FLAG" ]; then
    echo "================================================================="
    echo " EXECUTING STAGE 3: Final Verification Backtest                  "
    echo "================================================================="
    "$TARGET_PYTHON" engines/research/main_research.py --backtest

    # Verify Artifact Generation (Check for at least one model)
    if ls models/river_pipeline_*.pkl 1> /dev/null 2>&1; then
        echo "--- SUCCESS: River Classifiers Saved. ---"
    else
        echo "--- FAILURE: Model files missing. ---"
        exit 1
    fi
fi

echo ""
echo "================================================================="
echo "             ONLINE PIPELINE COMPLETE                        "
echo "================================================================="
echo "1. Models:  models/"
echo "2. Reports: reports/"
echo "3. Logs:    logs/"
echo "================================================================="