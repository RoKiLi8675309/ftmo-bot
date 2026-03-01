# =============================================================================
# FILENAME: diagnose.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: diagnose.py
# DEPENDENCIES: unittest, numpy, redis, shared
# DESCRIPTION: Pre-Flight Forensic Diagnostics & PIPELINE VERIFICATION.
# 
# PHOENIX V17.5 UPDATE (FTMO SURVIVAL PROTOCOL):
# 1. SIZING VALIDATION: Ensures sizing_method is 'risk_percentage' to hit FTMO targets.
# 2. CHOKE GUARD: Verifies trailing stops activate quickly at 1.5R.
# 3. GATE VALIDATION: Ensures ADX (22) and Hurst (0.58) are tightened to avoid chop.
# 4. REGIME VALIDATION: Ensures Regime Enforcement is set to HARD.
# =============================================================================
import unittest
import numpy as np
import pandas as pd
import logging
import time
import json
import uuid
import redis
from collections import deque

# Shared Imports
from shared import (
    OnlineFeatureEngineer,
    VPINMonitor,
    StreamingTripleBarrier,
    RiskManager,
    LogSymbols,
    VolumeBarAggregator,
    PrecisionGuard,
    SystemDiagnose,
    TradeContext,
    CONFIG,
    get_redis_connection
)

# Configure Test Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Diagnose")

class TestConfigurationIntegrity(unittest.TestCase):
    """
    V17.5 PRE-FLIGHT CHECK: Verifies that config.yaml is correctly loaded
    with the FTMO Survival Protocol parameters.
    """
    def test_aggressor_risk_params(self):
        """Verify Risk Management uses dynamic risk_percentage and correct sizing."""
        risk_conf = CONFIG.get('risk_management', {})
        sizing_method = risk_conf.get('sizing_method')
        base_risk = risk_conf.get('base_risk_per_trade_percent')
        scaled_risk = risk_conf.get('scaled_risk_percent')
        
        print(f"    [CONF] Sizing Method: {sizing_method} | Base Risk: {base_risk*100:.2f}% | Scaled Risk: {scaled_risk*100:.2f}%")
        
        # V17.1/17.5 UPDATE: Must use risk_percentage. Fixed 0.01 lots cannot pass FTMO.
        self.assertEqual(sizing_method, "risk_percentage", "CRITICAL: Must use dynamic risk_percentage to pass FTMO")
        
        # Ensure we are risking 0.25% per trade to allow sufficient statistical attempts
        self.assertEqual(base_risk, 0.0025, "CRITICAL: Base Risk must be 0.25% (0.0025) for FTMO scaling")
        
        # Scaled Risk capped at 0.50% (0.005) for Hot Hand
        self.assertEqual(scaled_risk, 0.005, "CRITICAL: Scaled Risk must be 0.50% (0.005)")

    def test_v17_5_execution_gates(self):
        """Verify the V17.5 tightened gates are active to prevent chop trading."""
        phx_conf = CONFIG.get('phoenix_strategy', {})
        features_conf = CONFIG.get('features', {})
        risk_conf = CONFIG.get('risk_management', {})
        
        adx_thresh = features_conf.get('adx', {}).get('threshold', 0)
        hurst_thresh = phx_conf.get('hurst_breakout_threshold', 0)
        trail_act = risk_conf.get('trailing_stop', {}).get('activation_r', 0)
        
        print(f"    [CONF] ADX: {adx_thresh} | Hurst: {hurst_thresh} | Trail Act: {trail_act}R")
        
        self.assertGreaterEqual(adx_thresh, 22.0, "CRITICAL: ADX threshold must be >= 22.0 to filter out chop")
        self.assertGreaterEqual(hurst_thresh, 0.58, "CRITICAL: Hurst breakout threshold must be >= 0.58 to capture strict trends")
        self.assertLessEqual(trail_act, 1.5, "CRITICAL: Trailing stop activation must be <= 1.5R to lock in profits quickly")

    def test_regime_settings(self):
        """Verify Regime Enforcement is HARD for maximum protection."""
        phx_conf = CONFIG.get('phoenix_strategy', {})
        regime_mode = phx_conf.get('regime_enforcement')
        print(f"    [CONF] Regime Mode: {regime_mode}")
        self.assertEqual(regime_mode, "HARD", "CRITICAL: Regime Enforcement must be HARD")

    def test_leverage_map_integrity(self):
        """V14.0+: Verify Leverage Map exists for Asset Classes."""
        risk_conf = CONFIG.get('risk_management', {})
        lev_map = risk_conf.get('leverage', {})
        required_keys = ['default', 'minor', 'gold', 'indices', 'crypto']
        
        print(f"    [CONF] Leverage Keys: {list(lev_map.keys())}")
        for k in required_keys:
            self.assertIn(k, lev_map, f"CRITICAL: Missing leverage config for '{k}'")


class TestInfrastructureAndExecution(unittest.TestCase):
    """
    V12.18 PIPELINE FORENSICS:
    Validates the connection between Linux Logic and Windows Execution.
    """
    def setUp(self):
        self.r = get_redis_connection(
            host=CONFIG['redis']['host'],
            port=CONFIG['redis']['port'],
            db=0,
            decode_responses=True
        )
        self.stream_key = CONFIG['redis']['trade_request_stream']

    def test_1_redis_connectivity(self):
        """Can we talk to the database?"""
        try:
            self.r.ping()
            print(f"    {LogSymbols.ONLINE} Redis Connection: ESTABLISHED ({CONFIG['redis']['host']})")
        except redis.ConnectionError:
            self.fail(f"{LogSymbols.CRITICAL} CRITICAL: Cannot connect to Redis. Check 'redis' container.")

    def test_2_windows_producer_heartbeat(self):
        """
        CRITICAL: Is the Windows machine writing to THIS Redis instance?
        Checks 'producer:heartbeat' timestamp.
        """
        heartbeat_key = CONFIG.get('producer', {}).get('heartbeat_key', 'producer:heartbeat')
        last_beat = self.r.get(heartbeat_key)
        
        if not last_beat:
            print(f"    {LogSymbols.OFFLINE} HEARTBEAT MISSING: Windows Producer has NOT written to key '{heartbeat_key}'.")
            print("    -> CAUSE: Windows is either offline OR connecting to a different Redis IP.")
            # We don't fail, but we warn LOUDLY
        else:
            delta = time.time() - float(last_beat)
            status = "ALIVE" if delta < 30 else f"STALE ({int(delta)}s ago)"
            icon = LogSymbols.ONLINE if delta < 30 else LogSymbols.WARNING
            print(f"    {icon} Windows Producer Status: {status}")
            
            if delta > 60:
                print(f"    {LogSymbols.CRITICAL} WARNING: Windows Producer is effectively DEAD (Last beat > 60s ago).")

    def test_3_verify_redis_write_permissions(self):
        """
        Verifies Redis Write capability WITHOUT sending live orders.
        Replaces the costly Probe Signal injection.
        """
        print(f"\n    {LogSymbols.DATABASE} VERIFYING REDIS WRITE PERMISSIONS (NO COST MODE)...")
        test_key = f"diagnose:write_test:{uuid.uuid4()}"
        test_val = "WRITE_TEST_OK"

        try:
            # 1. Write
            self.r.setex(test_key, 10, test_val)
            # 2. Read
            val = self.r.get(test_key)
            # 3. Verify
            self.assertEqual(val, test_val, "Redis Write/Read mismatch")
            # 4. Clean
            self.r.delete(test_key)

            print(f"    {LogSymbols.SUCCESS} Redis Write Check: PASSED")
            print(f"    -> Connectivity is healthy. No Limit Orders sent to Broker.")

        except Exception as e:
            self.fail(f"Redis Write Check Failed: {e}")

    def test_4_inspect_stream_flow(self):
        """Are signals actually landing in the stream?"""
        slen = self.r.xlen(self.stream_key)
        print(f"    {LogSymbols.VPIN} Stream '{self.stream_key}' Depth: {slen} messages")
        
        if slen > 0:
            last = self.r.xrevrange(self.stream_key, count=1)
            print(f"    -> Latest Message: {last[0][1]}")
        else:
            print("    -> Stream is empty.")


class TestFeatureEngineering(unittest.TestCase):
    """Validates the 'Golden Six' feature calculations."""
    def setUp(self):
        self.fe = OnlineFeatureEngineer(window_size=50)

    def test_entropy_calculation(self):
        # Case A: Constant Data (Extremely Low Entropy / Ordered)
        current_ts = 1000.0
        constant_prices = np.full(100, 100.0)
        
        for p in constant_prices:
            self.fe.prices.append(p)
            self.fe.entropy.buffer.append(p)
        
        features = self.fe.update(price=100.0, timestamp=current_ts, volume=100)
        entropy_ordered = features.get('entropy', 0.5)
        self.assertAlmostEqual(entropy_ordered, 0.0, places=4, msg="Constant data should have 0.0 entropy")

        # Case B: Random Noise
        self.fe = OnlineFeatureEngineer(window_size=50)
        np.random.seed(42)
        noise_prices = 100.0 + np.random.normal(0, 5.0, 150)
        
        for p in noise_prices:
            self.fe.prices.append(p)
            self.fe.entropy.buffer.append(p)
        features_noise = self.fe.update(price=100.0, timestamp=current_ts, volume=100)
        entropy_noise = features_noise.get('entropy', 0.5)
        self.assertGreater(entropy_noise, 0.1, msg="Random noise should have non-zero entropy")


class TestRiskCalculations(unittest.TestCase):
    def test_cross_pair_pip_value(self):
        ctx = TradeContext(
            symbol='AUDCAD',
            price=0.9000,
            stop_loss_price=0.8980,
            account_equity=100000.0,
            account_currency='USD',
            win_rate=0.55,
            risk_reward_ratio=1.0
        )
        mock_prices = {"USDCAD": 1.3500}
        
        trade, risk_usd = RiskManager.calculate_rck_size(
            context=ctx,
            conf=0.6,
            volatility=0.001,
            active_correlations=0,
            market_prices=mock_prices,
            free_margin=100000.0 
        )
        self.assertGreater(trade.volume, 0.0, "Trade volume should be > 0")
        self.assertGreater(risk_usd, 0.0, "Risk USD should be calculated")

    def test_precision_guard(self):
        digits = PrecisionGuard.get_digits("US30")
        self.assertTrue(digits in [1, 2], "Indices should have 1 or 2 digits")
        digits = PrecisionGuard.get_digits("BTCUSD")
        self.assertEqual(digits, 2, "Crypto heuristic should work")


if __name__ == '__main__':
    print(f"\n🔍 RUNNING PHOENIX V17.5 PIPELINE DIAGNOSTICS...")
    unittest.main(verbosity=2)