# =============================================================================
# FILENAME: diagnose.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: diagnose.py
# DEPENDENCIES: unittest, numpy, redis, shared
# DESCRIPTION: Pre-Flight Forensic Diagnostics & PIPELINE VERIFICATION.
# 
# PHOENIX V20.17 UPDATE (BINARY BASE + META FILTER PROTOCOL):
# 1. SIZING VALIDATION: Accepts 'fixed_lots' for the data collection phase.
# 2. CHOKE GUARD: Verifies trailing stops activate quickly at <= 1.0R.
# 3. ML FREEDOM ENFORCEMENT: Strictly asserts ADX == 0.0 to guarantee 
#    the Machine Learning model has absolute control over trade execution.
# 4. REGIME ENFORCEMENT: Asserts Regime Enforcement is completely DISABLED.
# 5. ANTI-MACHINE-GUNNING: Asserts base 15-minute cooldown (exponentially scales on losses).
# 6. SPREAD TRAP CURE: Asserts absolute minimum stop loss floor of 20.0 pips.
# 7. META FILTER CHECK: Asserts meta_labeling_threshold exists in the search space.
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
    RiskManager,
    LogSymbols,
    PrecisionGuard,
    TradeContext,
    CONFIG,
    get_redis_connection
)

# Configure Test Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Diagnose")

class TestConfigurationIntegrity(unittest.TestCase):
    """
    V20.17 PRE-FLIGHT CHECK: Verifies that config.yaml is correctly loaded
    with the Binary EV Protocol parameters to prevent trade starvation.
    """
    def test_aggressor_risk_params(self):
        """Verify Risk Management uses valid sizing (risk_percentage or fixed_lots)."""
        risk_conf = CONFIG.get('risk_management', {})
        sizing_method = risk_conf.get('sizing_method')
        base_risk = risk_conf.get('base_risk_per_trade_percent', 0.0)
        scaled_risk = risk_conf.get('scaled_risk_percent', 0.0)
        
        print(f"    [CONF] Sizing Method: {sizing_method} | Base Risk: {base_risk*100:.2f}% | Scaled Risk: {scaled_risk*100:.2f}%")
        
        # V20.10 REQUIREMENT: Allow fixed_lots for burn in, or risk_percentage for scale.
        self.assertIn(sizing_method, ["risk_percentage", "fixed_lots"], "CRITICAL: Must use dynamic 'risk_percentage' or 'fixed_lots' for data collection")
        
        # Ensure we are risking at least 0.25% per trade to allow sufficient statistical attempts (when using percentage)
        self.assertGreaterEqual(base_risk, 0.0025, "CRITICAL: Base Risk must be >= 0.25% (0.0025) for FTMO scaling")
        
        # Scaled Risk should be greater than or equal to base risk
        self.assertGreaterEqual(scaled_risk, base_risk, "CRITICAL: Scaled Risk must be >= Base Risk")

    def test_v20_10_ml_freedom_protocol(self):
        """Verify the V20.10 ML Freedom protocol is active to prevent trade starvation."""
        phx_conf = CONFIG.get('phoenix_strategy', {})
        features_conf = CONFIG.get('features', {})
        risk_conf = CONFIG.get('risk_management', {})
        
        adx_thresh = features_conf.get('adx', {}).get('threshold', 0)
        hurst_thresh = phx_conf.get('hurst_breakout_threshold', 0)
        trail_act = risk_conf.get('trailing_stop', {}).get('activation_r', 0)
        
        print(f"    [CONF] ADX: {adx_thresh} | Hurst: {hurst_thresh} | Trail Act: {trail_act}R")
        
        # V20.10 ML FREEDOM GATES: 
        # We strictly enforce ADX == 0.0 so heuristic filters never choke the ML model.
        self.assertEqual(adx_thresh, 0.0, "CRITICAL: ADX threshold MUST be exactly 0.0 to grant ML full freedom (Unchoked Standard).")
        self.assertGreaterEqual(hurst_thresh, 0.50, "CRITICAL: Hurst breakout threshold must be >= 0.50")
        self.assertLessEqual(trail_act, 1.0, "CRITICAL: Trailing stop activation must be <= 1.0R to lock in profits quickly")

    def test_v20_17_expert_trader_protocol(self):
        """Verify the V20.17 Expert Trader protocol is active to prevent machine gunning."""
        risk_conf = CONFIG.get('risk_management', {})
        cooldown = risk_conf.get('loss_cooldown_minutes', 0)
        min_sl = risk_conf.get('min_stop_loss_pips', 0.0)
        
        print(f"    [CONF] Cooldown: {cooldown}m | Min SL Floor: {min_sl} pips")
        
        # V20.17 EXPERT TRADER GATES:
        self.assertGreaterEqual(cooldown, 15, "CRITICAL: Base loss cooldown must be >= 15 minutes to prevent machine-gunning via exponential scaling.")
        self.assertGreaterEqual(min_sl, 20.0, "CRITICAL: Minimum Stop Loss must be >= 20.0 pips to escape spread traps.")

    def test_v20_17_binary_meta_protocol(self):
        """Verify Meta-Labeling Threshold is in the search space."""
        space = CONFIG.get('optimization_search_space', {})
        self.assertIn('meta_labeling_threshold', space, "CRITICAL: meta_labeling_threshold must be in the search space to tune the Binary EV filter.")
        print(f"    [CONF] Meta-Labeling Tuning Enabled: {list(space['meta_labeling_threshold'].keys())}")

    def test_regime_settings(self):
        """Verify Regime Enforcement is DISABLED to prevent heuristic over-filtering."""
        phx_conf = CONFIG.get('phoenix_strategy', {})
        regime_mode = phx_conf.get('regime_enforcement')
        print(f"    [CONF] Regime Mode: {regime_mode}")
        self.assertEqual(regime_mode, "DISABLED", "CRITICAL: Regime Enforcement MUST be DISABLED in V20.10 to allow ML to dictate trades.")

    def test_leverage_map_integrity(self):
        """Verify Leverage Map exists for Asset Classes."""
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
        heartbeat_key = CONFIG.get('redis', {}).get('heartbeat_key', 'producer:heartbeat')
        last_beat = self.r.get(heartbeat_key)
        
        if not last_beat:
            print(f"    {LogSymbols.OFFLINE} HEARTBEAT MISSING: Windows Producer has NOT written to key '{heartbeat_key}'.")
            print("    -> CAUSE: Windows is either offline OR connecting to a different Redis IP.")
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
        """
        V20.2 FIX: Saturated buffer test.
        Ensures the FE correctly returns low entropy for constant data 
        after the buffer is saturated via the update() call.
        """
        current_ts = 1000.0
        features = None
        
        # Saturated the internal buffer (window_size=50) by calling update()
        # Shannon Entropy of a constant sequence must be 0.0.
        for i in range(60):
            features = self.fe.update(price=100.0, timestamp=current_ts + i, volume=100)
        
        # Verify the key exists and the value is physically sound
        self.assertIsNotNone(features, "FE update should return a feature dict after saturation")
        entropy_val = features.get('entropy', 0.5)
        
        # Logic check: extremely low variation should yield very low entropy (< 0.2)
        # If we got 0.5, it means the key was missing or default hit.
        self.assertLess(entropy_val, 0.2, msg=f"Constant data should have low entropy, but got {entropy_val}. (Note: 0.5 indicates a missing key/default)")

    def test_log_returns(self):
        """Verify log return calculation parity."""
        current_ts = 1000.0
        self.fe.update(price=100.0, timestamp=current_ts, volume=100)
        features = self.fe.update(price=101.0, timestamp=current_ts+1, volume=100)
        
        expected_ret = np.log(101.0 / 100.0)
        self.assertAlmostEqual(features['log_ret'], expected_ret, places=6)


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
    print(f"\n🔍 RUNNING PHOENIX V20.17 PIPELINE DIAGNOSTICS...")
    unittest.main(verbosity=2)