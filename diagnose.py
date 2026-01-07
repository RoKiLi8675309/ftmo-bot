# =============================================================================
# FILENAME: diagnose.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: diagnose.py
# DEPENDENCIES: unittest, numpy, shared
# DESCRIPTION: Pre-Flight Forensic Diagnostics. Validates Math Kernels & Config.
# Updates:
# - Added Configuration Integrity Check (Unshackled Protocol Verification).
# - Validates Risk Parameters (1.0% Base, 1.5% Scaled).
# - Validates Regime Settings (DISABLED).
# CRITICAL: Must pass (Exit Code 0) before 'run_pipeline.sh' proceeds.
# =============================================================================
import unittest
import numpy as np
import pandas as pd
import logging
import time
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
    CONFIG
)

# Configure Test Logging
logging.basicConfig(level=logging.ERROR)

class TestConfigurationIntegrity(unittest.TestCase):
    """
    V12.7 PRE-FLIGHT CHECK: Verifies that config.yaml is correctly loaded
    with the Unshackled Protocol parameters.
    """
    def test_unshackled_risk_params(self):
        """
        Verify Risk Management is set to 1.0% Base / 1.5% Scaled.
        """
        risk_conf = CONFIG.get('risk_management', {})
        base_risk = risk_conf.get('base_risk_per_trade_percent')
        scaled_risk = risk_conf.get('scaled_risk_percent')
        
        print(f"   [CONF] Base Risk: {base_risk*100:.1f}% | Scaled Risk: {scaled_risk*100:.1f}%")
        
        self.assertEqual(base_risk, 0.010, "CRITICAL: Base Risk must be 1.0% (0.010)")
        self.assertEqual(scaled_risk, 0.015, "CRITICAL: Scaled Risk must be 1.5% (0.015)")

    def test_regime_settings(self):
        """
        Verify Regime Enforcement is DISABLED for maximum AI adaptability.
        """
        phx_conf = CONFIG.get('phoenix_strategy', {})
        regime_mode = phx_conf.get('regime_enforcement')
        
        print(f"   [CONF] Regime Mode: {regime_mode}")
        
        self.assertEqual(regime_mode, "DISABLED", "CRITICAL: Regime Enforcement must be DISABLED")

    def test_confidence_gates(self):
        """
        Verify Confidence Gating is removed (Min Probability = 0.00).
        """
        ml_conf = CONFIG.get('online_learning', {})
        min_prob = ml_conf.get('min_calibrated_probability')
        
        print(f"   [CONF] Min Probability: {min_prob}")
        
        self.assertLessEqual(min_prob, 0.01, "CRITICAL: Confidence Gate must be effectively disabled (<= 0.01)")

class TestFeatureEngineering(unittest.TestCase):
    """
    Validates the 'Golden Six' feature calculations.
    """
    def setUp(self):
        self.fe = OnlineFeatureEngineer(window_size=50)

    def test_entropy_calculation(self):
        """
        FORENSIC CHECK: Entropy should be 0.0 for constant data, high for noise.
        """
        # Case A: Constant Data (Extremely Low Entropy / Ordered)
        current_ts = 1000.0
        constant_prices = np.full(100, 100.0)
        
        for p in constant_prices:
            self.fe.prices.append(p)
            self.fe.entropy.buffer.append(p)
        
        features = self.fe.update(price=100.0, timestamp=current_ts, volume=100)
        entropy_ordered = features.get('entropy', 0.5)
        self.assertAlmostEqual(entropy_ordered, 0.0, places=4, msg="Constant data should have 0.0 entropy")

        # Case B: Random Noise (High Entropy / Disordered)
        self.fe = OnlineFeatureEngineer(window_size=50)
        np.random.seed(42)
        noise_prices = 100.0 + np.random.normal(0, 5.0, 150)
        
        for p in noise_prices:
            self.fe.prices.append(p)
            self.fe.entropy.buffer.append(p)
        features_noise = self.fe.update(price=100.0, timestamp=current_ts, volume=100)
        entropy_noise = features_noise.get('entropy', 0.5)
        self.assertGreater(entropy_noise, 0.1, msg="Random noise should have non-zero entropy")

class TestLabeling(unittest.TestCase):
    """
    Validates Triple Barrier Method (TBM) logic.
    """
    def setUp(self):
        self.tbm = StreamingTripleBarrier(
            vol_multiplier=2.0,
            barrier_len=50
        )

    def test_take_profit_hit(self):
        for _ in range(30):
            self.tbm.history.append(100.0)
        
        start_ts = 1000.0
        self.tbm.update(price=100.0, timestamp=start_ts)

        # Tick below TP -> No resolution
        resolved = self.tbm.update(price=100.1, timestamp=start_ts + 1)
        self.assertEqual(len(resolved), 0)

        # Tick above TP -> Buy Signal
        resolved = self.tbm.update(price=100.3, timestamp=start_ts + 2)
        self.assertGreaterEqual(len(resolved), 1)
        labels = [r[0] for r in resolved]
        self.assertIn(1, labels)

class TestVolumeAggregation(unittest.TestCase):
    def test_volume_threshold_carry_over(self):
        agg = VolumeBarAggregator(symbol="TEST", threshold=100)
        
        # 1. Add small tick
        bar = agg.process_tick(price=1.0, volume=50, timestamp=1000)
        self.assertIsNone(bar)
        
        # 2. Add triggering tick (50 + 60 = 110)
        bar = agg.process_tick(price=1.0, volume=60, timestamp=1001)
        self.assertIsNotNone(bar)
        self.assertEqual(bar.volume, 100.0)
        
        # 3. Verify Carry Over
        self.assertEqual(agg.current_volume, 10.0)

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
            market_prices=mock_prices
        )
        
        self.assertGreater(trade.volume, 0.0, "Trade volume should be > 0")
        self.assertGreater(risk_usd, 0.0, "Risk USD should be calculated")

    def test_precision_guard(self):
        digits = PrecisionGuard.get_digits("US30")
        self.assertTrue(digits in [1, 2], "Indices should have 1 or 2 digits")
        
        digits = PrecisionGuard.get_digits("BTCUSD")
        self.assertEqual(digits, 2, "Crypto heuristic should work")

if __name__ == '__main__':
    print(f"\n{LogSymbols.SEARCH} RUNNING PHOENIX V12.7 PRE-FLIGHT DIAGNOSTICS...")
    unittest.main(verbosity=2)