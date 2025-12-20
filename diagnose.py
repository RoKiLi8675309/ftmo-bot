# =============================================================================
# FILENAME: diagnose.py
# ENVIRONMENT: Linux/WSL2 (Python 3.11)
# PATH: diagnose.py
# DEPENDENCIES: unittest, numpy, shared
# DESCRIPTION: Pre-Flight Forensic Diagnostics. Validates Math Kernels.
# Updates:
# - Fixed Entropy Test to use Constant vs Noise.
# - Fixed RiskManager API call (Tuple Unpacking).
# - FIXED: Inject mock market prices for cross-pair risk tests.
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
    TradeContext
)

# Configure Test Logging
logging.basicConfig(level=logging.ERROR)

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
        # We use a flat line. All data falls into 1 bin -> Entropy = 0.
        current_ts = 1000.0
       
        # Populate internal buffer with IDENTICAL values
        # Use np.full to guarantee identical values
        constant_prices = np.full(100, 100.0)
       
        for p in constant_prices:
            self.fe.prices.append(p)
            self.fe.entropy.buffer.append(p)
       
        # Trigger update with SAME price
        features = self.fe.update(price=100.0, timestamp=current_ts, volume=100)
        entropy_ordered = features.get('entropy', 0.5)
        # Assertion: Perfect order should be 0.0 (thanks to our fix in features.py)
        self.assertAlmostEqual(entropy_ordered, 0.0, places=4, msg="Constant data should have 0.0 entropy")

        # Case B: Random Noise (High Entropy / Disordered)
        self.fe = OnlineFeatureEngineer(window_size=50)
        np.random.seed(42)
        # Generate noise that spans a range to fill multiple bins
        noise_prices = 100.0 + np.random.normal(0, 5.0, 150)
       
        for p in noise_prices:
            self.fe.prices.append(p)
            self.fe.entropy.buffer.append(p)
        features_noise = self.fe.update(price=100.0, timestamp=current_ts, volume=100)
        entropy_noise = features_noise.get('entropy', 0.5)
        # Assertion: Noise Entropy should be significant (> 0.5 usually for high variance)
        # print(f" DEBUG: Entropy Ordered={entropy_ordered:.4f} | Noise={entropy_noise:.4f}")
        self.assertGreater(entropy_noise, 0.1, msg="Random noise should have non-zero entropy")

class TestLabeling(unittest.TestCase):
    """
    Validates Triple Barrier Method (TBM) logic.
    """
    def setUp(self):
        # FIX: Ensure barrier_len is sufficient to trigger the 'len >= 20' check
        self.tbm = StreamingTripleBarrier(
            vol_multiplier=2.0,
            barrier_len=50  # Increased from 20 to avoid edge cases
        )

    def test_take_profit_hit(self):
        # 1. Seed History to establish Volatility
        # The TBM requires >= 20 points of history.
        for _ in range(30):
            self.tbm.history.append(100.0)
       
        # Volatility of constant price is 0.
        # TBM Fallback Logic: vol = price * 0.001 = 0.1
        # Barrier Width = 0.1 * 2.0 = 0.2
        # Top Barrier = 100.2
       
        start_ts = 1000.0
       
        # Entry Tick (Trigger Event Creation)
        self.tbm.update(price=100.0, timestamp=start_ts)

        # 2. Tick below TP (100.1 < 100.2) -> No resolution
        resolved = self.tbm.update(price=100.1, timestamp=start_ts + 1)
        self.assertEqual(len(resolved), 0)

        # 3. Tick above TP (100.3 > 100.2) -> Buy Signal (1)
        # This should resolve the event created at 100.0
        resolved = self.tbm.update(price=100.3, timestamp=start_ts + 2)
       
        # In streaming, we assert >= 1 because price jumps can resolve overlaps
        self.assertGreaterEqual(len(resolved), 1)
       
        # Verify Label correctness (Label 1 = Winner)
        # resolved is list of tuples: (label, timestamp)
        labels = [r[0] for r in resolved]
        self.assertIn(1, labels)

class TestVolumeAggregation(unittest.TestCase):
    def test_volume_threshold_carry_over(self):
        # Threshold 100
        agg = VolumeBarAggregator(symbol="TEST", threshold=100)
       
        # 1. Add small tick
        bar = agg.process_tick(price=1.0, volume=50, timestamp=1000)
        self.assertIsNone(bar)
       
        # 2. Add triggering tick (50 + 60 = 110)
        # This should trigger a bar close.
        bar = agg.process_tick(price=1.0, volume=60, timestamp=1001)
        self.assertIsNotNone(bar)
       
        # The bar should be EXACTLY the threshold (100)
        self.assertEqual(bar.volume, 100.0)
       
        # 3. Verify Carry Over
        # The internal accumulator should now be 10.0 (110 - 100)
        self.assertEqual(agg.current_volume, 10.0)

class TestRiskCalculations(unittest.TestCase):
    def test_cross_pair_pip_value(self):
        # Test AUDCAD (Quote = CAD)
        # The Risk Manager needs 1 / USDCAD to calculate the dollar value of a CAD pip.
        
        ctx = TradeContext(
            symbol='AUDCAD',
            price=0.9000,
            stop_loss_price=0.8980,
            account_equity=100000.0,
            account_currency='USD',
            win_rate=0.55,
            risk_reward_ratio=1.0
        )
       
        # MOCK DATA: Provide the auxiliary price needed for conversion
        mock_prices = {
            "USDCAD": 1.3500  # 1 USD = 1.35 CAD, so 1 CAD ~ 0.74 USD
        }

        # AUDIT FIX: Pass mock_prices to strict RiskManager
        trade, risk_usd = RiskManager.calculate_rck_size(
            context=ctx,
            conf=0.6,
            volatility=0.001,
            active_correlations=0,
            market_prices=mock_prices
        )
       
        self.assertGreater(trade.volume, 0.0, "Trade volume should be > 0 with valid conversion rate")
        self.assertGreater(risk_usd, 0.0, "Risk USD should be calculated")

    def test_precision_guard(self):
        # US30 check
        digits = PrecisionGuard.get_digits("US30")
        self.assertTrue(digits in [1, 2], "Indices should have 1 or 2 digits")
       
        # BTC check
        digits = PrecisionGuard.get_digits("BTCUSD")
        self.assertEqual(digits, 2, "Crypto heuristic should work")

if __name__ == '__main__':
    print("🔬 RUNNING FORENSIC DIAGNOSTICS...")
    unittest.main()