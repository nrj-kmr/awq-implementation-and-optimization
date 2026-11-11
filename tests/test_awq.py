"""
Unit tests for the awq_lab quantization package.
Runs cleanly using Python's built-in unittest framework without external test dependencies.
"""

import unittest
import torch
import torch.nn as nn

from awq_lab import (
    quantize_weight_group,
    pack_int4_weights,
    unpack_int4_weights,
    find_optimal_awq_scale,
    apply_awq_to_weights,
    ActivationProfiler
)


class TestAWQQuantization(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def test_quantize_shape_and_range(self):
        """Verify that quantized weight tensor preserves dimensions and remains close to original."""
        W = torch.randn(128, 256, device=self.device)
        W_q = quantize_weight_group(W, n_bits=4, group_size=64)
        
        self.assertEqual(W.shape, W_q.shape)
        # Quantization error should be reasonable for standard normal distribution
        mse = torch.mean((W - W_q) ** 2).item()
        self.assertLess(mse, 0.05)

    def test_bitwidth_monotonicity(self):
        """Verify that higher bit-width strictly reduces reconstruction error (4-bit vs 2-bit)."""
        W = torch.randn(64, 128, device=self.device)
        W_q4 = quantize_weight_group(W, n_bits=4, group_size=64)
        W_q2 = quantize_weight_group(W, n_bits=2, group_size=64)
        
        mse4 = torch.mean((W - W_q4) ** 2).item()
        mse2 = torch.mean((W - W_q2) ** 2).item()
        self.assertLess(mse4, mse2, "4-bit quantization must yield lower MSE than 2-bit")

    def test_int4_pack_unpack(self):
        """Verify lossless INT4 packing and unpacking into INT8 storage."""
        # Create random integer indices in range [0, 15] as int32
        q_weights = torch.randint(0, 16, (32, 64), dtype=torch.int32)
        packed = pack_int4_weights(q_weights)
        
        # Packed byte count should be exactly half of total elements (32 * 64 / 2 = 1024)
        self.assertEqual(packed.numel(), q_weights.numel() // 2)
        
        unpacked = unpack_int4_weights(packed, q_weights.shape)
        self.assertTrue(torch.equal(q_weights.float(), unpacked), "Unpacked INT4 values must match original")

    def test_awq_outlier_protection(self):
        """
        Verify that when activations have salient outlier channels,
        AWQ protective scaling strictly reduces Frobenius output error compared to naive RTN.
        """
        in_dim = 128
        out_dim = 64
        seq_len = 32
        
        # Create weights
        W = torch.randn(out_dim, in_dim, device=self.device)
        
        # Create activations with extreme outliers in channels 5, 12, 45 (10-20x larger)
        X = torch.randn(seq_len, in_dim, device=self.device)
        X[:, 5] *= 15.0
        X[:, 12] *= 18.0
        X[:, 45] *= 20.0
        
        Y_true = X @ W.T
        
        # 1. Naive RTN
        W_rtn = quantize_weight_group(W, n_bits=4, group_size=64)
        mse_rtn = torch.mean((Y_true - X @ W_rtn.T) ** 2).item()
        
        # 2. AWQ Grid Search
        scale, alpha, hist = find_optimal_awq_scale(W, X, n_bits=4, group_size=64, grid_steps=10)
        best_mse = min(hist.values())
        
        self.assertLess(best_mse, mse_rtn, "AWQ optimal scaling must reduce MSE relative to naive RTN")
        self.assertGreater(alpha, 0.0, "Optimal alpha should be non-zero for activation outliers")

    def test_activation_profiler_hook(self):
        """Verify that ActivationProfiler registers hooks, records activations, and detaches properly."""
        layer = nn.Linear(32, 16).to(self.device)
        profiler = ActivationProfiler(layer)
        profiler.register_layer("linear_test", layer)
        
        dummy_input = torch.randn(4, 32, device=self.device)
        with torch.no_grad():
            layer(dummy_input)
            
        cached_X = profiler.cached_activations["linear_test"]
        self.assertEqual(cached_X.shape, (4, 32))
        
        stats = profiler.compute_channel_stats(cached_X)
        self.assertIn("mean", stats)
        self.assertIn("max", stats)
        self.assertIn("max_to_mean_ratio", stats)
        self.assertGreater(stats["max_to_mean_ratio"], 1.0)
        
        profiler.remove_hooks()
        self.assertEqual(len(profiler.hooks), 0)


if __name__ == "__main__":
    unittest.main()
