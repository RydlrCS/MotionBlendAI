import unittest
import torch
import numpy as np
from project.ganimator.train import SPADEGenerator, create_skeleton_id_map

class TestSPADEGANimator(unittest.TestCase):
    def setUp(self):
        # Set up a generator with known dimensions
        self.motion_dim = 10
        self.cond_dim = 2
        self.seq_len = 60
        self.generator = SPADEGenerator(motion_dim=self.motion_dim, cond_dim=self.cond_dim, seq_len=self.seq_len)

    def test_forward_shape(self):
        """
        Test that the generator forward pass returns the correct output shape.
        """
        batch_size = 3
        seq1 = torch.randn(batch_size, self.seq_len, self.motion_dim)
        seq2 = torch.randn(batch_size, self.seq_len, self.motion_dim)
        cond = create_skeleton_id_map(batch_size, self.seq_len, 0.5)
        out = self.generator(seq1, seq2, cond)
        self.assertEqual(out.shape, (batch_size, self.seq_len, self.motion_dim))

    def test_blending_behavior(self):
        """
        Test that blending with different skeleton-ID maps produces different outputs.
        """
        batch_size = 1
        seq1 = torch.ones(batch_size, self.seq_len, self.motion_dim)
        seq2 = torch.zeros(batch_size, self.seq_len, self.motion_dim)
        cond_a = create_skeleton_id_map(batch_size, self.seq_len, 0.0)  # All seq1
        cond_b = create_skeleton_id_map(batch_size, self.seq_len, 1.0)  # All seq2
        out_a = self.generator(seq1, seq2, cond_a)
        out_b = self.generator(seq1, seq2, cond_b)
        # Outputs should be different for different blend ratios
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_gradients(self):
        """
        Test that gradients flow through the generator.
        """
        batch_size = 2
        seq1 = torch.randn(batch_size, self.seq_len, self.motion_dim, requires_grad=True)
        seq2 = torch.randn(batch_size, self.seq_len, self.motion_dim, requires_grad=True)
        cond = create_skeleton_id_map(batch_size, self.seq_len, 0.5)
        out = self.generator(seq1, seq2, cond)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(seq1.grad)
        self.assertIsNotNone(seq2.grad)

if __name__ == "__main__":
    unittest.main()
