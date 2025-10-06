import torch
import torch.nn as nn
# Example SNN module using Norse (if installed)
try:
    import norse.torch as snn
    class SNNLayer(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.fc = nn.Linear(input_size, hidden_size)
            self.lif = snn.LIFCell()
        def forward(self, x):
            # x: (batch, seq, features)
            s = None
            outputs = []
            for t in range(x.shape[1]):
                z = self.fc(x[:, t, :])
                o, s = self.lif(z, s)
                outputs.append(o)
            return torch.stack(outputs, dim=1)
except ImportError:
    class SNNLayer(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.fc = nn.Linear(input_size, hidden_size)
        def forward(self, x):
            # Fallback: just a linear layer
            return self.fc(x)

class SNNBlendNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.snn = SNNLayer(hidden_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, input_dim, batch_first=True)
    def forward(self, src, tgt, blend_weight):
        # src, tgt: (batch, seq, features)
        src_enc, _ = self.encoder(src)
        tgt_enc, _ = self.encoder(tgt)
        blend = blend_weight * src_enc + (1 - blend_weight) * tgt_enc
        snn_out = self.snn(blend)
        out, _ = self.decoder(snn_out)
        return out

def blend_motion_snn(src_arr, tgt_arr, blend_weight=0.5):
    # src_arr, tgt_arr: (seq, joints, 3)
    src = torch.tensor(src_arr).float().view(1, src_arr.shape[0], -1)
    tgt = torch.tensor(tgt_arr).float().view(1, tgt_arr.shape[0], -1)
    model = SNNBlendNet(input_dim=src.shape[2], hidden_dim=128)
    with torch.no_grad():
        blended = model(src, tgt, torch.tensor([[blend_weight]]))
    return blended.numpy().reshape(src_arr.shape)

# --- TEST ---
import numpy as np
import unittest

class TestSNNBlendNet(unittest.TestCase):
    def test_blend_shapes(self):
        src = np.random.randn(20, 15, 3)
        tgt = np.random.randn(20, 15, 3)
        out = blend_motion_snn(src, tgt, blend_weight=0.7)
        self.assertEqual(out.shape, src.shape)
        self.assertFalse(np.allclose(out, src))
        self.assertFalse(np.allclose(out, tgt))

if __name__ == "__main__":
    unittest.main()
