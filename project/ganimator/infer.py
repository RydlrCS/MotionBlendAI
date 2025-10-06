generator = Generator()
generator.eval()

# SPADE-GANimator Inference Script
# Loads the trained model and performs motion blending with skeleton-ID conditioning.

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# --- SPADE Block (same as in train.py) ---
class SPADE(nn.Module):
    def __init__(self, norm_nc, cond_nc):
        super().__init__()
        self.norm = nn.BatchNorm1d(norm_nc, affine=False)
        self.mlp_gamma = nn.Linear(cond_nc, norm_nc)
        self.mlp_beta = nn.Linear(cond_nc, norm_nc)
    def forward(self, x, cond):
        x_norm = self.norm(x)
        gamma = self.mlp_gamma(cond).unsqueeze(2)
        beta = self.mlp_beta(cond).unsqueeze(2)
        return x_norm * (1 + gamma) + beta

# --- Generator (same as in train.py) ---
class SPADEGenerator(nn.Module):
    def __init__(self, motion_dim=10, cond_dim=2, hidden_dim=64, seq_len=60):
        super().__init__()
        self.seq_len = seq_len
        self.fc_in = nn.Linear(motion_dim * 2 + cond_dim, hidden_dim)
        self.spade1 = SPADE(hidden_dim, cond_dim)
        self.spade2 = SPADE(hidden_dim, cond_dim)
        self.fc_out = nn.Linear(hidden_dim, motion_dim)
    def forward(self, seq1, seq2, cond):
        x = torch.cat([seq1, seq2], dim=2)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).expand(-1, self.seq_len, -1)
        x = torch.cat([x, cond], dim=2)
        x = self.fc_in(x)
        x = x.transpose(1, 2)
        x = self.spade1(x, cond.mean(dim=1))
        x = F.relu(x)
        x = self.spade2(x, cond.mean(dim=1))
        x = F.relu(x)
        x = x.transpose(1, 2)
        out = self.fc_out(x)
        return out

# --- Skeleton-ID Conditioning Utility ---
def create_skeleton_id_map(batch_size, seq_len, blend_ratio):
    id_map = torch.zeros((batch_size, seq_len, 2))
    for b in range(batch_size):
        for t in range(seq_len):
            alpha = min(1.0, max(0.0, 1.0 - abs(t / seq_len - blend_ratio)))
            id_map[b, t, 0] = alpha
            id_map[b, t, 1] = 1 - alpha
    return id_map

if __name__ == '__main__':
    # Parameters (should match training)
    motion_dim = 10
    seq_len = 60
    cond_dim = 2

    # Load trained model
    model_path = 'models/ganimator_spade.pth'
    generator = SPADEGenerator(motion_dim=motion_dim, cond_dim=cond_dim, seq_len=seq_len)
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"[INFO] Loaded model from {model_path}")
    else:
        print(f"[WARN] Model file {model_path} not found. Using random weights.")
    generator.eval()

    # Example: Blend two random motions with a blend ratio
    seq1 = torch.randn(1, seq_len, motion_dim)
    seq2 = torch.randn(1, seq_len, motion_dim)
    blend_ratio = 0.5  # 0 = all seq1, 1 = all seq2, 0.5 = blend
    cond = create_skeleton_id_map(1, seq_len, blend_ratio)

    with torch.no_grad():
        blended = generator(seq1, seq2, cond)
    print('Blended motion shape:', blended.shape)
    print('Blended motion (first frame):', blended[0, 0])
