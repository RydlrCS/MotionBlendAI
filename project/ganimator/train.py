
# SPADE-GANimator Training Script
# Implements a generator with SPADE blocks and skeleton-ID conditioning for motion blending.
# For hackathon use: code is simplified and well-commented for clarity.


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
# Import extraction from motion_extractor
import sys
sys.path.append(os.path.dirname(__file__))
from motion_extractor import extract_glb_joints

# --- SPADE Block ---
class SPADE(nn.Module):
    """
    SPADE (Spatially-Adaptive Denormalization) block for 1D motion data.
    Modulates normalized activations using a conditioning tensor (e.g., skeleton-ID map).
    """
    def __init__(self, norm_nc, cond_nc):
        super().__init__()
        self.norm = nn.BatchNorm1d(norm_nc, affine=False)
        self.mlp_gamma = nn.Linear(cond_nc, norm_nc)
        self.mlp_beta = nn.Linear(cond_nc, norm_nc)
    def forward(self, x, cond):
        # x: (batch, C, T), cond: (batch, cond_nc)
        x_norm = self.norm(x)
        gamma = self.mlp_gamma(cond).unsqueeze(2)  # (batch, C, 1)
        beta = self.mlp_beta(cond).unsqueeze(2)
        return x_norm * (1 + gamma) + beta

# --- Generator with SPADE blocks ---
class SPADEGenerator(nn.Module):
    """
    Generator for motion blending using SPADE blocks and skeleton-ID conditioning.
    Inputs: two motion sequences, skeleton-ID map, and optional noise.
    Outputs: blended motion sequence.
    """
    def __init__(self, motion_dim=10, cond_dim=2, hidden_dim=64, seq_len=60):
        super().__init__()
        self.seq_len = seq_len
        self.fc_in = nn.Linear(motion_dim * 2 + cond_dim, hidden_dim)
        self.spade1 = SPADE(hidden_dim, cond_dim)
        self.spade2 = SPADE(hidden_dim, cond_dim)
        self.fc_out = nn.Linear(hidden_dim, motion_dim)
    def forward(self, seq1, seq2, cond):
        # seq1, seq2: (batch, seq_len, motion_dim)
        # cond: (batch, cond_dim) or (batch, seq_len, cond_dim)
        x = torch.cat([seq1, seq2], dim=2)  # (batch, seq_len, motion_dim*2)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).expand(-1, self.seq_len, -1)  # (batch, seq_len, cond_dim)
        x = torch.cat([x, cond], dim=2)  # (batch, seq_len, motion_dim*2+cond_dim)
        x = self.fc_in(x)  # (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        x = self.spade1(x, cond.mean(dim=1))
        x = F.relu(x)
        x = self.spade2(x, cond.mean(dim=1))
        x = F.relu(x)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        out = self.fc_out(x)  # (batch, seq_len, motion_dim)
        return out

# --- Dummy Discriminator (for adversarial loss) ---
class Discriminator(nn.Module):
    def __init__(self, motion_dim=10, hidden_dim=64, seq_len=60):
        super().__init__()
        self.fc1 = nn.Linear(motion_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, seq):
        # seq: (batch, seq_len, motion_dim)
        x = seq.reshape(seq.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# --- Skeleton-ID Conditioning Utility ---
def create_skeleton_id_map(batch_size, seq_len, blend_ratio):
    """
    Create a skeleton-ID map for blending. For two motions, this is a (batch, seq_len, 2) tensor
    where each frame is [1,0] (motion A), [0,1] (motion B), or a blend.
    """
    id_map = torch.zeros((batch_size, seq_len, 2))
    for b in range(batch_size):
        for t in range(seq_len):
            alpha = min(1.0, max(0.0, 1.0 - abs(t / seq_len - blend_ratio)))
            id_map[b, t, 0] = alpha
            id_map[b, t, 1] = 1 - alpha
    return id_map

# --- Training Loop (simplified for hackathon) ---
def train():
    # Hyperparameters
    motion_dim = 10
    seq_len = 60
    batch_size = 4
    num_epochs = 2
    lr = 1e-3


    # Load two real GLB files from build/build_motions
    build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build/build_motions'))
    file1 = os.path.join(build_dir, 'Air Kicking_mixamo.glb')
    file2 = os.path.join(build_dir, 'Air Punch_mixamo.glb')
    arr1 = extract_glb_joints(file1)
    arr2 = extract_glb_joints(file2)
    if arr1 is None or arr2 is None:
        raise RuntimeError('Failed to extract joint data from GLB files.')
    # Use only the first frame if needed, or repeat to seq_len
    # arr1, arr2: (1, joints, 3)
    # Flatten joints for motion_dim
    arr1 = arr1.reshape(-1, arr1.shape[1]*arr1.shape[2])  # (1, motion_dim)
    arr2 = arr2.reshape(-1, arr2.shape[1]*arr2.shape[2])
    # If not enough frames, repeat
    arr1 = arr1.repeat(seq_len, axis=0) if arr1.shape[0] < seq_len else arr1[:seq_len]
    arr2 = arr2.repeat(seq_len, axis=0) if arr2.shape[0] < seq_len else arr2[:seq_len]
    # Truncate or pad to match motion_dim
    min_dim = min(arr1.shape[1], arr2.shape[1], motion_dim)
    arr1 = arr1[:, :min_dim]
    arr2 = arr2[:, :min_dim]
    # Build dataset
    dataset = [(torch.tensor(arr1, dtype=torch.float32), torch.tensor(arr2, dtype=torch.float32)) for _ in range(20)]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models
    generator = SPADEGenerator(motion_dim=motion_dim, cond_dim=2, seq_len=seq_len)
    discriminator = Discriminator(motion_dim=motion_dim, seq_len=seq_len)
    g_opt = torch.optim.Adam(generator.parameters(), lr=lr)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        for i, (seq1, seq2) in enumerate(dataloader):
            # Create blend schedule (random blend point)
            blend_ratio = torch.rand(1).item()
            cond = create_skeleton_id_map(seq1.size(0), seq_len, blend_ratio)

            # --- Train Discriminator ---
            d_opt.zero_grad()
            real = seq1  # Use seq1 as 'real' for demo
            fake = generator(seq1, seq2, cond)
            d_real = discriminator(real)
            d_fake = discriminator(fake.detach())
            d_loss = -torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8))
            d_loss.backward()
            d_opt.step()

            # --- Train Generator ---
            g_opt.zero_grad()
            fake = generator(seq1, seq2, cond)
            d_fake = discriminator(fake)
            # Adversarial + reconstruction loss
            adv_loss = -torch.mean(torch.log(d_fake + 1e-8))
            rec_loss = loss_fn(fake, seq1)  # For demo, reconstruct seq1
            g_loss = adv_loss + 0.1 * rec_loss
            g_loss.backward()
            g_opt.step()

            if i % 2 == 0:
                print(f"Epoch {epoch} Iter {i}: D_loss={d_loss.item():.4f} G_loss={g_loss.item():.4f}")

    # Save model (for hackathon: local file)
    os.makedirs('models', exist_ok=True)
    torch.save(generator.state_dict(), 'models/ganimator_spade.pth')
    print("[INFO] Model saved to models/ganimator_spade.pth")

if __name__ == '__main__':
    train()
