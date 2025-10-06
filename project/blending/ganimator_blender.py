def load_model_from_gcs(model_path, bucket_name):
def create_identity_map(seq1, seq2, mix_ratio):

"""
SPADE-GANimator Blender for MoCap motion blending.
Loads a trained SPADE-GANimator model from GCS or local file and provides a blend() API.
Implements SPADE blocks and skeleton-ID conditioning for hackathon use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

# --- SPADE Block (same as in train/infer) ---
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

# --- Generator (same as in train/infer) ---
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
def create_identity_map(batch_size, seq_len, blend_ratio):
    id_map = torch.zeros((batch_size, seq_len, 2))
    for b in range(batch_size):
        for t in range(seq_len):
            alpha = min(1.0, max(0.0, 1.0 - abs(t / seq_len - blend_ratio)))
            id_map[b, t, 0] = alpha
            id_map[b, t, 1] = 1 - alpha
    return id_map

# --- Model Loader ---
def load_model(model_path, bucket_name=None):
    """
    Load model weights from local file or GCS.
    """
    if bucket_name and GCS_AVAILABLE:
        # Download from GCS
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(model_path)
        local_path = '/tmp/ganimator_model.pth'
        blob.download_to_filename(local_path)
        model_file = local_path
    else:
        model_file = model_path
    model = SPADEGenerator()
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        print(f"[INFO] Loaded model from {model_file}")
    else:
        print(f"[WARN] Model file {model_file} not found. Using random weights.")
    model.eval()
    return model

# --- Blender API ---
class GANimatorBlender:
    def __init__(self, model_path='models/ganimator_spade.pth', bucket_name=None):
        self.model = load_model(model_path, bucket_name)
        self.seq_len = 60
        self.motion_dim = 10

    def blend(self, seq1, seq2, mix_ratio):
        """
        Blend two motion sequences using SPADE-GANimator.
        Args:
            seq1, seq2: torch.Tensor, shape (batch, seq_len, motion_dim)
            mix_ratio: float, blend factor (0-1)
        Returns:
            torch.Tensor: blended motion sequence
        """
        batch = seq1.size(0)
        cond = create_identity_map(batch, self.seq_len, mix_ratio)
        with torch.no_grad():
            blended = self.model(seq1, seq2, cond)
        return blended

# --- Example usage ---
if __name__ == '__main__':
    blender = GANimatorBlender()
    seq1 = torch.randn(1, 60, 10)
    seq2 = torch.randn(1, 60, 10)
    mix_ratio = 0.5
    blended = blender.blend(seq1, seq2, mix_ratio)
    print('Blended motion shape:', blended.shape)
    print('Blended motion (first frame):', blended[0, 0])
