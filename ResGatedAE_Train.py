import os
import traceback
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ============================================================
# ---------------------- CONFIG ------------------------------
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DINO params
DINO_BASE_SIZE = 518
DINO_PATCH_SIZE = 14

# Training params
BATCH_SIZE = 4
LR = 1e-4

# PNNR params
PNNR_SAMPLE_RATIO = 0.5
MAX_BANK_SIZE = 100000

# Global feature extractor cache
_feature_extractor = None

# ============================================================
# ------------------- DINO FEATURES ---------------------------
# ============================================================

def load_dinov2_model():
    global _feature_extractor
    if _feature_extractor is not None:
        return _feature_extractor
    
    print(f"Loading DINOv2 model on {DEVICE}...", flush=True)
    try:
        _feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(DEVICE)
        _feature_extractor.eval()
        print("✔ DINOv2 Loaded.", flush=True)
    except Exception as e:
        print(f"❌ Failed to load DINOv2: {e}")
        traceback.print_exc()
        return None
    return _feature_extractor

def preprocess_image(img_np, size):
    """Resizes and normalizes image for DINO"""
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    if isinstance(img_np, Image.Image):
        img_np = np.array(img_np)
        
    return t(img_np).unsqueeze(0).to(DEVICE)

def extract_dino_features(img_np):
    model = load_dinov2_model()
    if model is None:
        raise RuntimeError("DINO Model not loaded.")

    with torch.no_grad():
        size = (DINO_BASE_SIZE, DINO_BASE_SIZE)
        img_t = preprocess_image(img_np, size)
        features = model.get_intermediate_layers(img_t, n=1, reshape=True)[0]
    
    return features.squeeze(0).permute(1, 2, 0).cpu().numpy()

# ============================================================
# ------------------ MODEL BLOCKS -----------------------------
# ============================================================

class LightFFNBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x, *args, **kwargs):
        # *args and **kwargs added to safely ignore 'kv' inputs if passed
        return x + self.ffn(self.norm(x))

# ============================================================
# ------------------ RGAE MODEL -------------------------------
# ============================================================

class RGAE_Reconstructor(nn.Module):
    def __init__(self, feature_dim, num_layers=2, bottleneck_ratio=4, dropout_p=0.3):
        super().__init__()

        self.feature_dim = feature_dim
        latent_dim = feature_dim // bottleneck_ratio

        # Global Prototype
        self.prototype_token = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Aggregation
        self.aggregation = nn.ModuleList([
            LightFFNBlock(feature_dim) for _ in range(1)
        ])

        # Bottleneck
        self.pre_bottleneck = nn.Linear(feature_dim, latent_dim)
        self.bottleneck = nn.ModuleList([
            LightFFNBlock(latent_dim) for _ in range(num_layers)
        ])
        self.post_bottleneck = nn.Linear(latent_dim, feature_dim)

        # Decoder
        self.decoder = nn.ModuleList([
            LightFFNBlock(feature_dim) for _ in range(num_layers)
        ])

        self.gate_weights = nn.Parameter(torch.ones(num_layers))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        patch_tokens = x.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # 1. Aggregation 
        # Note: Since we use FFN, proto doesn't actually 'attend' to patch_tokens here.
        # It just transforms the learned prototype parameter.
        proto = self.prototype_token.expand(B, -1, -1)
        for blk in self.aggregation:
            proto = blk(proto) 

        # 2. Bottleneck
        z = self.pre_bottleneck(patch_tokens)
        z = self.dropout(z)
        for blk in self.bottleneck:
            z = blk(z)
        z = self.post_bottleneck(z)

        # 3. Decoding 
        decoded_layers = []
        for blk in self.decoder:
            decoded_layers.append(blk(z))

        # 4. Gating
        decoded_stack = torch.stack(decoded_layers, dim=1)
        gates = torch.softmax(self.gate_weights, dim=0)
        final_decode = (gates.view(1, -1, 1, 1) * decoded_stack).sum(dim=1)

        # 5. Reshape
        out = final_decode.permute(0, 2, 1).reshape(B, C, H, W)
        
        return out

# ============================================================
# ------------------ TRAINING -------------------------------
# ============================================================

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, feats):
        self.feats = [torch.from_numpy(f).permute(2,0,1).float() for f in feats]

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]

def train_rgae(feats, feature_dim, epochs, save_path):
    dataset = FeatureDataset(feats)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = RGAE_Reconstructor(feature_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"Starting RGAE training for {epochs} epochs...", flush=True)
    model.train()

    for ep in range(epochs):
        total_loss = 0
        for x in loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {ep+1}/{epochs} | Loss: {total_loss/len(loader):.6f}", flush=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✔ Model saved to {save_path}")
    return model

# ============================================================
# ------------------ PNNR BANK -------------------------------
# ============================================================

def build_pnnr_bank(feats, save_path, max_bank_size=100000, seed=42):
    rng = np.random.RandomState(seed)
    all_patches = []
    
    print("Building PNNR Bank...", flush=True)
    for f in feats:
        H, W, C = f.shape
        patches = f.reshape(-1, C)
        all_patches.append(patches)
    
    all_patches = np.vstack(all_patches)
    total_patches = all_patches.shape[0]

    if total_patches <= max_bank_size:
        bank = all_patches.astype(np.float32)
    else:
        idx = rng.choice(total_patches, size=max_bank_size, replace=False)
        bank = all_patches[idx].astype(np.float32)

    norms = np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8
    bank_normed = bank / norms

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, bank)
    np.save(save_path.replace(".npy", "_norm.npy"), bank_normed)
    
    print(f"✔ PNNR Bank saved to {save_path} (Size: {bank.shape})")
    return bank, bank_normed

def pnnr_reconstruct(feature_map, bank_raw, bank_normed):
    H, W, C = feature_map.shape
    q = torch.from_numpy(feature_map.reshape(-1, C)).to(DEVICE).float()
    q_norm = F.normalize(q, dim=1)
    
    bank_t = torch.from_numpy(bank_normed).to(DEVICE).float().t()
    
    # Cosine Similarity
    sims = torch.matmul(q_norm, bank_t)
    top1 = torch.argmax(sims, dim=1)
    
    bank_raw_t = torch.from_numpy(bank_raw).to(DEVICE).float()
    recon = bank_raw_t[top1].cpu().numpy()
    
    return recon.reshape(H, W, C)

# ============================================================
# ------------------ MAIN -----------------------------------
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to folder containing images")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default="checkpoints")
    args = parser.parse_args()

    try:
        if not os.path.exists(args.data_path):
            print(f"❌ Error: Data path '{args.data_path}' not found.")
            exit(1)

        # 1. Load Images
        print(f"Scanning {args.data_path}...", flush=True)
        image_files = sorted([
            os.path.join(args.data_path, f) 
            for f in os.listdir(args.data_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        
        if len(image_files) == 0:
            print("❌ No images found! Check your path.")
            exit(1)

        print(f"Found {len(image_files)} images. Extracting features...", flush=True)

        # 2. Extract Features
        features = []
        for p in tqdm(image_files):
            img = Image.open(p).convert("RGB")
            feat = extract_dino_features(np.array(img))
            features.append(feat)

        H, W, C = features[0].shape
        print(f"Feature Dim: {C}, Grid: {H}x{W}")

        # 3. Define Paths
        rgae_path = os.path.join(args.output_dir, "rgae_model.pth")
        bank_path = os.path.join(args.output_dir, "pnnr_bank.npy")

        # 4. Train RGAE
        rgae = train_rgae(features, C, args.epochs, rgae_path)

        # 5. Build Bank
        bank, bank_norm = build_pnnr_bank(features, bank_path)

        print("Pipeline finished successfully.")

    except Exception:
        traceback.print_exc()
