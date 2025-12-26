import os
import traceback
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ============================================================
# ---------------------- CONFIG ------------------------------
# ============================================================

DATA_ROOT = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()

NUM_IMAGES = 1500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DINO params
DINO_BASE_SIZE = 518
DINO_PATCH_SIZE = 14

# RGAE training
EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-4

# PNNR
PNNR_SAMPLE_RATIO = 0.5
MAX_BANK_SIZE = 100000

SCRIPT_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "RGAE_AD2.pth")
PNNR_BANK_PATH = os.path.join(SCRIPT_DIR, "models", "pnnr_bank.npy")

_feature_extractor = None

# ============================================================
# ------------------- DINO FEATURES ---------------------------
# ============================================================

def load_dino():
    global _feature_extractor
    if _feature_extractor is not None:
        return _feature_extractor

    _feature_extractor = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14"
    ).to(DEVICE).eval()

    return _feature_extractor


def preprocess(img_np, size):
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return t(img_np).unsqueeze(0).to(DEVICE)


def extract_dino_features(img_np):
    model = load_dino()

    with torch.no_grad():
        img_t = preprocess(img_np, (DINO_BASE_SIZE, DINO_BASE_SIZE))
        feat = model.get_intermediate_layers(img_t, n=1, reshape=True)[0]

    return feat.squeeze(0).permute(1, 2, 0).cpu().numpy()

# ============================================================
# ------------------ RGAE MODEL -------------------------------
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
        return x + self.ffn(self.norm(x))


class RGAE_Reconstructor(nn.Module):
def __init__(
        self,
        feature_dim,
        grid_h,
        grid_w,
        num_layers=2,
        bottleneck_ratio=4,
        dropout_p=0.3
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.grid_h, self.grid_w = grid_h, grid_w

        latent_dim = feature_dim // bottleneck_ratio

        # ---------- Aggregation (global normal context) ----------
        self.aggregation = nn.ModuleList([
            MockTransformerBlock(feature_dim)
        ])

        self.prototype_token = nn.Parameter(
            torch.randn(1, 1, feature_dim)
        )

        # ---------- Bottleneck (capacity-limited) ----------
        self.pre_bottleneck = nn.Linear(feature_dim, latent_dim)

        self.bottleneck = nn.ModuleList([
            MockTransformerBlock(latent_dim)
            for _ in range(num_layers)
        ])

        self.post_bottleneck = nn.Linear(latent_dim, feature_dim)

        # ---------- Decoder ----------
        self.decoder = nn.ModuleList([
            MockTransformerBlock(feature_dim)
            for _ in range(num_layers)
        ])

        # ---------- Gating ----------
        self.gate_weights = nn.Parameter(torch.ones(num_layers))

        # ---------- Stochasticity ----------
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # ---- patch tokens ----
        patch_tokens = x.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # ---- global prototype aggregation ----
        proto = self.prototype_token.expand(B, -1, -1)
        for blk in self.aggregation:
            proto = blk(proto, patch_tokens)

        # ---- capacity bottleneck ----
        z = self.pre_bottleneck(patch_tokens)     # (B, HW, C//r)
        z = self.dropout(z)                        # IMPORTANT

        for blk in self.bottleneck:
            z = blk(z)

        z = self.post_bottleneck(z)               # back to C

        # ---- decoding conditioned on proto ----
        decoded = []
        for blk in self.decoder:
            decoded.append(blk(z, proto))

        decoded = torch.stack(decoded, dim=1)     # (B, L, HW, C)

        gates = torch.softmax(self.gate_weights, dim=0)
        decoded = (gates.view(1, -1, 1, 1) * decoded).sum(dim=1)

        # ---- reshape back to feature map ----
        out = decoded.permute(0, 2, 1).reshape(B, C, H, W)

        return None, out

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


def train_rgae_with_features(feats, feature_dim):
    dataset = FeatureDataset(feats)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    model = RGAE_Reconstructor(feature_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("Training RGAE...", flush=True)

    for ep in range(EPOCHS):
        total = 0
        for x in loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {ep+1}/{EPOCHS} | Loss: {total/len(loader):.6f}", flush=True)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    return model

# ============================================================
# ------------------ PNNR BANK -------------------------------
# ============================================================

def build_pnnr_bank(feats):
    rng = np.random.RandomState(seed)
    all_patches = []
    for f in feature_maps_np:
        H, W, C = f.shape
        patches = f.reshape(-1, C)
        all_patches.append(patches)
    all_patches = np.vstack(all_patches)
    total = all_patches.shape[0]
    if total <= max_bank_size:
        bank = all_patches.astype(np.float32)
    else:
        idx = rng.choice(total, size=max_bank_size, replace=False)
        bank = all_patches[idx].astype(np.float32)
    norms = np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8
    bank_normed = bank / norms
    os.makedirs(os.path.dirname(BANK_SAVE_PATH), exist_ok=True)
    np.save(BANK_SAVE_PATH, bank)
    np.save(BANK_SAVE_PATH.replace(".npy", "_norm.npy"), bank_normed)
    print(f"Saved CPR bank: {BANK_SAVE_PATH}  (size={bank.shape[0]} x {bank.shape[1]})", flush=True)
    return bank, bank_normed bank, bank_norm


def pnnr_reconstruct(feature_map, bank, bank_norm):
    device = DEVICE
    q = torch.from_numpy(feature_map_np.reshape(-1, feature_map_np.shape[2])).to(device).float()
    q_norm = F.normalize(q, dim=1)
    bank_t = torch.from_numpy(bank_normed).to(device).float().t()
    sims = torch.matmul(q_norm, bank_t)
    top1 = torch.argmax(sims, dim=1)
    bank_raw_t = torch.from_numpy(bank_raw).to(device).float()
    recon = bank_raw_t[top1].cpu().numpy()
    H, W, C = feature_map_np.shape
    recon_map = recon.reshape(H, W, C)
    return recon_map

# ============================================================
# ------------------ MAIN -----------------------------------
# ============================================================

if __name__ == "__main__":
    try:
        image_files = sorted(os.listdir(DATA_ROOT))[:NUM_IMAGES]

        features = []
        for p in tqdm(image_files):
            img = np.array(Image.open(p).convert("RGB"))
            features.append(extract_dino_features(img))

        H, W, C = features[0].shape

        rgae = train_rgae_with_features(features, C)
        bank, bank_norm = build_pnnr_bank(features)

        test_feat = features[0]

        with torch.no_grad():
            t = torch.from_numpy(test_feat).permute(2,0,1).unsqueeze(0).to(DEVICE)
            recon_rgae = rgae(t).squeeze().permute(1,2,0).cpu().numpy()

        recon_pnnr = pnnr_reconstruct(test_feat, bank, bank_norm)
        anomaly_map = np.mean((recon_rgae - recon_pnnr) ** 2, axis=2)

        print("RGAE + PNNR pipeline finished successfully ✔")

    except Exception:
        traceback.print_exc()
