import os
import traceback
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# --- (Settings and Helper Functions remain the same) ---
DATA_ROOT = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
# CATEGORY = "wallplugs"
# # ... (all other settings and helper functions are identical) ...
# VALIDATE_PATH_CANDIDATES = [
#     os.path.join(DATA_ROOT, CATEGORY, CATEGORY, "train", "good"),
#     os.path.join(DATA_ROOT, CATEGORY, "train", "good"),
#     os.path.join(DATA_ROOT, CATEGORY, CATEGORY, "train", "good"),
#     os.path.join(DATA_ROOT, CATEGORY, "train", "good"),
# ]

NUM_IMAGES = 1500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DINO feature extraction params
DINO_BASE_SIZE = 518
DINO_PATCH_SIZE = 14

# INP training params
EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-4

# CPR bank
MAX_BANK_SIZE = 100000

# Save paths (use script folder)
SCRIPT_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "Ad2KWallpGatedInp_train.pth")
BANK_SAVE_PATH = os.path.join(SCRIPT_DIR, "models", "Ad2KWallpGatedcpr_bank.npy")

_feature_extractor = None

def load_dinov2_model():
    global _feature_extractor
    if _feature_extractor is not None:
        return _feature_extractor
    print(f"Loading DINOv2 model (torch.hub) on device {DEVICE}", flush=True)
    try:
        _feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(DEVICE)
        _feature_extractor.eval()
        patch_sz = getattr(_feature_extractor.patch_embed, "patch_size", None)
        if isinstance(patch_sz, (tuple, list)):
            patch_sz = patch_sz[0]
        print(f"Loaded DINOv2. patch_size={patch_sz}", flush=True)
    except Exception as e:
        print("Failed to load DINOv2 model (network or hub issue). Falling back to dummy features.", flush=True)
        print("Error:", e, flush=True)
        _feature_extractor = None
    return _feature_extractor

def preprocess_image(img_np: np.ndarray, size: tuple):
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return t(img_np).unsqueeze(0).to(DEVICE)

def extract_spatial_features(img_np: np.ndarray, strategy="multi-scale"):
    """Return numpy array (grid_h, grid_w, C). If DINO not available, return random features."""
    model = load_dinov2_model()
    if model is None:
        # Dummy features: choose grid ~ DINO expected base/patch
        grid = DINO_BASE_SIZE // DINO_PATCH_SIZE
        feat_dim = 768 * (3 if strategy == "multi-scale" else 1)
        return np.random.randn(grid, grid, feat_dim).astype(np.float32)

    patch_size = model.patch_embed.patch_size[0]
    with torch.no_grad():
        if strategy == "single-scale":
            size = (DINO_BASE_SIZE, DINO_BASE_SIZE)
            img_t = preprocess_image(img_np, size)
            features = model.get_intermediate_layers(img_t, n=1, reshape=True)[0]
        else:
            scales = [1.0, 0.75, 0.5]
            base = DINO_BASE_SIZE
            feats = []
            for s in scales:
                h = int(base * s) // patch_size * patch_size
                w = int(base * s) // patch_size * patch_size
                if h == 0 or w == 0:
                    continue
                img_t = preprocess_image(img_np, (h, w))
                f = model.get_intermediate_layers(img_t, n=1, reshape=True)[0]
                f = torch.nn.functional.interpolate(
                    f, size=(base // patch_size, base // patch_size), mode='bilinear', align_corners=False
                )
                feats.append(f)
            features = torch.cat(feats, dim=1)
    features = features.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return features


class MockTransformerBlock(nn.Module):
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


class INP_Former(nn.Module):
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


class FeatureMapDataset(torch.utils.data.Dataset):
    def __init__(self, features_list):
        self.features = [torch.from_numpy(f).permute(2,0,1).float() for f in features_list]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

# --- MODIFIED TRAINING FUNCTION ---
def train_inp_with_features(feature_maps_np, grid_h, grid_w, feature_dim,
                            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=DEVICE):
    dataset = FeatureMapDataset(feature_maps_np)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = INP_Former(feature_dim, grid_h, grid_w, num_layers=2).to(device)
    
    # --- 🔹 Set up differential learning rates ---
    # Give the gate a much higher learning rate to encourage it to learn
    gate_lr = 1e-2 
    
    # Separate the parameters into two groups
    gate_params = [p for name, p in model.named_parameters() if 'gate_weights' in name]
    base_params = [p for name, p in model.named_parameters() if 'gate_weights' not in name]
    
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': lr},
        {'params': gate_params, 'lr': gate_lr}
    ], lr=lr)
    # ---------------------------------------------
    
    criterion = nn.MSELoss()
    print("Starting INP training with differential learning rate...", flush=True)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
        epoch_loss = running_loss / len(dataset)
        
        # Also print the learned gate weights to see if they are changing
        with torch.no_grad():
            learned_gates = torch.softmax(model.gate_weights, dim=0).cpu().numpy()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f} - Gates: {np.round(learned_gates, 3)}", flush=True)

    print("INP training finished.", flush=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("Saved INP model to", MODEL_PATH, flush=True)
    return model

def build_cpr_bank(feature_maps_np, max_bank_size=MAX_BANK_SIZE, seed=42):
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
    return bank, bank_normed


def cpr_reconstruct_from_bank(feature_map_np, bank_normed, bank_raw):
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


def gather_training_images(folder_path, num_images=NUM_IMAGES):
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    if not os.path.exists(folder_path):
        raise RuntimeError(f"No folder found at {folder_path}")
    all_images = [os.path.join(folder_path, f)
                  for f in os.listdir(folder_path)
                  if os.path.splitext(f)[1].lower() in image_extensions]
    if len(all_images) == 0:
        raise RuntimeError(f"No image files found in {folder_path}")
    selected = sorted(all_images)[:min(num_images, len(all_images))]
    print(f"Selected {len(selected)} images for training from {folder_path}", flush=True)
    return selected


if __name__ == "__main__":
    try:
        VALIDATE_PATH = None
        for p in VALIDATE_PATH_CANDIDATES:
            if os.path.exists(p) and os.path.isdir(p):
                VALIDATE_PATH = p
                break
        if VALIDATE_PATH is None:
            matches = []
            for root, dirs, files in os.walk(os.path.join(DATA_ROOT, CATEGORY)):
                for d in dirs:
                    if "valid" in d.lower():
                        matches.append(os.path.join(root, d))
            if matches:
                VALIDATE_PATH = matches[0]
        if VALIDATE_PATH is None:
            VALIDATE_PATH = VALIDATE_PATH_CANDIDATES[0]

        print("Using validation folder:", VALIDATE_PATH, flush=True)

        img_paths = gather_training_images(VALIDATE_PATH, NUM_IMAGES)
        print("DEBUG: img_paths collected =", img_paths, flush=True)

        print("Extracting DINOv2 features for training images (this may take time).", flush=True)
        feats = []
        for p in tqdm(img_paths, desc="DINO extract"):
            img_np = np.array(Image.open(p).convert("RGB"))
            feat = extract_spatial_features(img_np, strategy="single-scale")
            feats.append(feat.astype(np.float32))

        grid_h, grid_w, feature_dim = feats[0].shape
        print("Feature grid:", grid_h, grid_w, "feature_dim:", feature_dim, flush=True)

        inp_model = train_inp_with_features(feats, grid_h, grid_w, feature_dim,
                                            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=DEVICE)

        print("Building CPR bank (subsampling patches)...", flush=True)
        bank_raw, bank_normed = build_cpr_bank(feats, max_bank_size=MAX_BANK_SIZE)

        test_idx = 0
        test_feat = feats[test_idx]
        inp_model.eval()
        with torch.no_grad():
            t = torch.from_numpy(test_feat).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            _, recon_t = inp_model(t)
            recon_np = recon_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

        bank_norm_path = BANK_SAVE_PATH.replace(".npy", "_norm.npy")
        bank_norm_loaded = np.load(bank_norm_path)
        bank_raw_loaded = np.load(BANK_SAVE_PATH)
        recon_from_bank = cpr_reconstruct_from_bank(test_feat, bank_norm_loaded, bank_raw_loaded)

        amap = np.mean((recon_np - recon_from_bank) ** 2, axis=2)
        orig_img = np.array(Image.open(img_paths[test_idx]).convert("RGB"))
        H_orig, W_orig = orig_img.shape[:2]
        amap_up = F.interpolate(torch.from_numpy(amap).unsqueeze(0).unsqueeze(0),
                               size=(H_orig, W_orig), mode='bilinear', align_corners=False).squeeze().numpy()
        amap_norm = (amap_up - amap_up.min()) / (np.ptp(amap_up) + 1e-8)
        amap_img = (255 * amap_norm).astype(np.uint8)

        out_path = os.path.join(SCRIPT_DIR, "models", "anomaly_test_inp_cpr.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        import cv2
        cv2.imwrite(out_path, amap_img)
        print("Saved test anomaly map to:", out_path, flush=True)

        print("All done. Trained INP saved to:", MODEL_PATH, "CPR bank saved to:", BANK_SAVE_PATH, flush=True)

    except Exception as e:
        print("ERROR during execution:", str(e), flush=True)
        traceback.print_exc()
        raise
