import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ============================================================
# 1. CONFIGURATION & ARGS
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINO_BASE_SIZE = 518
DINO_PATCH_SIZE = 14

# Global cache
_feature_extractor = None

# ============================================================
# 2. MODEL DEFINITIONS
# ============================================================

class LightFFNBlock(nn.Module):
    """
    Lightweight FFN block used in the RGAE architecture.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, *args, **kwargs):
        return x + self.ffn(self.norm(x))

class RGAE(nn.Module):
    """
    The RGAE (Residual Gated Autoencoder) Model Architecture.
    Matches the structure expected by your trained weights.
    """
    def __init__(self, feature_dim, grid_h, grid_w, num_layers=2):
        super().__init__()
        self.bottleneck = nn.ModuleList([LightFFNBlock(feature_dim) for _ in range(num_layers)])
        self.aggregation = nn.ModuleList([LightFFNBlock(feature_dim)])
        self.decoder = nn.ModuleList([LightFFNBlock(feature_dim) for _ in range(num_layers)])
        
        self.prototype_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.gate_weights = nn.Parameter(torch.ones(num_layers))
        
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.feature_dim = feature_dim

    def forward(self, x):
        B, C, H, W = x.shape
        patch_tokens = x.flatten(2).permute(0, 2, 1) # (B, HW, C)
        
        # Aggregation
        agg_proto = self.prototype_token.repeat(B, 1, 1)
        for blk in self.aggregation:
            agg_proto = blk(agg_proto, patch_tokens)
        
        # Bottleneck
        z = patch_tokens
        for blk in self.bottleneck:
            z = blk(z)
        
        # Decoding with Gating
        de_list = [blk(z, agg_proto) for blk in self.decoder]
        de_stack = torch.stack(de_list, dim=1)
        gates = torch.softmax(self.gate_weights, dim=0)
        de_fused = (gates.view(1, -1, 1, 1) * de_stack).sum(dim=1)
        
        # Reshape back to feature map
        de_map = de_fused.permute(0, 2, 1).reshape(B, C, H, W)
        return None, de_map

# ============================================================
# 3. UTILITIES
# ============================================================

def load_dinov2_model():
    global _feature_extractor
    if _feature_extractor is not None:
        return _feature_extractor
    
    print(f"Loading DINOv2 model on {DEVICE}...")
    try:
        _feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False).to(DEVICE)
        _feature_extractor.eval()
        print("✔ DINOv2 loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load DINOv2: {e}")
        return None
    return _feature_extractor

def extract_spatial_features(img_np):
    model = load_dinov2_model()
    if model is None:
        # Fallback dummy features
        grid = DINO_BASE_SIZE // DINO_PATCH_SIZE
        return np.random.randn(grid, grid, 768).astype(np.float32)

    # Preprocess
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((DINO_BASE_SIZE, DINO_BASE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_t = t(img_np).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        features = model.get_intermediate_layers(img_t, n=1, reshape=True)[0]
    
    # Return (H, W, C) numpy array
    return features.squeeze(0).permute(1, 2, 0).cpu().numpy()

def cpr_reconstruct_from_bank(feature_map_np, bank_normed, bank_raw):
    """
    Reconstructs features using the Nearest Neighbor from the Memory Bank.
    """
    H, W, C = feature_map_np.shape
    
    # Query vectors
    q = torch.from_numpy(feature_map_np.reshape(-1, C)).to(DEVICE).float()
    q_norm = F.normalize(q, dim=1)
    
    # Bank vectors
    bank_t = torch.from_numpy(bank_normed).to(DEVICE).float().t()
    
    # Similarity & Retrieval
    sims = torch.matmul(q_norm, bank_t)
    top1 = torch.argmax(sims, dim=1)
    
    bank_raw_t = torch.from_numpy(bank_raw).to(DEVICE).float()
    recon = bank_raw_t[top1].cpu().numpy()
    
    return recon.reshape(H, W, C)

def load_inp_model(model_path, feature_dim, grid_h, grid_w):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = RGAE(feature_dim, grid_h, grid_w, num_layers=2).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

# ============================================================
# 4. MAIN EXECUTION
# ============================================================

def generate_heatmaps(args):
    # 1. Setup & Paths
    
    # Heuristic: try specific name first, then generic
    model_path = os.path.join(args.checkpoint_dir, f"SingleCandGatedInp_train.pth") 
    if not os.path.exists(model_path):
         model_path = os.path.join(args.checkpoint_dir, f"rgae_{args.category}.pth")

    bank_path = os.path.join(args.checkpoint_dir, f"SingleCandGatedcpr_bank.npy")
    if not os.path.exists(bank_path):
        bank_path = os.path.join(args.checkpoint_dir, f"pnnr_bank_{args.category}.npy")

    if not os.path.exists(args.data_path):
        print(f"❌ Error: Image directory not found: {args.data_path}")
        return
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return
    if not os.path.exists(bank_path):
        print(f"❌ Error: Bank file not found: {bank_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load Memory Bank
    print(f"Loading Bank from {bank_path}...")
    bank_raw = np.load(bank_path)
    bank_norm_path = bank_path.replace(".npy", "_norm.npy")
    
    if os.path.exists(bank_norm_path):
        bank_normed = np.load(bank_norm_path)
    else:
        norms = np.linalg.norm(bank_raw, axis=1, keepdims=True) + 1e-8
        bank_normed = bank_raw / norms
    print("✔ Bank Loaded.")

    # 3. Load DINO
    load_dinov2_model()

    # 4. Gather Images
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    test_images = sorted([
        os.path.join(args.data_path, f) for f in os.listdir(args.data_path)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    
    if not test_images:
        print("❌ No images found in directory.")
        return
    
    print(f"Found {len(test_images)} images. Processing...")

    # 5. Initialize Model (Using first image for dimensions)
    img0 = np.array(Image.open(test_images[0]).convert("RGB"))
    feat0 = extract_spatial_features(img0)
    H_feat, W_feat, C_feat = feat0.shape
    print(f"Feature Grid: {H_feat}x{W_feat}, Dim: {C_feat}")

    inp_model = load_inp_model(model_path, C_feat, H_feat, W_feat)
    print("✔ Model Loaded.")

    # 6. Inference Loop
    for i, img_path in enumerate(tqdm(test_images)):
        img_name = os.path.basename(img_path)
        img_np = np.array(Image.open(img_path).convert("RGB"))

        # Feature Extraction
        feat_np = extract_spatial_features(img_np)

        # RGAE/INP Reconstruction
        t = torch.from_numpy(feat_np).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            _, recon_inp = inp_model(t)
        recon_inp_np = recon_inp.squeeze(0).permute(1,2,0).cpu().numpy()

        # CPR/Bank Reconstruction
        recon_bank_np = cpr_reconstruct_from_bank(feat_np, bank_normed, bank_raw)

        # Anomaly Map (MSE)
        amap = np.mean((recon_inp_np - recon_bank_np)**2, axis=2)

        # Upscale to original image size
        H_orig, W_orig = img_np.shape[:2]
        amap_up = F.interpolate(
            torch.from_numpy(amap).unsqueeze(0).unsqueeze(0),
            size=(H_orig, W_orig),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        # Normalize & Colorize
        amap_norm = (amap_up - amap_up.min()) / (np.ptp(amap_up) + 1e-8)
        amap_img = (255 * amap_norm).astype(np.uint8)
        heatmap = cv2.applyColorMap(amap_img, cv2.COLORMAP_JET)
        
        # Overlay
        # Convert RGB(PIL) -> BGR(OpenCV) for correct color overlay
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

        # Save
        out_path = os.path.join(args.output_dir, f"heatmap_{img_name}")
        cv2.imwrite(out_path, overlay)

    print(f"✔ Done! Heatmaps saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to folder containing test images")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Directory containing .pth and .npy files")
    parser.add_argument('--output_dir', type=str, default="heatmaps", help="Where to save output images")
    parser.add_argument('--category', type=str, default="candle", help="Category name (used for fallback filename matching)")
    
    args = parser.parse_args()
    
    generate_heatmaps(args)
