import os
import argparse
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# ============================================================
# 1. CONFIGURATION & ARGS
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINO_BASE_SIZE = 518
PIXEL_SUBSAMPLE_PERCENT = 0.5  # Speeds up pixel-AUC calculation

# Global cache for DINO model
_feature_extractor = None

# ============================================================
# 2. MODEL DEFINITIONS
# ============================================================

class LightFFNBlock(nn.Module):
    """
    Lightweight Feed-Forward Network Block.
    (Renamed from MockTransformerBlock for clarity since it uses FFN)
    """
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
        # *args allows compatibility if 'kv' is passed by mistake
        return x + self.ffn(self.norm(x))

class RGAE(nn.Module):
    def __init__(self, feature_dim, num_layers=2, bottleneck_ratio=4, dropout_p=0.3):
        super().__init__()
        
        latent_dim = feature_dim // bottleneck_ratio

        # Global Prototype
        self.prototype_token = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Layers
        self.aggregation = nn.ModuleList([LightFFNBlock(feature_dim) for _ in range(1)])
        self.pre_bottleneck = nn.Linear(feature_dim, latent_dim)
        self.bottleneck = nn.ModuleList([LightFFNBlock(latent_dim) for _ in range(num_layers)])
        self.post_bottleneck = nn.Linear(latent_dim, feature_dim)
        self.decoder = nn.ModuleList([LightFFNBlock(feature_dim) for _ in range(num_layers)])
        
        # Gating
        self.gate_weights = nn.Parameter(torch.ones(num_layers))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        patch_tokens = x.flatten(2).permute(0, 2, 1) # (B, HW, C)

        # 1. Aggregation
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

        # 4. Gating & Fusion
        decoded_stack = torch.stack(decoded_layers, dim=1)
        gates = torch.softmax(self.gate_weights, dim=0)
        final_decode = (gates.view(1, -1, 1, 1) * decoded_stack).sum(dim=1)

        return None, final_decode.permute(0, 2, 1).reshape(B, C, H, W)

class AnomalyAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, anomaly_map):
        flat = anomaly_map.flatten()
        mean, var, maxv = flat.mean(), flat.var(), flat.max()
        
        # Robust top-k
        k = max(1, int(0.01 * flat.numel()))
        topk = torch.topk(flat, k).values.mean()

        stats = torch.stack([mean, var, maxv, topk]).unsqueeze(0)
        weight = self.mlp(stats).squeeze()
        
        return weight * topk + (1 - weight) * mean

# ============================================================
# 3. UTILITIES: DINO & PNNR
# ============================================================

def load_dinov2_model():
    global _feature_extractor
    if _feature_extractor is not None:
        return _feature_extractor

    print(f"Loading DINOv2 on {DEVICE}...", flush=True)
    try:
        _feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False).to(DEVICE)
        _feature_extractor.eval()
        print("✔ DINOv2 loaded.")
    except Exception as e:
        print(f"❌ Failed to load DINOv2: {e}")
        return None
    return _feature_extractor

def extract_spatial_features(img_np):
    model = load_dinov2_model()
    
    # Fallback if model fails (for testing flow without internet)
    if model is None:
        return np.random.randn(37, 37, 768).astype(np.float32)

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
    
    return features.squeeze(0).permute(1, 2, 0).cpu().numpy()

def pnnr_reconstruct_from_bank(feature_map_np, bank_normed, bank_raw):
    H, W, C = feature_map_np.shape
    q = torch.from_numpy(feature_map_np.reshape(-1, C)).to(DEVICE).float()
    q_norm = F.normalize(q, dim=1)
    
    bank_t = torch.from_numpy(bank_normed).to(DEVICE).float().t()
    
    # Cosine Similarity
    sims = torch.matmul(q_norm, bank_t)
    top1 = torch.argmax(sims, dim=1)
    
    bank_raw_t = torch.from_numpy(bank_raw).to(DEVICE).float()
    recon = bank_raw_t[top1].cpu().numpy()
    
    return recon.reshape(H, W, C)

# ============================================================
# 4. MAIN EVALUATION LOOP
# ============================================================

def run_evaluation(args):
    print(f"\n{'='*20} Evaluating category: {args.category} {'='*20}")
    
    # 1. Setup Paths
    model_path = os.path.join(args.checkpoint_dir, f"rgae_{args.category}.pth")
    bank_path = os.path.join(args.checkpoint_dir, f"pnnr_bank_{args.category}.npy")
    
    test_good_dir = os.path.join(args.data_root, args.category, "test", "good")
    test_bad_dir = os.path.join(args.data_root, args.category, "test", "bad")
    mask_dir = os.path.join(args.data_root, args.category, "ground_truth", "bad")

    if not os.path.exists(model_path) or not os.path.exists(bank_path):
        print(f"❌ Error: Model files not found in {args.checkpoint_dir}")
        print(f"   Expected: {model_path} and {bank_path}")
        return

    # 2. Init Models
    # Extract one dummy feature to get dimensions
    dummy_feat = extract_spatial_features(np.zeros((224, 224, 3), dtype=np.uint8))
    H, W, C = dummy_feat.shape

    rgae = RGAE(feature_dim=C).to(DEVICE)
    rgae.load_state_dict(torch.load(model_path, map_location=DEVICE))
    rgae.eval()
    
    aggregator = AnomalyAggregator().to(DEVICE).eval()
    # Note: If you trained the aggregator, load its weights here:
    # aggregator.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "aggregator.pth")))

    # 3. Load Bank
    pnnr_bank_raw = np.load(bank_path)
    pnnr_bank_normed = pnnr_bank_raw / (np.linalg.norm(pnnr_bank_raw, axis=1, keepdims=True) + 1e-8)

    # 4. Inference Loop
    image_scores = []
    image_labels = []
    all_pixel_preds = []
    all_pixel_labels = []

    # Gather images
    images = []
    for label, folder in [(0, test_good_dir), (1, test_bad_dir)]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append((os.path.join(folder, f), label))

    if not images:
        print("❌ No images found for evaluation.")
        return

    print(f"Testing {len(images)} images...")
    
    for img_path, label in tqdm(images, desc="Evaluating"):
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        feat_np = extract_spatial_features(img_np)
        
        # RGAE Recon
        with torch.no_grad():
            t = torch.from_numpy(feat_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE).float()
            _, recon_rgae = rgae(t)
            recon_rgae_np = recon_rgae.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
        # PNNR Recon
        recon_pnnr = pnnr_reconstruct_from_bank(feat_np, pnnr_bank_normed, pnnr_bank_raw)
        
        # Anomaly Map Calculation
        anomaly_map = np.mean((recon_rgae_np - recon_pnnr) ** 2, axis=2)
        
        # Image Score
        with torch.no_grad():
            score = aggregator(torch.from_numpy(anomaly_map).to(DEVICE)).item()
        image_scores.append(score)
        image_labels.append(label)

        # Pixel Metrics (Upscaling)
        H_orig, W_orig = img_np.shape[:2]
        amap_up = F.interpolate(
            torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0),
            size=(H_orig, W_orig), mode="bilinear", align_corners=False
        ).squeeze().numpy()
        
        all_pixel_preds.append(amap_up.flatten())

        # Ground Truth Mask
        if label == 1: # Bad
            mask_name = os.path.basename(img_path).split('.')[0]
            # Try png first, then same extension as image
            mask_path = os.path.join(mask_dir, mask_name + "_mask.png")
            if not os.path.exists(mask_path):
                 mask_path = os.path.join(mask_dir, mask_name + ".png")
            
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert("L"))
                all_pixel_labels.append((mask > 0).astype(np.uint8).flatten())
            else:
                all_pixel_labels.append(np.zeros(H_orig * W_orig, dtype=np.uint8))
        else:
            all_pixel_labels.append(np.zeros(H_orig * W_orig, dtype=np.uint8))

    # 5. Metrics Calculation
    try:
        image_auc = roc_auc_score(image_labels, image_scores)
    except Exception:
        image_auc = 0.5 # Handle single-class case

    # Subsample pixels for faster AUC
    all_pixel_preds = np.concatenate(all_pixel_preds)
    all_pixel_labels = np.concatenate(all_pixel_labels)
    
    if len(all_pixel_labels) > 100000:
        idx = np.random.RandomState(42).choice(
            len(all_pixel_labels), 
            int(len(all_pixel_labels) * PIXEL_SUBSAMPLE_PERCENT), 
            replace=False
        )
        pixel_auc = roc_auc_score(all_pixel_labels[idx], all_pixel_preds[idx])
    else:
        pixel_auc = roc_auc_score(all_pixel_labels, all_pixel_preds)

    print("\n" + "="*50)
    print(f"{'Category':<20} | {args.category}")
    print(f"{'Image-level AUC':<20} | {image_auc:.4f}")
    print(f"{'Pixel-level AUC':<20} | {pixel_auc:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help="Path to dataset root folder")
    parser.add_argument('--category', type=str, required=True, help="Category name (e.g., macaroni1)")
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints", help="Directory containing .pth and .npy files")
    
    args = parser.parse_args()
    
    try:
        load_dinov2_model()
        run_evaluation(args)
    except Exception:
        traceback.print_exc()
