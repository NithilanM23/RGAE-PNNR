import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import traceback

# ----------------------------
# Settings
# ----------------------------
DATA_ROOT = r"C:\Users\amnit\Downloads\spot-diff-main\spot-diff-main\VisA_pytorch\1cls"
CATEGORY = "macaroni1"

PIXEL_SUBSAMPLE_PERCENT = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

DINO_BASE_SIZE = 518
DINO_PATCH_SIZE = 14

# ----------------------------
# DINOv2 Feature Extractor
# ----------------------------
_feature_extractor = None

def load_dinov2_model():
    global _feature_extractor
    if _feature_extractor is not None:
        return _feature_extractor

    print(f"\nLoading DINOv2 model on {DEVICE}...")
    try:
        _feature_extractor = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14',
            verbose=False
        ).to(DEVICE)
        _feature_extractor.eval()
        print("DINOv2 model loaded successfully.")
    except Exception:
        print("Failed to load DINOv2. Using dummy features.")
        _feature_extractor = None
    return _feature_extractor


def preprocess_image(img_np: np.ndarray, size: tuple):
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return t(img_np).unsqueeze(0).to(DEVICE)


def extract_spatial_features(img_np: np.ndarray, strategy="single-scale"):
    model = load_dinov2_model()

    if model is None:
        grid = DINO_BASE_SIZE // DINO_PATCH_SIZE
        feat_dim = 768 * (3 if strategy == "multi-scale" else 1)
        return np.random.randn(grid, grid, feat_dim).astype(np.float32)

    patch_size = model.patch_embed.patch_size[0]

    with torch.no_grad():
        if strategy == "single-scale":
            img_t = preprocess_image(img_np, (DINO_BASE_SIZE, DINO_BASE_SIZE))
            features = model.get_intermediate_layers(img_t, n=1, reshape=True)[0]
        else:
            scales = [1.0, 0.75, 0.5]
            feats = []
            for s in scales:
                h = int(DINO_BASE_SIZE * s) // patch_size * patch_size
                w = int(DINO_BASE_SIZE * s) // patch_size * patch_size
                if h == 0 or w == 0:
                    continue
                img_t = preprocess_image(img_np, (h, w))
                f = model.get_intermediate_layers(img_t, n=1, reshape=True)[0]
                f = F.interpolate(
                    f,
                    size=(DINO_BASE_SIZE // patch_size, DINO_BASE_SIZE // patch_size),
                    mode="bilinear",
                    align_corners=False
                )
                feats.append(f)
            features = torch.cat(feats, dim=1)

    return features.squeeze(0).permute(1, 2, 0).cpu().numpy()


def pnnr_reconstruct_from_bank(feature_map_np, bank_normed, bank_raw):
    H, W, C = feature_map_np.shape

    q = torch.from_numpy(feature_map_np.reshape(-1, C)).to(DEVICE).float()
    q_norm = F.normalize(q, dim=1)

    bank_t = torch.from_numpy(bank_normed).to(DEVICE).float().t()
    sims = torch.matmul(q_norm, bank_t)
    top1 = torch.argmax(sims, dim=1)

    bank_raw_t = torch.from_numpy(bank_raw).to(DEVICE).float()
    recon = bank_raw_t[top1].cpu().numpy()

    return recon.reshape(H, W, C)


# ----------------------------
# Anomaly Aggregator
# ----------------------------
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

        mean = flat.mean()
        var = flat.var()
        maxv = flat.max()

        k = max(1, int(0.01 * flat.numel()))
        topk = torch.topk(flat, k).values.mean()

        stats = torch.stack([mean, var, maxv, topk]).unsqueeze(0)
        weight = self.mlp(stats).squeeze()

        return weight * topk + (1 - weight) * mean


# ----------------------------
# RGAE Blocks
# ----------------------------
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


class RGAE(nn.Module):
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

        latent_dim = feature_dim // bottleneck_ratio

        self.aggregation = nn.ModuleList([
            MockTransformerBlock(feature_dim)
        ])

        self.prototype_token = nn.Parameter(torch.randn(1, 1, feature_dim))

        self.pre_bottleneck = nn.Linear(feature_dim, latent_dim)
        self.bottleneck = nn.ModuleList([
            MockTransformerBlock(latent_dim) for _ in range(num_layers)
        ])
        self.post_bottleneck = nn.Linear(latent_dim, feature_dim)

        self.decoder = nn.ModuleList([
            MockTransformerBlock(feature_dim) for _ in range(num_layers)
        ])

        self.gate_weights = nn.Parameter(torch.ones(num_layers))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        B, C, H, W = x.shape

        patch_tokens = x.flatten(2).permute(0, 2, 1)
        proto = self.prototype_token.expand(B, -1, -1)

        for blk in self.aggregation:
            proto = blk(proto, patch_tokens)

        z = self.pre_bottleneck(patch_tokens)
        z = self.dropout(z)

        for blk in self.bottleneck:
            z = blk(z)

        z = self.post_bottleneck(z)

        decoded = []
        for blk in self.decoder:
            decoded.append(blk(z, proto))

        decoded = torch.stack(decoded, dim=1)
        gates = torch.softmax(self.gate_weights, dim=0)
        decoded = (gates.view(1, -1, 1, 1) * decoded).sum(dim=1)

        return None, decoded.permute(0, 2, 1).reshape(B, C, H, W)


# ----------------------------
# Evaluation
# ----------------------------
def run_evaluation(category):
    print(f"\n{'='*20} Evaluating category: {category} {'='*20}")

    model_save_dir = os.path.join(SCRIPT_DIR, "models")
    model_path = os.path.join(model_save_dir, "VisBank3Mac1GatedRGAE_train.pth")
    bank_path = os.path.join(model_save_dir, "VisBank3Mac1Gatedpnnr_bank.npy")

    test_good_dir, test_bad_dir, mask_dir = [
        os.path.join(DATA_ROOT, category, d)
        for d in ["test/good", "test/bad", "ground_truth/bad"]
    ]

    if not all(os.path.exists(p) for p in [model_path, bank_path]):
        print("ERROR: Model files not found.")
        return

    dummy_feat = extract_spatial_features(np.zeros((100, 100, 3), dtype=np.uint8))
    grid_h, grid_w, feature_dim = dummy_feat.shape

    model = RGAE(feature_dim, grid_h, grid_w).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    aggregator = AnomalyAggregator().to(DEVICE).eval()

    pnnr_bank_raw = np.load(bank_path)
    pnnr_bank_normed = pnnr_bank_raw / (
        np.linalg.norm(pnnr_bank_raw, axis=1, keepdims=True) + 1e-8
    )

    image_scores, image_labels = [], []
    all_pixel_preds, all_pixel_labels = [], []

    good_images = [os.path.join(test_good_dir, f) for f in os.listdir(test_good_dir)]
    bad_images = [os.path.join(test_bad_dir, f) for f in os.listdir(test_bad_dir)]

    for img_path in tqdm(good_images + bad_images, desc=f"Testing {category}", leave=False):
        is_bad = "bad" in os.path.dirname(img_path)

        img_np = np.array(Image.open(img_path).convert("RGB"))
        feat_np = extract_spatial_features(img_np)

        with torch.no_grad():
            t = torch.from_numpy(feat_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            _, recon_rgae = model(t)
            recon_rgae_np = recon_rgae.squeeze(0).permute(1, 2, 0).cpu().numpy()

        recon_pnnr = pnnr_reconstruct_from_bank(
            feat_np, pnnr_bank_normed, pnnr_bank_raw
        )

        anomaly_map = np.mean((recon_rgae_np - recon_pnnr) ** 2, axis=2)

        with torch.no_grad():
            score = aggregator(torch.from_numpy(anomaly_map).to(DEVICE)).item()

        image_scores.append(score)
        image_labels.append(1 if is_bad else 0)

        H, W = img_np.shape[:2]
        amap_up = F.interpolate(
            torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()

        all_pixel_preds.append(amap_up.flatten())

        if is_bad:
            mask = np.array(
                Image.open(os.path.join(mask_dir, os.path.basename(img_path).replace(".JPG", ".png")))
            )
            all_pixel_labels.append((mask > 0).astype(np.uint8).flatten())
        else:
            all_pixel_labels.append(np.zeros(H * W, dtype=np.uint8))

    image_auc = roc_auc_score(image_labels, image_scores)

    all_pixel_preds = np.concatenate(all_pixel_preds)
    all_pixel_labels = np.concatenate(all_pixel_labels)

    idx = np.random.RandomState(42).choice(
        len(all_pixel_labels),
        int(len(all_pixel_labels) * PIXEL_SUBSAMPLE_PERCENT),
        replace=False
    )

    pixel_auc = roc_auc_score(all_pixel_labels[idx], all_pixel_preds[idx])

    print("\n" + "="*50)
    print(f"{'Category':<20} | {CATEGORY}")
    print(f"{'Image-level AUC':<20} | {image_auc:.4f}")
    print(f"{'Pixel-level AUC':<20} | {pixel_auc:.4f}")
    print("="*50)


if __name__ == "__main__":
    try:
        load_dinov2_model()
        run_evaluation(CATEGORY)
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
