import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR1=r'C:\Users\amnit\Desktop\rank1_codeV2\rank1_codeV2\rank1_code'
IMG_DIR2=r'C:\Users\amnit\Downloads\spot-diff-main\spot-diff-main\VisA_pytorch\1cls'
#
# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "SingleCandGatedInp_train.pth")
BANK_SAVE_PATH = os.path.join(SCRIPT_DIR, "models", "SingleCandGatedcpr_bank.npy")
#VALIDATE_PATH = os.path.join(IMG_DIR1, "mvtec_ad_2", "metal_nut", "test", "scratch")
VALIDATE_PATH = os.path.join(IMG_DIR2, 'candle', "test", "bad")

# ----------------------------
# DINOv2 / Feature Extractor
# ----------------------------
DINO_BASE_SIZE = 518
DINO_PATCH_SIZE = 14  # ViT-B/14

_feature_extractor = None

def load_dinov2_model():
    global _feature_extractor
    if _feature_extractor is not None:
        return _feature_extractor
    print(f"Loading DINOv2 model on {DEVICE}")
    try:
        _feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(DEVICE)
        _feature_extractor.eval()
        patch_sz = getattr(_feature_extractor.patch_embed, "patch_size", None)
        if isinstance(patch_sz, (tuple, list)):
            patch_sz = patch_sz[0]
        print(f"DINOv2 loaded. patch_size={patch_sz}")
    except Exception as e:
        print("Failed to load DINOv2. Using dummy features.", flush=True)
        print(e)
        _feature_extractor = None
    return _feature_extractor

def preprocess_image(img_np: np.ndarray, size: tuple):
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return t(img_np).unsqueeze(0).to(DEVICE)

def extract_spatial_features(img_np: np.ndarray, strategy="single-scale"):
    model = load_dinov2_model()
    if model is None:
        grid = DINO_BASE_SIZE // DINO_PATCH_SIZE
        feat_dim = 768*3 if strategy=="multi-scale" else 768
        return np.random.randn(grid, grid, feat_dim).astype(np.float32)
    
    patch_size = model.patch_embed.patch_size[0]
    with torch.no_grad():
        if strategy=="single-scale":
            img_t = preprocess_image(img_np, (DINO_BASE_SIZE, DINO_BASE_SIZE))
            features = model.get_intermediate_layers(img_t, n=1, reshape=True)[0]
        else:
            scales = [1.0, 0.75, 0.5]
            feats = []
            for s in scales:
                h = int(DINO_BASE_SIZE*s)//patch_size*patch_size
                w = int(DINO_BASE_SIZE*s)//patch_size*patch_size
                if h==0 or w==0:
                    continue
                img_t = preprocess_image(img_np, (h,w))
                f = model.get_intermediate_layers(img_t, n=1, reshape=True)[0]
                f = F.interpolate(f, size=(DINO_BASE_SIZE//patch_size, DINO_BASE_SIZE//patch_size),
                                  mode='bilinear', align_corners=False)
                feats.append(f)
            features = torch.cat(feats, dim=1)
    features = features.squeeze(0).permute(1,2,0).cpu().numpy()  # (H_grid, W_grid, C)
    return features

# ----------------------------
# CPR reconstruction
# ----------------------------
def cpr_reconstruct_from_bank(feature_map_np, bank_normed, bank_raw):
    L,C = feature_map_np.reshape(-1, feature_map_np.shape[2]).shape
    q = torch.from_numpy(feature_map_np.reshape(-1,C)).to(DEVICE).float()
    q_norm = F.normalize(q, dim=1)
    bank_t = torch.from_numpy(bank_normed).to(DEVICE).float().t()
    sims = torch.matmul(q_norm, bank_t)
    top1 = torch.argmax(sims, dim=1)
    bank_raw_t = torch.from_numpy(bank_raw).to(DEVICE).float()
    recon = bank_raw_t[top1].cpu().numpy()
    H,W,C = feature_map_np.shape
    return recon.reshape(H,W,C)

# ----------------------------
# INP_Former (trained code)
# ----------------------------
class MockTransformerBlock(nn.Module):
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

class INP_Former(nn.Module):
    def __init__(self, feature_dim, grid_h, grid_w, num_layers=2):
        super().__init__()
        self.bottleneck = nn.ModuleList([MockTransformerBlock(feature_dim) for _ in range(num_layers)])
        self.aggregation = nn.ModuleList([MockTransformerBlock(feature_dim)])
        self.decoder = nn.ModuleList([MockTransformerBlock(feature_dim) for _ in range(num_layers)])
        self.prototype_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.grid_h, self.grid_w, self.feature_dim = grid_h, grid_w, feature_dim
        self.gate_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, x):
        B, C, H, W = x.shape
        patch_tokens = x.flatten(2).permute(0, 2, 1)
        agg_proto = self.prototype_token.repeat(B, 1, 1)
        for blk in self.aggregation:
            agg_proto = blk(agg_proto, patch_tokens)
        z = patch_tokens
        for blk in self.bottleneck:
            z = blk(z)
        de_list = [blk(z, agg_proto) for blk in self.decoder]
        de_stack = torch.stack(de_list, dim=1)
        gates = torch.softmax(self.gate_weights, dim=0)
        de_fused = (gates.view(1, -1, 1, 1) * de_stack).sum(dim=1)
        de_map = de_fused.permute(0, 2, 1).reshape(B, C, H, W)
        return None, de_map

def load_inp_model(model_path, feature_dim, grid_h, grid_w):
    model = INP_Former(feature_dim, grid_h, grid_w, num_layers=2).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

# ----------------------------
# Test images
# ----------------------------
def gather_images(folder, num_images=12 ):
    exts = ['.png','.jpg','.jpeg','.bmp']
    all_images = [os.path.join(folder,f) for f in os.listdir(folder)
                  if os.path.splitext(f)[1].lower() in exts]
    return sorted(all_images)[15:25]#[:num_images]

# ----------------------------
# Main inference
# ----------------------------
if __name__=="__main__":
    # Load CPR
    bank_raw = np.load(BANK_SAVE_PATH)
    bank_normed = np.load(BANK_SAVE_PATH.replace(".npy","_norm.npy"))
    print("✅ Loaded CPR bank")

    # Load DINOv2
    dinov2_model = load_dinov2_model()
    print("✅ Loaded DINOv2")

    # Gather test images
    test_imgs = gather_images(VALIDATE_PATH)
    print(f"Testing {len(test_imgs)} images")

    # Use first image to get feature dim
    img0 = np.array(Image.open(test_imgs[0]).convert("RGB"))
    feat0 = extract_spatial_features(img0, strategy="single-scale")
    H_feat, W_feat, C_feat = feat0.shape
    print("Feature grid:", H_feat, W_feat, "feature_dim:", C_feat)

    # Load INP model
    inp_model = load_inp_model(MODEL_PATH, feature_dim=C_feat, grid_h=H_feat, grid_w=W_feat)
    print("✅ Loaded INP model")
    # Load INP model

    # ===== INSERT HERE =====
    print("\n===== RGAE (formerly INP) Model Parameters =====")

    total_params = 0
    trainable_params = 0

    for name, param in inp_model.named_parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num
        print(f"{name:40s} | shape={tuple(param.shape)} | params={num:,}")

    print("\n---------------------------------------------")
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print("---------------------------------------------\n")
    # ======================


    # Run inference
    os.makedirs(os.path.join(SCRIPT_DIR,"models"), exist_ok=True)

    for i, img_path in enumerate(test_imgs):
        img_np = np.array(Image.open(img_path).convert("RGB"))

        start_total = time.perf_counter()

        # ---- Feature extraction ----
        t_feat_start = time.perf_counter()
        feat_np = extract_spatial_features(img_np, strategy="single-scale")
        t_feat_end = time.perf_counter()

        # ---- INP forward ----
        t = torch.from_numpy(feat_np).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
        t_inp_start = time.perf_counter()
        with torch.no_grad():
            _, recon = inp_model(t)
        t_inp_end = time.perf_counter()

        recon_np = recon.squeeze(0).permute(1,2,0).cpu().numpy()

        # ---- CPR reconstruction ----
        t_cpr_start = time.perf_counter()
        recon_bank = cpr_reconstruct_from_bank(feat_np, bank_normed, bank_raw)
        t_cpr_end = time.perf_counter()

        # ---- Anomaly map ----
        amap = np.mean((recon_np - recon_bank)**2, axis=2)
        H_orig, W_orig = img_np.shape[:2]
        amap_up = F.interpolate(
            torch.from_numpy(amap).unsqueeze(0).unsqueeze(0),
            size=(H_orig, W_orig),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        amap_norm = (amap_up - amap_up.min()) / (np.ptp(amap_up) + 1e-8)
        amap_img = (255 * amap_norm).astype(np.uint8)

        # ---- Overlay ----
        heatmap = cv2.applyColorMap(amap_img, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        out_path = os.path.join(SCRIPT_DIR, "models", f"anomaly_overlay_{i}.png")
        cv2.imwrite(out_path, overlay)
        print(f"Saved overlay anomaly map {i} -> {out_path}")

        end_total = time.perf_counter()
        total_time = end_total - start_total  # seconds
        fps = 1.0 / total_time if total_time > 0 else 0.0

        print(
            f"[Image {i}] "
            f"Feature: {(t_feat_end - t_feat_start)*1000:.1f} ms | "
            f"INP: {(t_inp_end - t_inp_start)*1000:.1f} ms | "
            f"CPR: {(t_cpr_end - t_cpr_start)*1000:.1f} ms | "
            f"Total: {(end_total - start_total)*1000:.1f} ms"
            f"FPS: {fps:.2f}"
        )


