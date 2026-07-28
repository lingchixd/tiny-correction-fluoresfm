"""
Baseline evaluation (CPU only; no GPU or model needed).

For each held-out image, computes two references against the GT (10x panel):
  - provided_lr : the provided low-resolution panel directly against GT
                  (the main baseline used in the paper)
  - bicubic     : LR downsampled by sf then bicubic-upsampled back, a standard
                  bicubic interpolation reference symmetric to the model path

Protocol matches the main evaluation: percentile normalization (0.03 / 0.995),
the fine-tuning image excluded by filename, same held-out set.
Output: baseline_compare.csv (one row per image, plus mean and std).

These are the "Provided LR (baseline)" and "Bicubic (ref.)" rows in Table 1.

Usage (CPU is enough; can run alongside the GPU evaluation):
    python baseline_bicubic.py
Edit the paths in CONFIG.
"""
import os, csv
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# ============== configuration, consistent with the main evaluation ==============
CONFIG = {
    "data_dir": r"./data/spleen",
    "crop_panels": 3, "panel_lr": 2, "panel_hr": 1,
    "sf_lr": 2,                                          # same scale factor as the main evaluation
    "finetune_sample_name": "Spleen_10x fused 2.png",   # fine-tuning image, excluded from evaluation
    "output_dir": r"./results_progressive",
}
# ===============================================================================

def load_panel(path, crop_panels, panel_idx):
    img = Image.open(path); W, H = img.size
    pw = W // crop_panels; left = pw * panel_idx
    img = img.crop((left, 0, left+pw, H))
    return np.array(img.convert("L")).astype(np.float32)

def normalize_percentile(img, p_low=0.03, p_high=0.995):
    lo = np.percentile(img, p_low*100); hi = np.percentile(img, p_high*100)
    return np.clip((img-lo)/(hi-lo+1e-8), 0, None)

def compute_metrics(pred, gt):
    pred_n, gt_n = normalize_percentile(pred), normalize_percentile(gt)
    return psnr_fn(gt_n, pred_n, data_range=1.0), ssim_fn(gt_n, pred_n, data_range=1.0)

def bicubic_roundtrip(img, sf):
    """Standard bicubic reference symmetric to the model path: the LR panel is
    already a low-quality image at GT size, so downsample by sf with bicubic and
    upsample back by sf with bicubic. Implemented with PIL, no torch dependency.
    Input is assumed normalized to ~[0, 1]."""
    H, W = img.shape
    pil = Image.fromarray((np.clip(img, 0, 1) * 255.0).astype(np.float32)).convert("F")
    down = pil.resize((max(W//sf, 1), max(H//sf, 1)), Image.BICUBIC)
    up   = down.resize((W, H), Image.BICUBIC)
    return np.asarray(up, dtype=np.float32) / 255.0

def main():
    cfg = CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)
    files = sorted([f for f in os.listdir(cfg["data_dir"]) if f.endswith(".png")])
    eval_files = [f for f in files if f != cfg["finetune_sample_name"]]
    print(f"[INFO] {len(files)} images, {len(eval_files)} held out "
          f"(fine-tuning image {cfg['finetune_sample_name']} excluded)")

    rows = []
    for j, fname in enumerate(eval_files):
        path = os.path.join(cfg["data_dir"], fname)
        img_lr = load_panel(path, cfg["crop_panels"], cfg["panel_lr"])
        img_hr = load_panel(path, cfg["crop_panels"], cfg["panel_hr"])

        # Reference 1: provided LR directly against GT (main baseline)
        p_lr, s_lr = compute_metrics(img_lr, img_hr)
        # Reference 2: standard bicubic roundtrip
        img_bic = bicubic_roundtrip(normalize_percentile(img_lr), cfg["sf_lr"])
        p_bic, s_bic = compute_metrics(img_bic, img_hr)

        rows.append((fname, p_lr, s_lr, p_bic, s_bic))
        print(f"  [{j+1}/{len(eval_files)}] {fname[:30]:30s}  "
              f"providedLR PSNR {p_lr:5.2f} SSIM {s_lr:.4f} | "
              f"bicubic PSNR {p_bic:5.2f} SSIM {s_bic:.4f}")

    arr = np.array([(r[1], r[2], r[3], r[4]) for r in rows])
    mean = arr.mean(0); std = arr.std(0)

    out = os.path.join(cfg["output_dir"], "baseline_compare.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename","psnr_providedLR","ssim_providedLR","psnr_bicubic","ssim_bicubic"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.5f}", f"{r[3]:.4f}", f"{r[4]:.5f}"])
        w.writerow(["MEAN", f"{mean[0]:.4f}", f"{mean[1]:.5f}", f"{mean[2]:.4f}", f"{mean[3]:.5f}"])
        w.writerow(["STD",  f"{std[0]:.4f}",  f"{std[1]:.5f}",  f"{std[2]:.4f}",  f"{std[3]:.5f}"])

    print("\n" + "="*60)
    print("BASELINE summary (n=%d)" % len(eval_files))
    print("="*60)
    print(f"{'':16}{'PSNR':>12}{'SSIM':>12}")
    print(f"{'provided LR':16}{mean[0]:>12.4f}{mean[1]:>12.4f}")
    print(f"{'bicubic':16}{mean[2]:>12.4f}{mean[3]:>12.4f}")
    print(f"\n[INFO] saved: {out}")

if __name__ == "__main__":
    main()
