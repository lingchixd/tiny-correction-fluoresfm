"""
Progressive unfreezing: batch evaluation.

Runs inference for each checkpoint under the progressive/ directory on the same
held-out set (excluding the single fine-tuning image), and computes PSNR/SSIM.

Outputs two tables:
  - curve_summary.csv    : one row per tier (strategy, trainable_params, %,
                           mean/std PSNR, mean/std SSIM). Used for the
                           "parameter budget vs. performance" curve (Fig. 1).
  - curve_persample.csv  : one row per tier per image. Used for the paired
                           statistical tests (Table 1) and Holm correction.

Reproduces the numbers in Table 1 and the curve in Fig. 1 of the paper.

Note: fine-tuning uses a fixed random seed (see finetune_progressive.py), so the
checkpoints and reported metrics are reproducible. Metrics are means over the 24
held-out images.

Usage:
    python evaluate_progressive.py
Edit the paths in CONFIG to point to your local data and checkpoints.
"""

import os, csv, json, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import _MODEL_CONFIGS

from models.unet_sd_c import UNetModel
import utils.optim as utils_optim

# ==============================================================================
CONFIG = {
    # Directory holding the paired FUSE images (three-panel PNGs).
    "data_dir": r"./data/spleen",

    "crop_panels": 3, "panel_lr": 2, "panel_hr": 1,

    # Directory of the progressively-unfrozen checkpoints, and the tier order
    # (determines the x-axis of the budget curve, from fewest to most params).
    "ckpt_dir": r"./checkpoints/fluoresfm/progressive",
    "strategies": ["out_only", "in_only", "in_out", "outer2", "outer4", "all"],

    # Pretrained (no fine-tuning) checkpoint, used as a reference line.
    "checkpoint_pretrained": r"./checkpoints/fluoresfm/pretrained.pt",

    "biomedclip_json": r"./checkpoints/clip/biomedclip/open_clip_config.json",
    "biomedclip_bin":  r"./checkpoints/clip/biomedclip/open_clip_pytorch_model.bin",

    "prompt": (
        "Task: super-resolution with a scale factor of 2; "
        "sample: fixed spleen tissue; structure: nuclei; "
        "fluorescence indicator: Hoechst; "
        "input microscope: wide-field microscope with excitation numerical aperture (NA) of 0.1, "
        "detection numerical aperture (NA) of 0.1; "
        "input pixel size: 1850 x 1850 nm. Nearest interpolation with a factor of 2; "
        "target microscope: wide-field microscope with excitation numerical aperture (NA) of 0.25, "
        "detection numerical aperture (NA) of 0.25; "
        "target pixel size: 925 x 925 nm."
    ),

    "sf_lr": 2, "patch_size": 64, "overlap": 16, "amp": True,
    "device": "cuda:0",

    # The single image used for fine-tuning. Excluded from evaluation by filename
    # so it never enters the held-out set.
    "finetune_sample_name": "Spleen_10x fused 2.png",

    # Number of held-out images to evaluate. Use None for the full held-out set
    # (this reproduces the paper). Set to a small integer only for a quick preview.
    "max_eval_images": None,

    "output_dir": r"./results_progressive",
}
# ==============================================================================


class Patch_stitcher:
    def __init__(self, patch_size=64, overlap=0, padding_mode="constant"):
        self.ps, self.ol, self.padding_mode = patch_size, overlap, padding_mode
        self._generate_mask()

    def _generate_mask(self):
        ps, ol = self.ps, self.ol
        self.patch_mask_lu    = np.pad(np.ones((1,1,ps-ol,ps-ol)),     ((0,0),(0,0),(0,ol+1),(0,ol+1)),      "linear_ramp")[..., 0:-1, 0:-1]
        self.patch_mask_mu    = np.pad(np.ones((1,1,ps-ol,ps-2*ol)),   ((0,0),(0,0),(0,ol+1),(ol+1,ol+1)),   "linear_ramp")[..., 0:-1, 1:-1]
        self.patch_mask_ru    = np.pad(np.ones((1,1,ps-ol,ps-ol)),     ((0,0),(0,0),(0,ol+1),(ol+1,0)),      "linear_ramp")[..., 0:-1, 1:]
        self.patch_mask_lm    = np.pad(np.ones((1,1,ps-2*ol,ps-ol)),   ((0,0),(0,0),(ol+1,ol+1),(0,ol+1)),   "linear_ramp")[..., 1:-1, 0:-1]
        self.patch_mask_mm    = np.pad(np.ones((1,1,ps-2*ol,ps-2*ol)), ((0,0),(0,0),(ol+1,ol+1),(ol+1,ol+1)),"linear_ramp")[..., 1:-1, 1:-1]
        self.patch_mask_rm    = np.pad(np.ones((1,1,ps-2*ol,ps-ol)),   ((0,0),(0,0),(ol+1,ol+1),(ol+1,0)),   "linear_ramp")[..., 1:-1, 1:]
        self.patch_mask_lb    = np.pad(np.ones((1,1,ps-ol,ps-ol)),     ((0,0),(0,0),(ol+1,0),(0,ol+1)),      "linear_ramp")[..., 1:, 0:-1]
        self.patch_mask_mb    = np.pad(np.ones((1,1,ps-ol,ps-2*ol)),   ((0,0),(0,0),(ol+1,0),(ol+1,ol+1)),   "linear_ramp")[..., 1:, 1:-1]
        self.patch_mask_rb    = np.pad(np.ones((1,1,ps-ol,ps-ol)),     ((0,0),(0,0),(ol+1,0),(ol+1,0)),      "linear_ramp")[..., 1:, 1:]
        self.patch_mask_lu_01 = np.pad(np.ones((1,1,ps-ol,ps)),        ((0,0),(0,0),(0,ol+1),(0,0)),         "linear_ramp")[..., 0:-1, :]
        self.patch_mask_lm_01 = np.pad(np.ones((1,1,ps-2*ol,ps)),      ((0,0),(0,0),(ol+1,ol+1),(0,0)),      "linear_ramp")[..., 1:-1, :]
        self.patch_mask_lb_01 = np.pad(np.ones((1,1,ps-ol,ps)),        ((0,0),(0,0),(ol+1,0),(0,0)),         "linear_ramp")[..., 1:, :]
        self.patch_mask_lu_10 = np.pad(np.ones((1,1,ps,ps-ol)),        ((0,0),(0,0),(0,0),(0,ol+1)),         "linear_ramp")[..., 0:-1]
        self.patch_mask_mu_10 = np.pad(np.ones((1,1,ps,ps-2*ol)),      ((0,0),(0,0),(0,0),(ol+1,ol+1)),      "linear_ramp")[..., 1:-1]
        self.patch_mask_ru_10 = np.pad(np.ones((1,1,ps,ps-ol)),        ((0,0),(0,0),(0,0),(ol+1,0)),         "linear_ramp")[..., 1:]
        self.patch_mask_lu_11 = np.ones((1,1,ps,ps))

    def unfold(self, img):
        Ny, Nx = img.shape[-2], img.shape[-1]
        step = self.ps - self.ol
        npy  = math.ceil((Ny - self.ps) / step) + 1
        npx  = math.ceil((Nx - self.ps) / step) + 1
        Ny_pad, Nx_pad = npy*step+self.ol, npx*step+self.ol
        img_pad = F.pad(img, (0, Nx_pad-Nx, 0, Ny_pad-Ny), mode=self.padding_mode)
        patches = torch.zeros((npy*npx, img.shape[1], self.ps, self.ps),
                              device=img_pad.device, dtype=img_pad.dtype)
        for i in range(npy):
            for j in range(npx):
                patches[i*npx+j] = img_pad[0,:, i*step:i*step+self.ps, j*step:j*step+self.ps]
        return patches

    def fold_linear_ramp(self, patches, original_image_shape):
        if isinstance(patches, torch.Tensor): patches = patches.cpu().numpy()
        bs, nc, Ny, Nx = original_image_shape
        step = self.ps - self.ol
        npy  = math.ceil((Ny - self.ps) / step) + 1
        npx  = math.ceil((Nx - self.ps) / step) + 1
        N = npy*npx
        patches   = patches.reshape(N, bs, nc, self.ps, self.ps)
        patch_pad = np.zeros((bs, nc, npy*step+self.ol, npx*step+self.ol))
        mlu = self.patch_mask_lu_11 if (npx==1 and npy==1) else self.patch_mask_lu_01 if npx==1 else self.patch_mask_lu_10 if npy==1 else self.patch_mask_lu
        mmu = self.patch_mask_mu_10 if npy==1 else self.patch_mask_mu
        mru = self.patch_mask_ru_10 if npy==1 else self.patch_mask_ru
        mlm = self.patch_mask_lm_01 if npx==1 else self.patch_mask_lm
        mlb = self.patch_mask_lb_01 if npx==1 else self.patch_mask_lb
        masks = [[mlu, mmu, mru],[mlm, self.patch_mask_mm, self.patch_mask_rm],[mlb, self.patch_mask_mb, self.patch_mask_rb]]
        for i in range(npy):
            ri = 0 if i==0 else (2 if i==npy-1 else 1)
            for j in range(npx):
                ci = 0 if j==0 else (2 if j==npx-1 else 1)
                patch = patches[i*npx+j].copy()
                if self.ol > 0: patch *= masks[ri][ci]
                patch_pad[:,:, i*step:i*step+self.ps, j*step:j*step+self.ps] += patch
        return patch_pad[..., :Ny, :Nx]


def load_panel(path, crop_panels, panel_idx):
    img = Image.open(path); W, H = img.size
    pw = W // crop_panels; left = pw*panel_idx
    img = img.crop((left, 0, left+pw, H))
    return np.array(img.convert("L")).astype(np.float32)

def normalize_percentile(img, p_low=0.03, p_high=0.995):
    lo = np.percentile(img, p_low*100); hi = np.percentile(img, p_high*100)
    return np.clip((img-lo)/(hi-lo+1e-8), 0, None)

def interp_sf(img, sf):
    if sf==1: return img
    t = torch.from_numpy(img[None,None])
    return F.interpolate(t, scale_factor=sf, mode="nearest")[0,0].numpy()

def downsample_sf(img, sf):
    if sf==1: return img
    t = torch.from_numpy(img[None,None])
    return F.avg_pool2d(t, kernel_size=sf, stride=sf)[0,0].numpy()

def compute_metrics(pred, gt):
    pred_n, gt_n = normalize_percentile(pred), normalize_percentile(gt)
    return psnr_fn(gt_n, pred_n, data_range=1.0), ssim_fn(gt_n, pred_n, data_range=1.0)

def load_model(ckpt_path, device):
    model = UNetModel(in_channels=1, out_channels=1, channels=320, n_res_blocks=1,
                      attention_levels=[0,1,2,3], channel_multipliers=[1,2,4,4],
                      n_heads=8, tf_layers=1, d_cond=768, pixel_shuffle=False, scale_factor=4).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = utils_optim.on_load_checkpoint(ckpt["model_state_dict"])
    model.load_state_dict(sd); model.eval()
    meta = {"trainable_params": ckpt.get("trainable_params"),
            "total_params": ckpt.get("total_params")}
    return model, meta

def load_embedder(cfg, device):
    with open(cfg["biomedclip_json"]) as f: config = json.load(f)
    name = "biomedclip_local"
    if name not in _MODEL_CONFIGS: _MODEL_CONFIGS[name] = config["model_cfg"]
    tok = get_tokenizer(name)
    clip_model, _, _ = create_model_and_transforms(model_name=name, pretrained=cfg["biomedclip_bin"],
        **{f"image_{k}": v for k,v in config["preprocess_cfg"].items()})
    clip_model.to(device); tm = clip_model.text; tm.output_tokens = True; tm.eval()
    return tok, tm

@torch.no_grad()
def encode_prompt(prompt, tok, tm, device, cl=160):
    tokens = tok([prompt], context_length=cl).to(device)
    _, feats = tm(tokens); return feats

@torch.no_grad()
def run_inference(model, img_lr_up, text_emb, cfg, device, tag=""):
    H, W = img_lr_up.shape
    st = Patch_stitcher(patch_size=cfg["patch_size"], overlap=cfg["overlap"], padding_mode="reflect")
    img_t = torch.from_numpy(img_lr_up[None,None]).to(device)
    patches = st.unfold(img_t)
    N = patches.shape[0]; bs = min(32, N)
    out = torch.zeros_like(patches)
    for i in range(0, N, bs):
        batch = patches[i:i+bs].to(device)
        t = torch.zeros(batch.shape[0], device=device)
        cond = text_emb.expand(batch.shape[0], -1, -1)
        with torch.autocast("cuda", torch.float16, enabled=(cfg["amp"] and device.type=="cuda")):
            pred = model(batch, t, cond)
        out[i:i+bs] = pred.cpu()
    step = cfg["patch_size"]-cfg["overlap"]
    npy = math.ceil((H-cfg["patch_size"])/step)+1
    npx = math.ceil((W-cfg["patch_size"])/step)+1
    restored = st.fold_linear_ramp(out, (1,1,npy*step+cfg["overlap"], npx*step+cfg["overlap"]))
    restored = restored[0,0,:H,:W].astype(np.float32)
    return downsample_sf(restored, cfg["sf_lr"])


def evaluate_checkpoint(ckpt_path, files, text_emb, cfg, device, strategy_name=""):
    """Evaluate one checkpoint on the held-out set; return per-sample (psnr, ssim) and meta."""
    model, meta = load_model(ckpt_path, device)
    rows = []
    eval_files = [f for f in files if f != cfg["finetune_sample_name"]]
    if cfg.get("max_eval_images") is not None:
        eval_files = eval_files[:cfg["max_eval_images"]]
    n_total = len(eval_files)
    for j, fname in enumerate(eval_files):
        path = os.path.join(cfg["data_dir"], fname)
        img_lr = load_panel(path, cfg["crop_panels"], cfg["panel_lr"])
        img_hr = load_panel(path, cfg["crop_panels"], cfg["panel_hr"])
        img_lr_up = interp_sf(normalize_percentile(img_lr), cfg["sf_lr"])
        restored  = run_inference(model, img_lr_up, text_emb, cfg, device,
                                  tag=f"[{strategy_name} {j+1}/{n_total}]")
        p, s = compute_metrics(restored, img_hr)
        rows.append((fname, p, s))
        print(f"    [{strategy_name}] {j+1}/{n_total}  {fname[:30]:30s}  "
              f"PSNR {p:5.2f}  SSIM {s:.4f}")
    del model; torch.cuda.empty_cache()
    return rows, meta


def main():
    cfg = CONFIG; device = torch.device(cfg["device"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    files = sorted([f for f in os.listdir(cfg["data_dir"]) if f.endswith(".png")])
    n_eval = len([f for f in files if f != cfg["finetune_sample_name"]])
    print(f"[INFO] {len(files)} images, {n_eval} held out "
          f"(fine-tuning image {cfg['finetune_sample_name']} excluded)")

    tok, tm = load_embedder(cfg, device)
    text_emb = encode_prompt(cfg["prompt"], tok, tm, device)
    del tok, tm; torch.cuda.empty_cache()

    persample_rows = []   # (strategy, fname, psnr, ssim)
    summary_rows   = []   # (strategy, trainable, pct, psnr_mean, psnr_std, ssim_mean, ssim_std)

    # Pretrained (no fine-tuning) reference line; trainable = 0.
    eval_list = [("pretrained", cfg["checkpoint_pretrained"], 0)]
    for s in cfg["strategies"]:
        eval_list.append((s, os.path.join(cfg["ckpt_dir"], f"spleen_{s}.pt"), None))

    total_params = 683650561  # total model parameters, used for the percentage column

    for strategy, path, trainable_override in eval_list:
        if not os.path.exists(path):
            print(f"[WARN] not found: {path}, skipping {strategy}"); continue
        print(f"\n[INFO] evaluating {strategy} ...")
        rows, meta = evaluate_checkpoint(path, files, text_emb, cfg, device, strategy_name=strategy)
        ps = np.array([r[1] for r in rows]); ss = np.array([r[2] for r in rows])
        trainable = trainable_override if trainable_override is not None else meta.get("trainable_params")
        pct = (100*trainable/total_params) if trainable else 0.0
        summary_rows.append((strategy, trainable, pct, ps.mean(), ps.std(), ss.mean(), ss.std()))
        for fname, p, s in rows:
            persample_rows.append((strategy, fname, p, s))
        print(f"    trainable={trainable}  PSNR {ps.mean():.3f}±{ps.std():.3f}  SSIM {ss.mean():.4f}±{ss.std():.4f}")

    # Baseline: provided LR panel directly against GT (no model), same held-out set.
    base_p, base_s = [], []
    eval_files = [f for f in files if f != cfg["finetune_sample_name"]]
    if cfg.get("max_eval_images") is not None:
        eval_files = eval_files[:cfg["max_eval_images"]]
    for fname in eval_files:
        path = os.path.join(cfg["data_dir"], fname)
        img_lr = load_panel(path, cfg["crop_panels"], cfg["panel_lr"])
        img_hr = load_panel(path, cfg["crop_panels"], cfg["panel_hr"])
        p, s = compute_metrics(img_lr, img_hr)
        base_p.append(p); base_s.append(s)
        persample_rows.append(("baseline", fname, p, s))
    summary_rows.insert(0, ("baseline", 0, 0.0,
                            np.mean(base_p), np.std(base_p), np.mean(base_s), np.std(base_s)))

    sp = os.path.join(cfg["output_dir"], "curve_summary.csv")
    with open(sp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy","trainable_params","trainable_pct","psnr_mean","psnr_std","ssim_mean","ssim_std"])
        for r in summary_rows:
            w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.4f}", f"{r[4]:.4f}", f"{r[5]:.5f}", f"{r[6]:.5f}"])

    pp = os.path.join(cfg["output_dir"], "curve_persample.csv")
    with open(pp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy","filename","psnr","ssim"])
        for r in persample_rows: w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.5f}"])

    print("\n" + "="*64)
    print("SUMMARY (by trainable parameter count)")
    print("="*64)
    print(f"{'strategy':12}{'trainable':>14}{'%':>10}{'PSNR':>10}{'SSIM':>10}")
    for r in summary_rows:
        print(f"{r[0]:12}{str(r[1]):>14}{r[2]:>9.4f}%{r[3]:>10.3f}{r[5]:>10.4f}")
    print(f"\n[INFO] saved: {sp}")
    print(f"[INFO] saved: {pp}")


if __name__ == "__main__":
    main()
