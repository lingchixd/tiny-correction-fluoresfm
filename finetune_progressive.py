"""
FluoResFM progressive-unfreezing fine-tuning.

Fine-tunes FluoResFM on a single image under a series of unfreezing tiers, from
tuning only the output side (~3.5K parameters) up to full fine-tuning. All tiers
share the same hyperparameters; the only variable is the unfrozen range, so they
are directly comparable. Each tier starts from the same pretrained weights and is
saved as a separate checkpoint.

Tiers (symmetric outside-in unfreezing):
  out_only : output convolution only                         (~3.5K,  ~0.0005%)
  in_only  : input convolution block only                    (~3.2K,  ~0.0005%)
  in_out   : input block + output convolution                (~6.7K,  ~0.001%)
  outer2   : outermost 2 input + 2 output blocks + out conv
  outer4   : outermost 4 input + 4 output blocks + out conv
  all      : full fine-tuning                                 (100%)

A fixed random seed is set so that patch sampling and training are reproducible.

Requires the official FluoResFM implementation on the PYTHONPATH (for
models.unet_sd_c and utils.optim). See README.

Usage:
    python finetune_progressive.py
Edit CONFIG to select the tissue and point to your local data and checkpoints.
"""

import json, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import _MODEL_CONFIGS

from models.unet_sd_c import UNetModel
import utils.optim as utils_optim

# ==============================================================================
# Prompts per tissue. The only difference is the "sample" field.
# ==============================================================================
PROMPTS = {
    "spleen": (
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
    "kidney": (
        "Task: super-resolution with a scale factor of 2; "
        "sample: fixed kidney tissue; structure: nuclei; "
        "fluorescence indicator: Hoechst; "
        "input microscope: wide-field microscope with excitation numerical aperture (NA) of 0.1, "
        "detection numerical aperture (NA) of 0.1; "
        "input pixel size: 1850 x 1850 nm. Nearest interpolation with a factor of 2; "
        "target microscope: wide-field microscope with excitation numerical aperture (NA) of 0.25, "
        "detection numerical aperture (NA) of 0.25; "
        "target pixel size: 925 x 925 nm."
    ),
}

# ==============================================================================
CONFIG = {
    # Tissue: "spleen" or "kidney". Selects the prompt and the output filename.
    "tissue": "spleen",

    # The single image used for fine-tuning (three-panel PNG).
    "input": r"./data/spleen/Spleen_10x fused 2.png",

    "crop_panels": 3,
    "panel_lr":    2,   # right panel  = low-resolution input
    "panel_hr":    1,   # middle panel = high-resolution reference (target)

    # Start from the original pretrained weights so every tier is a clean
    # pretrained -> single-image fine-tuning chain.
    "checkpoint":      r"./checkpoints/fluoresfm/pretrained.pt",
    "biomedclip_json": r"./checkpoints/clip/biomedclip/open_clip_config.json",
    "biomedclip_bin":  r"./checkpoints/clip/biomedclip/open_clip_pytorch_model.bin",

    # Checkpoints are saved here, one per tier, named <tissue>_<tier>.pt
    "output_dir": r"./checkpoints/fluoresfm/progressive",

    "sf_lr":        2,
    "patch_size":   64,
    "n_patches":    1000,
    "batch_size":   8,
    "num_epochs":   50,
    "lr":           1e-5,
    "lr_decay_every": 10000,
    "lr_decay_rate":  0.5,
    "lr_min":         1e-7,

    # Tiers to run, in order. "all" is heaviest and placed last.
    "strategies": ["out_only", "in_only", "in_out", "outer2", "outer4", "all"],

    "seed":   42,
    "device": "cuda:0",
    "amp":    True,
    "log_every": 10,
}
# ==============================================================================


def load_panel(path, crop_panels, panel_idx):
    img = Image.open(path)
    W, H = img.size
    panel_w = W // crop_panels
    left    = panel_w * panel_idx
    img     = img.crop((left, 0, left+panel_w, H))
    return np.array(img.convert("L")).astype(np.float32)


def normalize(img, p_low=0.03, p_high=0.995):
    lo = np.percentile(img, p_low * 100)
    hi = np.percentile(img, p_high * 100)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, None), lo, hi


def interp_sf(img, sf):
    if sf == 1:
        return img
    t = torch.from_numpy(img[None, None])
    return F.interpolate(t, scale_factor=sf, mode="nearest")[0, 0].numpy()


def make_patches(img_lr, img_hr, patch_size, n_patches):
    H, W = img_lr.shape
    assert img_lr.shape == img_hr.shape
    lrs, hrs = [], []
    for _ in range(n_patches):
        y = np.random.randint(0, H - patch_size)
        x = np.random.randint(0, W - patch_size)
        lrs.append(img_lr[y:y+patch_size, x:x+patch_size])
        hrs.append(img_hr[y:y+patch_size, x:x+patch_size])
    lr_tensor = torch.from_numpy(np.array(lrs)[:, None]).float()
    hr_tensor = torch.from_numpy(np.array(hrs)[:, None]).float()
    return lr_tensor, hr_tensor


def load_model(checkpoint, device):
    model = UNetModel(
        in_channels=1, out_channels=1, channels=320, n_res_blocks=1,
        attention_levels=[0,1,2,3], channel_multipliers=[1,2,4,4],
        n_heads=8, tf_layers=1, d_cond=768, pixel_shuffle=False, scale_factor=4,
    ).to(device)
    ckpt       = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = utils_optim.on_load_checkpoint(ckpt["model_state_dict"])
    model.load_state_dict(state_dict)
    return model


def load_embedder(cfg, device):
    with open(cfg["biomedclip_json"]) as f:
        config = json.load(f)
    model_name = "biomedclip_local"
    if model_name not in _MODEL_CONFIGS:
        _MODEL_CONFIGS[model_name] = config["model_cfg"]
    tokenizer        = get_tokenizer(model_name)
    clip_model, _, _ = create_model_and_transforms(
        model_name=model_name, pretrained=cfg["biomedclip_bin"],
        **{f"image_{k}": v for k, v in config["preprocess_cfg"].items()},
    )
    clip_model.to(device)
    text_model = clip_model.text
    text_model.output_tokens = True
    text_model.eval()
    for p in text_model.parameters():
        p.requires_grad = False
    return tokenizer, text_model


@torch.no_grad()
def encode_prompt(prompt, tokenizer, text_model, device, context_length=160):
    tokens = tokenizer([prompt], context_length=context_length).to(device)
    _, feats = text_model(tokens)
    return feats


# ------------------------------------------------------------------------------
# Progressive unfreezing under a single rule: symmetrically unfreeze the
# outermost k input blocks and k output blocks, plus the output convolution.
# input_blocks are ordered outer(shallow) -> inner(deep) by increasing index;
# output_blocks are the reverse, so the largest index is the outermost.
# ------------------------------------------------------------------------------
def set_finetune_progressive(model, strategy):
    for p in model.parameters():
        p.requires_grad = False

    n_out = len(model.output_blocks)   # = 8

    def unfreeze(module):
        for p in module.parameters():
            p.requires_grad = True

    if strategy == "out_only":
        unfreeze(model.out)

    elif strategy == "in_only":
        unfreeze(model.input_blocks[0])

    elif strategy == "in_out":
        unfreeze(model.input_blocks[0])
        unfreeze(model.out)

    elif strategy == "outer2":
        for i in [0, 1]:
            unfreeze(model.input_blocks[i])
        for i in [n_out-1, n_out-2]:
            unfreeze(model.output_blocks[i])
        unfreeze(model.out)

    elif strategy == "outer4":
        for i in [0, 1, 2, 3]:
            unfreeze(model.input_blocks[i])
        for i in [n_out-1, n_out-2, n_out-3, n_out-4]:
            unfreeze(model.output_blocks[i])
        unfreeze(model.out)

    elif strategy == "all":
        for p in model.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"unknown strategy: {strategy}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[INFO] strategy='{strategy}'  trainable {trainable:,}/{total:,} "
          f"({100*trainable/total:.4f}%)")
    return trainable, total


def train_one_strategy(strategy, cfg, device, lr_patches, hr_patches, text_emb):
    """Reload pretrained weights, unfreeze per strategy, train, and save."""
    print("\n" + "="*70)
    print(f"  training tier: {strategy}")
    print("="*70)

    # Each tier restarts from the original pretrained weights (keeps the chain
    # clean and the tiers comparable).
    model = load_model(cfg["checkpoint"], device)
    trainable, total = set_finetune_progressive(model, strategy)
    model.train()

    dataset    = TensorDataset(lr_patches, hr_patches)
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"]
    )
    scaler = torch.GradScaler("cuda", enabled=cfg["amp"])

    i_iter    = 0
    best_loss = float("inf")
    for epoch in tqdm(range(cfg["num_epochs"]), desc=f"[{strategy}]"):
        epoch_loss, n_batches = 0.0, 0
        for lr_batch, hr_batch in dataloader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            bs       = lr_batch.shape[0]
            t        = torch.zeros(bs, device=device)
            cond     = text_emb.expand(bs, -1, -1)

            with torch.autocast("cuda", torch.float16,
                                enabled=(cfg["amp"] and device.type=="cuda")):
                pred = model(lr_batch, t, cond)
                loss = F.l1_loss(pred, hr_batch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            i_iter += 1
            if i_iter % cfg["lr_decay_every"] == 0:
                for g in optimizer.param_groups:
                    g["lr"] = max(g["lr"] * cfg["lr_decay_rate"], cfg["lr_min"])

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        if (epoch + 1) % cfg["log_every"] == 0:
            print(f"\n  [{strategy}] epoch {epoch+1}/{cfg['num_epochs']} "
                  f"loss {avg_loss:.5f} lr {optimizer.param_groups[0]['lr']:.2e}")
        best_loss = min(best_loss, avg_loss)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], f"{cfg['tissue']}_{strategy}.pt")
    torch.save({"model_state_dict": model.state_dict(),
                "strategy": strategy,
                "tissue": cfg["tissue"],
                "trainable_params": trainable,
                "total_params": total,
                "best_loss": best_loss}, out_path)
    print(f"[INFO] tier '{strategy}' saved to: {out_path}  (best_loss {best_loss:.5f})")

    del model, optimizer
    torch.cuda.empty_cache()
    return out_path, trainable, best_loss


def main():
    cfg    = CONFIG
    device = torch.device(cfg["device"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    prompt = PROMPTS[cfg["tissue"]]

    # Data is prepared once; all tiers share the same patches for comparability.
    print("[INFO] preparing data ...")
    img_lr_raw = load_panel(cfg["input"], cfg["crop_panels"], cfg["panel_lr"])
    img_hr_raw = load_panel(cfg["input"], cfg["crop_panels"], cfg["panel_hr"])
    img_lr_norm, _, _ = normalize(img_lr_raw)
    img_hr_norm, _, _ = normalize(img_hr_raw)
    img_lr_up = interp_sf(img_lr_norm, cfg["sf_lr"])
    img_hr_up = interp_sf(img_hr_norm, cfg["sf_lr"])
    lr_patches, hr_patches = make_patches(img_lr_up, img_hr_up,
                                          cfg["patch_size"], cfg["n_patches"])
    print(f"[INFO] {len(lr_patches)} patch pairs")

    print("[INFO] loading BiomedCLIP and encoding prompt ...")
    tokenizer, textmodel = load_embedder(cfg, device)
    text_emb = encode_prompt(prompt, tokenizer, textmodel, device)
    del tokenizer, textmodel
    torch.cuda.empty_cache()

    summary = []
    for strategy in cfg["strategies"]:
        try:
            path, trainable, best_loss = train_one_strategy(
                strategy, cfg, device, lr_patches, hr_patches, text_emb)
            summary.append((strategy, trainable, best_loss, path))
        except RuntimeError as e:
            # e.g. OOM on the "all" tier; earlier tiers are already saved.
            print(f"[WARN] tier '{strategy}' failed: {e}")
            summary.append((strategy, None, None, "FAILED"))
            torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("  all tiers done, summary")
    print("="*70)
    print(f"{'strategy':12} {'trainable':>14} {'best_loss':>12}")
    for s, tr, bl, p in summary:
        tr_s = f"{tr:,}" if tr is not None else "FAILED"
        bl_s = f"{bl:.5f}" if bl is not None else "-"
        print(f"{s:12} {tr_s:>14} {bl_s:>12}")
    print("\n[INFO] next, run evaluate_progressive.py to get PSNR/SSIM per checkpoint")


if __name__ == "__main__":
    main()
