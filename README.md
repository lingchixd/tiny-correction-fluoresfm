# A Tiny Correction in the Right Place

Code for the paper **"A Tiny Correction in the Right Place: Adapting a Fluorescence Super-Resolution Foundation Model"** (MICCAI 2026, The 2nd Workshop on Efficient Medical AI).

We study parameter-efficient adaptation of the FluoResFM fluorescence super-resolution foundation model to fiber optic microscopy with ultraviolet surface excitation (FUSE), an imaging condition outside its pretraining distribution. The central finding is that tuning only the output side on a single image (about 3.5K parameters, roughly 0.0005% of the model) reverses a negative-transfer effect and surpasses the low-resolution baseline, while freeing more parameters degrades performance under single-image overfitting.


## Repository contents

| File | Purpose | Paper reference |
|------|---------|-----------------|
| `finetune_progressive.py` | Progressive-unfreezing fine-tuning (out / in / in+out / outer2 / outer4 / all), one checkpoint per tier | Sec. 3.4, Table 1 |
| `evaluate_progressive.py` | Batch evaluation on the held-out set; PSNR/SSIM; per-sample and summary CSVs | Sec. 3.6, Table 1, Fig. 1 |
| `baseline_bicubic.py` | Provided-LR and bicubic-roundtrip baselines (CPU only) | Table 1 (baseline, bicubic rows) |
| `lr_sweep_all.py` | Learning-rate sensitivity sweep for full fine-tuning | Sec. 4.2 |
| `holm_correction.py` | Holm multiple-comparison correction on the paired Wilcoxon tests | Sec. 3.6, Table 1 |


## Dependencies

This code builds on the official **FluoResFM** implementation. The training and
evaluation scripts import `models.unet_sd_c` and `utils.optim` from FluoResFM, so
the official repository must be available on the `PYTHONPATH`. Please obtain it
from the original FluoResFM release; it is not redistributed here.

Python packages:

```bash
conda create -n fluoresfm python=3.10
conda activate fluoresfm
pip install torch torchvision numpy pillow scikit-image scipy open_clip_torch tqdm
```

A `requirements.txt` with pinned versions is also provided.


## Pretrained weights

This work uses the official FluoResFM pretrained weights without architectural
change. Obtain them from the FluoResFM release and place the checkpoint at
`./checkpoints/fluoresfm/pretrained.pt`. The text encoder uses BiomedCLIP; place
its config and weights under `./checkpoints/clip/biomedclip/`.


## Data

The FUSE images of mouse spleen and kidney were acquired in-house. All animal
procedures were approved by the relevant institutional ethics committee. The raw
imaging data are **not publicly released**; they are available from the authors
upon reasonable request.

Each sample is a three-panel PNG of the same field. For this 2x task the middle
panel is the high-resolution reference and the right panel is the low-resolution
input (the leftmost panel is a higher-magnification acquisition not used here).
For each tissue, a single image is reserved for fine-tuning and 24 images are held
out for evaluation, disjoint from the fine-tuning image at the image level.

Expected layout:

```
data/
  spleen/     # three-panel PNGs
  kidney/
checkpoints/
  fluoresfm/
    pretrained.pt
    progressive/        # produced by finetune_progressive.py
  clip/biomedclip/
```


## Reproducing the results

Shared hyperparameters (identical across all tiers; the only variable is the
unfrozen range):

- Optimizer: AdamW, L1 loss
- Learning rate: constant `1e-5`
- Epochs: `50`
- Fixed random seed (`42`), set before patch sampling, so training is reproducible
- Patch-based inference: patch size 64, overlap 16, linear-ramp blending
- Evaluation: PSNR/SSIM after percentile normalization (0.03 / 0.995), paired
  Wilcoxon signed-rank test with Holm correction

```bash
# 1. Fine-tune each tier on a single image (edit CONFIG: tissue = spleen or kidney)
python finetune_progressive.py

# 2. Evaluate all tiers on the held-out set (writes curve_summary.csv, curve_persample.csv)
python evaluate_progressive.py

# 3. Baselines (provided-LR and bicubic; CPU only)
python baseline_bicubic.py

# 4. Multiple-comparison correction on the paired tests
python holm_correction.py

# 5. Learning-rate sensitivity for full fine-tuning (Sec. 4.2)
python lr_sweep_all.py
```

Set the paths in each script's `CONFIG` block to your local data and checkpoints.
For spleen, `finetune_progressive.py` reproduces the checkpoints behind Table 1
and Fig. 1; switch `tissue` to `kidney` for the cross-tissue results (Sec. 4.6).



## License

Released under the MIT License. See `LICENSE`. Note that the official FluoResFM
code, on which these scripts depend, is subject to its own license.
