"""
Learning-rate sensitivity sweep for full fine-tuning (the "all" tier).

Purpose: address the concern that the poor performance of full fine-tuning might
be due only to a poorly chosen learning rate. All parameters are unfrozen; the
only variable is the learning rate. Training and evaluation follow exactly the
same protocol as finetune_progressive.py and evaluate_progressive.py. For each
learning rate, the model is retrained from the original pretrained weights, saved,
and evaluated on the same held-out set. Output: lr_sweep_all.csv.

Interpretation: if "all" stays below out_only (spleen 15.057 / 0.442) at every
learning rate, the "over-tuning is harmful" effect cannot be attributed to a
learning-rate mismatch. This supports the argument in Section 4.2 of the paper.

A fixed random seed is set so that the sweep is reproducible.

Requires the official FluoResFM implementation on the PYTHONPATH (models.unet_sd_c
and utils.optim). See README.

Usage:
    python lr_sweep_all.py
Edit the paths in CONFIG.
"""
import os, csv, json, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import _MODEL_CONFIGS

from models.unet_sd_c import UNetModel
import utils.optim as utils_optim

CONFIG = {
    "input":    r"./data/spleen/Spleen_10x fused 2.png",
    "data_dir": r"./data/spleen",

    "crop_panels": 3, "panel_lr": 2, "panel_hr": 1,

    "checkpoint":      r"./checkpoints/fluoresfm/pretrained.pt",
    "biomedclip_json": r"./checkpoints/clip/biomedclip/open_clip_config.json",
    "biomedclip_bin":  r"./checkpoints/clip/biomedclip/open_clip_pytorch_model.bin",

    "output_dir": r"./results_progressive",
    "ckpt_dir":   r"./checkpoints/fluoresfm/lr_sweep_all",

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

    # The only variable: learning rates spanning an order of magnitude above and
    # below the official 1e-5.
    "lr_list": [1e-4, 3e-5, 1e-5, 3e-6, 1e-6],

    # Remaining training hyperparameters, identical to finetune_progressive.py.
    "sf_lr": 2, "patch_size": 64, "n_patches": 1000,
    "batch_size": 8, "num_epochs": 50,

    # Evaluation protocol, identical to evaluate_progressive.py.
    "overlap": 16,
    "finetune_sample_name": "Spleen_10x fused 2.png",
    "max_eval_images": None,    # None = full held-out set (24); small int for a quick preview
    "eval_batch": 8,

    "seed": 42,
    "device": "cuda:0", "amp": True,
}


# data
def load_panel(path, crop_panels, panel_idx):
    img = Image.open(path); W, H = img.size
    pw = W // crop_panels; left = pw * panel_idx
    img = img.crop((left, 0, left+pw, H))
    return np.array(img.convert("L")).astype(np.float32)

def normalize_percentile(img, p_low=0.03, p_high=0.995):
    lo = np.percentile(img, p_low*100); hi = np.percentile(img, p_high*100)
    return np.clip((img-lo)/(hi-lo+1e-8), 0, None)

def interp_sf(img, sf):
    if sf == 1: return img
    t = torch.from_numpy(img[None, None])
    return F.interpolate(t, scale_factor=sf, mode="nearest")[0, 0].numpy()

def downsample_sf(img, sf):
    if sf == 1: return img
    t = torch.from_numpy(img[None, None])
    return F.avg_pool2d(t, kernel_size=sf, stride=sf)[0, 0].numpy()

def compute_metrics(pred, gt):
    pred_n, gt_n = normalize_percentile(pred), normalize_percentile(gt)
    return psnr_fn(gt_n, pred_n, data_range=1.0), ssim_fn(gt_n, pred_n, data_range=1.0)

def make_patches(img_lr, img_hr, patch_size, n_patches):
    H, W = img_lr.shape
    lrs, hrs = [], []
    for _ in range(n_patches):
        y = np.random.randint(0, H - patch_size)
        x = np.random.randint(0, W - patch_size)
        lrs.append(img_lr[y:y+patch_size, x:x+patch_size])
        hrs.append(img_hr[y:y+patch_size, x:x+patch_size])
    return (torch.from_numpy(np.array(lrs)[:, None]).float(),
            torch.from_numpy(np.array(hrs)[:, None]).float())


# patch stitcher
class Patch_stitcher:
    def __init__(self, patch_size=64, overlap=0, padding_mode="reflect"):
        self.ps, self.ol, self.padding_mode = patch_size, overlap, padding_mode
        self._gen()
    def _gen(self):
        ps, ol = self.ps, self.ol
        self.m_lu=np.pad(np.ones((1,1,ps-ol,ps-ol)),((0,0),(0,0),(0,ol+1),(0,ol+1)),"linear_ramp")[...,0:-1,0:-1]
        self.m_mu=np.pad(np.ones((1,1,ps-ol,ps-2*ol)),((0,0),(0,0),(0,ol+1),(ol+1,ol+1)),"linear_ramp")[...,0:-1,1:-1]
        self.m_ru=np.pad(np.ones((1,1,ps-ol,ps-ol)),((0,0),(0,0),(0,ol+1),(ol+1,0)),"linear_ramp")[...,0:-1,1:]
        self.m_lm=np.pad(np.ones((1,1,ps-2*ol,ps-ol)),((0,0),(0,0),(ol+1,ol+1),(0,ol+1)),"linear_ramp")[...,1:-1,0:-1]
        self.m_mm=np.pad(np.ones((1,1,ps-2*ol,ps-2*ol)),((0,0),(0,0),(ol+1,ol+1),(ol+1,ol+1)),"linear_ramp")[...,1:-1,1:-1]
        self.m_rm=np.pad(np.ones((1,1,ps-2*ol,ps-ol)),((0,0),(0,0),(ol+1,ol+1),(ol+1,0)),"linear_ramp")[...,1:-1,1:]
        self.m_lb=np.pad(np.ones((1,1,ps-ol,ps-ol)),((0,0),(0,0),(ol+1,0),(0,ol+1)),"linear_ramp")[...,1:,0:-1]
        self.m_mb=np.pad(np.ones((1,1,ps-ol,ps-2*ol)),((0,0),(0,0),(ol+1,0),(ol+1,ol+1)),"linear_ramp")[...,1:,1:-1]
        self.m_rb=np.pad(np.ones((1,1,ps-ol,ps-ol)),((0,0),(0,0),(ol+1,0),(ol+1,0)),"linear_ramp")[...,1:,1:]
        self.m_lu01=np.pad(np.ones((1,1,ps-ol,ps)),((0,0),(0,0),(0,ol+1),(0,0)),"linear_ramp")[...,0:-1,:]
        self.m_lm01=np.pad(np.ones((1,1,ps-2*ol,ps)),((0,0),(0,0),(ol+1,ol+1),(0,0)),"linear_ramp")[...,1:-1,:]
        self.m_lb01=np.pad(np.ones((1,1,ps-ol,ps)),((0,0),(0,0),(ol+1,0),(0,0)),"linear_ramp")[...,1:,:]
        self.m_lu10=np.pad(np.ones((1,1,ps,ps-ol)),((0,0),(0,0),(0,0),(0,ol+1)),"linear_ramp")[...,0:-1]
        self.m_mu10=np.pad(np.ones((1,1,ps,ps-2*ol)),((0,0),(0,0),(0,0),(ol+1,ol+1)),"linear_ramp")[...,1:-1]
        self.m_ru10=np.pad(np.ones((1,1,ps,ps-ol)),((0,0),(0,0),(0,0),(ol+1,0)),"linear_ramp")[...,1:]
        self.m_lu11=np.ones((1,1,ps,ps))
    def unfold(self, img):
        Ny,Nx=img.shape[-2],img.shape[-1]; step=self.ps-self.ol
        npy=math.ceil((Ny-self.ps)/step)+1; npx=math.ceil((Nx-self.ps)/step)+1
        Ny_pad,Nx_pad=npy*step+self.ol,npx*step+self.ol
        img_pad=F.pad(img,(0,Nx_pad-Nx,0,Ny_pad-Ny),mode=self.padding_mode)
        patches=torch.zeros((npy*npx,img.shape[1],self.ps,self.ps),device=img_pad.device,dtype=img_pad.dtype)
        for i in range(npy):
            for j in range(npx):
                patches[i*npx+j]=img_pad[0,:,i*step:i*step+self.ps,j*step:j*step+self.ps]
        return patches
    def fold(self, patches, shape):
        if isinstance(patches,torch.Tensor): patches=patches.cpu().numpy()
        bs,nc,Ny,Nx=shape; step=self.ps-self.ol
        npy=math.ceil((Ny-self.ps)/step)+1; npx=math.ceil((Nx-self.ps)/step)+1
        patches=patches.reshape(npy*npx,bs,nc,self.ps,self.ps)
        pad=np.zeros((bs,nc,npy*step+self.ol,npx*step+self.ol))
        mlu=self.m_lu11 if(npx==1 and npy==1) else self.m_lu01 if npx==1 else self.m_lu10 if npy==1 else self.m_lu
        mmu=self.m_mu10 if npy==1 else self.m_mu; mru=self.m_ru10 if npy==1 else self.m_ru
        mlm=self.m_lm01 if npx==1 else self.m_lm; mlb=self.m_lb01 if npx==1 else self.m_lb
        masks=[[mlu,mmu,mru],[mlm,self.m_mm,self.m_rm],[mlb,self.m_mb,self.m_rb]]
        for i in range(npy):
            ri=0 if i==0 else(2 if i==npy-1 else 1)
            for j in range(npx):
                ci=0 if j==0 else(2 if j==npx-1 else 1)
                p=patches[i*npx+j].copy()
                if self.ol>0: p*=masks[ri][ci]
                pad[:,:,i*step:i*step+self.ps,j*step:j*step+self.ps]+=p
        return pad[...,:Ny,:Nx]


def build_model(device):
    return UNetModel(in_channels=1,out_channels=1,channels=320,n_res_blocks=1,
        attention_levels=[0,1,2,3],channel_multipliers=[1,2,4,4],
        n_heads=8,tf_layers=1,d_cond=768,pixel_shuffle=False,scale_factor=4).to(device)

def load_pretrained(model, ckpt_path, device):
    ckpt=torch.load(ckpt_path,map_location=device,weights_only=True)
    sd=utils_optim.on_load_checkpoint(ckpt["model_state_dict"])
    model.load_state_dict(sd); return model

def load_embedder(cfg, device):
    with open(cfg["biomedclip_json"]) as f: config=json.load(f)
    name="biomedclip_local"
    if name not in _MODEL_CONFIGS: _MODEL_CONFIGS[name]=config["model_cfg"]
    tok=get_tokenizer(name)
    clip_model,_,_=create_model_and_transforms(model_name=name,pretrained=cfg["biomedclip_bin"],
        **{f"image_{k}":v for k,v in config["preprocess_cfg"].items()})
    clip_model.to(device); tm=clip_model.text; tm.output_tokens=True; tm.eval()
    for p in tm.parameters(): p.requires_grad=False
    return tok, tm

@torch.no_grad()
def encode_prompt(prompt, tok, tm, device, cl=160):
    tokens=tok([prompt],context_length=cl).to(device)
    _,feats=tm(tokens); return feats


def train_all_at_lr(lr, cfg, device, lr_patches, hr_patches, text_emb):
    """Unfreeze all parameters, train from pretrained weights at the given lr, save."""
    print(f"\n{'='*60}\n  training all @ lr={lr:.0e}\n{'='*60}")
    model = load_pretrained(build_model(device), cfg["checkpoint"], device)
    for p in model.parameters():
        p.requires_grad = True   # all: everything unfrozen
    model.train()

    loader = DataLoader(TensorDataset(lr_patches, hr_patches),
                        batch_size=cfg["batch_size"], shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.GradScaler("cuda", enabled=cfg["amp"])

    for epoch in tqdm(range(cfg["num_epochs"]), desc=f"[all lr={lr:.0e}]"):
        ep_loss, nb = 0.0, 0
        for lr_b, hr_b in loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            bs = lr_b.shape[0]
            t = torch.zeros(bs, device=device); cond = text_emb.expand(bs, -1, -1)
            with torch.autocast("cuda", torch.float16, enabled=(cfg["amp"] and device.type=="cuda")):
                pred = model(lr_b, t, cond); loss = F.l1_loss(pred, hr_b)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ep_loss += loss.item(); nb += 1

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    out_path = os.path.join(cfg["ckpt_dir"], f"all_lr{lr:.0e}.pt")
    torch.save({"model_state_dict": model.state_dict(), "lr": lr,
                "final_loss": ep_loss/nb}, out_path)
    print(f"[INFO] saved {out_path}  (final loss {ep_loss/nb:.5f})")
    del model, opt; torch.cuda.empty_cache()
    return out_path


@torch.no_grad()
def infer_one(model, img_lr_up, text_emb, cfg, device):
    H,W=img_lr_up.shape
    st=Patch_stitcher(cfg["patch_size"],cfg["overlap"],"reflect")
    patches=st.unfold(torch.from_numpy(img_lr_up[None,None]).to(device))
    N=patches.shape[0]; bs=min(cfg["eval_batch"],N); out=torch.zeros_like(patches)
    for i in range(0,N,bs):
        b=patches[i:i+bs].to(device); t=torch.zeros(b.shape[0],device=device)
        cond=text_emb.expand(b.shape[0],-1,-1)
        with torch.autocast("cuda",torch.float16,enabled=(cfg["amp"] and device.type=="cuda")):
            pred=model(b,t,cond)
        out[i:i+bs]=pred.cpu()
    step=cfg["patch_size"]-cfg["overlap"]
    npy=math.ceil((H-cfg["patch_size"])/step)+1; npx=math.ceil((W-cfg["patch_size"])/step)+1
    r=st.fold(out,(1,1,npy*step+cfg["overlap"],npx*step+cfg["overlap"]))
    return downsample_sf(r[0,0,:H,:W].astype(np.float32),cfg["sf_lr"])


def evaluate_ckpt(ckpt_path, eval_files, text_emb, cfg, device):
    model=load_pretrained(build_model(device), cfg["checkpoint"], device)  # placeholder structure
    ckpt=torch.load(ckpt_path,map_location=device,weights_only=True)
    model.load_state_dict(utils_optim.on_load_checkpoint(ckpt["model_state_dict"]))
    model.eval()
    ps,ss=[],[]
    for j,fname in enumerate(eval_files):
        path=os.path.join(cfg["data_dir"],fname)
        il=load_panel(path,cfg["crop_panels"],cfg["panel_lr"])
        ih=load_panel(path,cfg["crop_panels"],cfg["panel_hr"])
        il_up=interp_sf(normalize_percentile(il),cfg["sf_lr"])
        r=infer_one(model,il_up,text_emb,cfg,device)
        p,s=compute_metrics(r,ih); ps.append(p); ss.append(s)
        print(f"    img {j+1}/{len(eval_files)}  PSNR {p:5.2f}  SSIM {s:.4f}", end="\r")
    print()
    del model; torch.cuda.empty_cache()
    return float(np.mean(ps)), float(np.std(ps)), float(np.mean(ss)), float(np.std(ss))


def main():
    cfg=CONFIG; device=torch.device(cfg["device"])
    np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])

    # Data prep (identical to finetune_progressive; all lrs share the same patches)
    il=load_panel(cfg["input"],cfg["crop_panels"],cfg["panel_lr"])
    ih=load_panel(cfg["input"],cfg["crop_panels"],cfg["panel_hr"])
    il_up=interp_sf(normalize_percentile(il),cfg["sf_lr"])
    ih_up=interp_sf(normalize_percentile(ih),cfg["sf_lr"])
    lr_patches,hr_patches=make_patches(il_up,ih_up,cfg["patch_size"],cfg["n_patches"])

    tok,tm=load_embedder(cfg,device); text_emb=encode_prompt(cfg["prompt"],tok,tm,device)
    del tok,tm; torch.cuda.empty_cache()

    files=sorted([f for f in os.listdir(cfg["data_dir"]) if f.endswith(".png")])
    eval_files=[f for f in files if f!=cfg["finetune_sample_name"]]
    if cfg["max_eval_images"] is not None:
        eval_files=eval_files[:cfg["max_eval_images"]]
    print(f"[INFO] {len(eval_files)} held-out images for evaluation")

    rows=[]
    for lr in cfg["lr_list"]:
        ckpt=train_all_at_lr(lr, cfg, device, lr_patches, hr_patches, text_emb)
        pm,psd,sm,ssd=evaluate_ckpt(ckpt, eval_files, text_emb, cfg, device)
        rows.append((lr,pm,psd,sm,ssd))
        print(f"[RESULT] all lr={lr:.0e}  PSNR {pm:.3f}±{psd:.3f}  SSIM {sm:.4f}±{ssd:.4f}")
        # overwrite the csv after each lr so a mid-run interruption keeps results
        out=os.path.join(cfg["output_dir"],"lr_sweep_all.csv")
        os.makedirs(cfg["output_dir"], exist_ok=True)
        with open(out,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["lr","psnr_mean","psnr_std","ssim_mean","ssim_std"])
            for r in rows: w.writerow([f"{r[0]:.0e}",f"{r[1]:.4f}",f"{r[2]:.4f}",f"{r[3]:.5f}",f"{r[4]:.5f}"])

    print("\n"+"="*60)
    print("ALL learning-rate sweep summary (compare out_only = 15.057 / 0.442)")
    print("="*60)
    print(f"{'lr':>8}{'PSNR':>10}{'SSIM':>10}")
    for r in rows:
        print(f"{r[0]:>8.0e}{r[1]:>10.3f}{r[3]:>10.4f}")
    print(f"\n[INFO] saved: {os.path.join(cfg['output_dir'],'lr_sweep_all.csv')}")


if __name__=="__main__":
    main()
