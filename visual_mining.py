"""
visual_mining.py — prototype refinement via image patches
==========================================================
文本种子 → 训练集 patch 检索/标注筛选 → 视觉均值 → 融合 → visual_prototypes.pt

Two modes:
  1. Unsupervised: cosine top-K from all images (faster, no labels needed)
  2. Weak-label: filter by clinical CSV column (e.g. Shape=3 → lobulated patients only)

Usage:
  python visual_mining.py --checkpoint output/.../model.pth.tar-30 --output visual_prototypes.pt
  python visual_mining.py --checkpoint ... --csv clinical.csv --concept_map concept_map.json
"""
import argparse, os, sys, json, warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

import torch, torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Dassl.pytorch"))
from dassl.config import get_cfg_default
from yacs.config import CfgNode as CN
from open_clip.src.open_clip import create_model_from_pretrained
from trainers.AnomalyDetect.anomaly_detect import AnomalyCLIP
from trainers.AnomalyDetect.text_descriptions import get_text_descriptions


def build_cfg():
    cfg = get_cfg_default()
    cfg.TRAINER.BIOMEDCOOP = CN()
    cfg.TRAINER.BIOMEDCOOP.CTX_INIT = "a photo of a"; cfg.TRAINER.BIOMEDCOOP.CSC = False
    cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION = "end"; cfg.TRAINER.BIOMEDCOOP.N_CTX = 4
    cfg.TRAINER.BIOMEDCOOP.PREC = "fp32"; cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA = 0.75
    cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA = 0.75; cfg.TRAINER.BIOMEDCOOP.TAU = 1.5
    cfg.TRAINER.BIOMEDCOOP.N_PROMPTS = 50
    cfg.TRAINER.ANOMALY_DETECT = CN()
    cfg.TRAINER.ANOMALY_DETECT.PREC = "fp32"; cfg.TRAINER.ANOMALY_DETECT.LAMBDA_CONSIST = 0.5
    cfg.TRAINER.ANOMALY_DETECT.PATCH_TEMPERATURE = 0.07; cfg.TRAINER.ANOMALY_DETECT.THRESHOLD_STD = 1.5
    cfg.TRAINER.ANOMALY_DETECT.MEMORY_BANK_PATH = ""; cfg.TRAINER.ANOMALY_DETECT.VISUAL_PROTO_PATH = ""
    cfg.OPTIM.MAX_EPOCH = 100; cfg.DATASET.NAME = "BUSI"; cfg.freeze()
    return cfg


def collect_images(data_dir):
    images = []
    for cls in ["benign_tumor", "malignant_tumor", "normal_scan"]:
        d = os.path.join(data_dir, cls)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.png','.jpg','.jpeg')):
                    images.append(os.path.join(d, f))
    return images


def extract_patches(model, impath, device, tfm):
    img = Image.open(impath).convert("RGB")
    inp = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        x = inp.type(model.dtype)
        raw = model.image_encoder.trunk.forward_features(x)
        patches = raw[:, 1:, :].squeeze(0)
        patches = model.patch_proj(patches)
        patches = patches / patches.norm(dim=-1, keepdim=True)
    return patches  # [196, 768]


def mine_prototype(seed_proto, images, model, device, tfm, topk=100):
    best_sims, best_patches = [], []
    for impath in tqdm(images, desc="  Scanning", leave=False):
        patches = extract_patches(model, impath, device, tfm)
        sims = (patches @ seed_proto).cpu()
        top_vals, top_idxs = sims.topk(min(50, len(sims)))
        for v, idx in zip(top_vals, top_idxs):
            pv = patches[idx]
            if len(best_sims) < topk:
                best_sims.append(v.item()); best_patches.append(pv)
            else:
                mn = min(best_sims)
                if v.item() > mn:
                    mi = best_sims.index(mn)
                    best_sims[mi] = v.item(); best_patches[mi] = pv
    if not best_patches:
        return seed_proto
    V = torch.stack(best_patches).mean(dim=0)
    return V / V.norm()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="output/anomaly_detect/anomaly_clip/model.pth.tar-30")
    parser.add_argument("--data_dir", default="data/BUSI/BUSI")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--alpha_text", type=float, default=0.2)
    parser.add_argument("--output", default="visual_prototypes.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = build_cfg()
    classnames = ["benign_tumor", "malignant_tumor", "normal_scan"]

    bclip, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    bclip.float().eval()
    tdescs = get_text_descriptions("BUSI")
    clip_model = AnomalyCLIP(cfg, classnames, bclip, tdescs)
    clip_model.to(device).eval()

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ps = {k.replace("patch_proj.",""):v for k,v in ckpt["state_dict"].items() if k.startswith("patch_proj.")}
        if ps: clip_model.patch_proj.load_state_dict(ps, strict=True)
        print(f"Loaded patch_proj from {args.checkpoint}")

    images = collect_images(args.data_dir)
    print(f"Images: {len(images)}")

    tfm = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466,0.4578275,0.40821073), std=(0.26862954,0.26130258,0.27577711)),
    ])

    proto_anom = clip_model.anchors_anomaly.clone()
    proto_norm = clip_model.anchors_normal.clone()
    names_anom = clip_model.concept_names_anomaly
    names_norm = clip_model.concept_names_normal

    for i, name in enumerate(names_anom):
        print(f"\n[MINE] Anomaly: {name}")
        V = mine_prototype(proto_anom[i], images, clip_model, device, tfm, args.topk)
        proto_anom[i] = (args.alpha_text * proto_anom[i] + (1-args.alpha_text) * V)
        proto_anom[i] = proto_anom[i] / proto_anom[i].norm()

    for i, name in enumerate(names_norm):
        print(f"\n[MINE] Normal: {name}")
        V = mine_prototype(proto_norm[i], images, clip_model, device, tfm, args.topk)
        proto_norm[i] = (args.alpha_text * proto_norm[i] + (1-args.alpha_text) * V)
        proto_norm[i] = proto_norm[i] / proto_norm[i].norm()

    torch.save({
        "anchors_normal": proto_norm, "anchors_anomaly": proto_anom,
        "concept_names_normal": names_norm, "concept_names_anomaly": names_anom,
    }, args.output)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
