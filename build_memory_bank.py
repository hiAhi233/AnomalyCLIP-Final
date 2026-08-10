"""
build_memory_bank.py — Normal Prototype Memory Bank
====================================================
从 BUSI normal_scan 图像构建正常组织 patch 特征记忆库 M.

原理:
  正常图 → BiomedCLIP ViT → 196 个 patch 特征(user)
  取所有正常 patch → M = [N_total, 512]
  异常: 1 - cos(patch_i, M_j) → 距离最近的正常 patch 越远越异常

Usage:
  python build_memory_bank.py                          # 默认: BUSI normal, save M
  python build_memory_bank.py --output memory_bank.pt  # 自定义输出路径
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

import torch, torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from open_clip.src.open_clip import create_model_from_pretrained
from dassl.config import get_cfg_default
from yacs.config import CfgNode as CN


def build_memory_bank(data_dir, device='cuda', output='memory_bank.pt',
                      organ='breast', max_patches=50000):
    """
    Build a normal patch feature memory bank.

    Args:
        data_dir: path to BUSI dataset root (e.g., data/BUSI/BUSI)
        device: 'cuda' or 'cpu'
        output: .pt file to save memory bank
        organ: organ name for metadata
        max_patches: max number of patches to keep (memory limit)
    """
    # load BiomedCLIP
    print('Loading BiomedCLIP...')
    model, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model.float().eval().to(device)

    # collect normal images
    norm_dir = os.path.join(data_dir, 'normal_scan')
    if not os.path.isdir(norm_dir):
        raise FileNotFoundError(f'Normal scan directory not found: {norm_dir}')
    img_files = [os.path.join(norm_dir, f) for f in os.listdir(norm_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f'Found {len(img_files)} normal images')

    tfm = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    all_patches = []
    for impath in tqdm(img_files, desc='Extracting normal patches'):
        img = Image.open(impath).convert('RGB')
        inp = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = model.visual.trunk.forward_features(inp.type(model.text.transformer.dtype))
            patches = raw[:, 1:, :].squeeze(0)       # [196, 768]
            patches = patches / patches.norm(dim=-1, keepdim=True)  # L2 norm
        all_patches.append(patches.cpu())

    M = torch.cat(all_patches, dim=0)                 # [total, 768]
    if M.shape[0] > max_patches:
        idx = torch.randperm(M.shape[0])[:max_patches]
        M = M[idx]
    print(f'Memory bank: {M.shape[0]} patches x {M.shape[1]} dims')

    torch.save({'M': M, 'organ': organ, 'dtype': 'float32'}, output)
    print(f'Saved → {output}')
    return M


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/BUSI/BUSI')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='memory_bank.pt')
    parser.add_argument('--max_patches', type=int, default=50000)
    args = parser.parse_args()
    build_memory_bank(args.data_dir, args.device, args.output, max_patches=args.max_patches)
