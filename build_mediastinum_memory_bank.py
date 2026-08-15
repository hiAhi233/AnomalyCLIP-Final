"""
build_mediastinum_memory_bank.py — 前纵隔占位"正常纵隔"记忆库 (无 ROI 方案)
================================================================
不再用 ROI 反选 (707 例没有 ROI)。改从"近乎正常"类别的患者建库:

  主源: hyperplasia (胸腺增生 25 人, 组织基本正常)
  辅源: benign_lesion (非肿瘤性病变/Castleman/纤维性肿瘤等 36 人)

这些扫描 95%+ 区域是正常纵隔/肺/胸壁组织, 全切片作为正常来源,
少量良性病灶 patch 的污染可接受 (s_img 低的 λ 回落机制兜底)。

特征与推理侧一致: BiomedCLIP trunk 原始 patch 特征 (768d, 归一化),
不再经过 patch_proj — 与已修复的 Path B 空间匹配。

Usage:
  python build_mediastinum_memory_bank.py
"""

import argparse, os, sys, warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")

import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image

from open_clip.src.open_clip import create_model_from_pretrained

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Mediastinum")
BANK_CLASSES = ["hyperplasia", "benign_lesion"]   # 近乎正常的类别
MAX_PATCHES_PER_PATIENT = 120
MAX_TOTAL_FEATURES = 50000
KEEP_EVERY = 4   # 每 4 个 patch 取 1 控制内存


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=DATA_DIR)
    ap.add_argument('--output', default='memory_bank_mediastinum.pt')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--max-features', type=int, default=MAX_TOTAL_FEATURES)
    args = ap.parse_args()

    print('Loading BiomedCLIP...')
    model, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.float().eval().to(args.device)

    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    # 收集良性组患者的所有切片
    slices = []   # (impath, pid)
    import re
    for cls in BANK_CLASSES:
        cls_dir = os.path.join(args.data_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f'  [SKIP] 类别目录不存在: {cls_dir}')
            continue
        for f in sorted(os.listdir(cls_dir)):
            if not f.endswith('.png'):
                continue
            m = re.match(r'P(\d+)_', f)
            pid = int(m.group(1)) if m else -1
            slices.append((os.path.join(cls_dir, f), pid))
    print(f'良性组切片: {len(slices)} 张')

    all_patches = []
    per_patient = {}

    for impath, pid in tqdm(slices, desc='Slices'):
        img = np.array(Image.open(impath).convert('RGB'))
        # 过滤纯黑背景切片 (肺尖/肩部层面无组织)
        if img.mean() < 3:
            continue
        inp = tfm(Image.fromarray(img))
        with torch.no_grad():
            raw = model.visual.trunk.forward_features(
                inp.unsqueeze(0).to(args.device).type(model.text.transformer.dtype))
            feats = raw[:, 1:, :].squeeze(0)             # [196, 768] 原始 patch 特征
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats[::KEEP_EVERY].cpu()                # 49 个
        all_patches.append(feats)
        per_patient[pid] = per_patient.get(pid, 0) + feats.shape[0]

    if not all_patches:
        print('没有提取到任何 patch！')
        return

    M = torch.cat(all_patches, dim=0)
    print(f'\nMemory bank (原始): {M.shape[0]} patches x {M.shape[1]} dims, '
          f'{len(per_patient)} 患者')

    if M.shape[0] > args.max_features:
        idx = torch.randperm(M.shape[0])[:args.max_features]
        M = M[idx]
        print(f'降采样 → {M.shape[0]} patches')

    torch.save({'M': M, 'organ': 'mediastinum_chest_ct',
                'dtype': 'float32', 'per_patient': per_patient,
                'classes': BANK_CLASSES},
               args.output)
    print(f'Saved → {args.output}')


if __name__ == '__main__':
    main()
