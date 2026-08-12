"""
build_thymoma_memory_bank.py — 胸腺瘤正常组织记忆库
================================================================
胸腺瘤没有 normal_scan 目录，但每例都有肿瘤 ROI 掩膜。
用 ROI 反选：ROI == 0 的区域 = 正常组织（纵隔/肺/软组织）。

流程:
  每例患者增强 CT (xxx020-origin.nii.gz) + ROI (xxx020.nii.gz)
  → 含肿瘤层面: ROI==0 正常区域网格裁剪
  → 正常组织 patch 图 → BiomedCLIP ViT → 768 维特征
  → 聚合到 M = [N, 768]

注意:
  训练/测试时同一患者的正常 patch 进记忆库会造成轻微泄漏，
  233 例数据下影响可接受；正式实验可做 leave-one-out。

Usage:
  python build_thymoma_memory_bank.py
"""

import argparse, os, sys, re, warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")

import numpy as np
import torch
import nibabel as nib
import torchvision.transforms as T
from tqdm import tqdm

from open_clip.src.open_clip import create_model_from_pretrained

NII_DIR = r"D:\图神经网络\异常检测\全部图像及ROI"
WINDOW_WIDTH = 400
WINDOW_LEVEL = 45
PATCH_SIZE = 224
MAX_PATCHES_PER_PATIENT = 120
MAX_TOTAL_FEATURES = 50000   # 总特征上限（每个 crop 提取 49 个特征）


def apply_window(ct_slice, width=WINDOW_WIDTH, level=WINDOW_LEVEL):
    low = level - width // 2
    high = level + width // 2
    img = np.clip(ct_slice, low, high)
    img = (img - low) / (high - low) * 255
    return img.astype(np.uint8)


def crop_normal_regions(ct_slice, roi_slice, size=PATCH_SIZE, max_crops=40):
    """从含肿瘤层面裁剪 ROI==0 的正常组织区域"""
    h, w = ct_slice.shape
    crops = []
    step = size // 2  # 50% 重叠网格
    for y in range(0, h - size + 1, step):
        for x in range(0, w - size + 1, step):
            if len(crops) >= max_crops:
                break
            roi_patch = roi_slice[y:y+size, x:x+size]
            tumor_ratio = (roi_patch > 0).mean()
            if tumor_ratio < 0.05:
                ct_patch = ct_slice[y:y+size, x:x+size]
                if ct_patch.std() > 5:  # 过滤纯黑背景
                    crops.append(ct_patch)
        if len(crops) >= max_crops:
            break
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nii_dir', default=NII_DIR)
    ap.add_argument('--output', default='memory_bank_thymoma.pt')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    print('Loading BiomedCLIP...')
    model, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.float().eval().to(args.device)

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    from PIL import Image

    # 递归扫描患者文件（增强期 020，病灶显示更清晰）
    patients = {}
    for root, dirs, files in os.walk(args.nii_dir):
        for f in files:
            m = re.match(r'^(\d{3})020(-origin)?\.nii\.gz$', f)
            if not m:
                continue
            pid = int(m.group(1))
            if m.group(2):
                patients.setdefault(pid, {})['ct'] = os.path.join(root, f)
            else:
                patients.setdefault(pid, {})['roi'] = os.path.join(root, f)

    complete = {pid: fs for pid, fs in patients.items()
                if 'ct' in fs and 'roi' in fs}
    print(f'可用患者: {len(complete)} 例')

    all_patches = []
    per_patient_counts = {}

    for pid in tqdm(sorted(complete.keys()), desc='Patients'):
        ct_path = complete[pid]['ct']
        roi_path = complete[pid]['roi']

        ct = nib.load(ct_path).get_fdata()
        roi = nib.load(roi_path).get_fdata()

        z_with_tumor = [z for z in range(roi.shape[2]) if (roi[:, :, z] > 0).sum() > 0]
        if not z_with_tumor:
            print(f'  [SKIP] P{pid}: ROI 为空')
            continue

        patient_patches = []
        for z in z_with_tumor:
            ct_img = apply_window(ct[:, :, z])
            roi_img = roi[:, :, z]
            patient_patches.extend(crop_normal_regions(ct_img, roi_img))

        if len(patient_patches) > MAX_PATCHES_PER_PATIENT:
            idx = np.random.default_rng(pid).choice(
                len(patient_patches), MAX_PATCHES_PER_PATIENT, replace=False)
            patient_patches = [patient_patches[i] for i in idx]

        per_patient_counts[pid] = len(patient_patches)

        for crop in patient_patches:
            img = np.stack([crop]*3, axis=-1)
            inp = tfm(Image.fromarray(img.astype(np.uint8)))
            with torch.no_grad():
                raw = model.visual.trunk.forward_features(
                    inp.unsqueeze(0).to(args.device).type(model.text.transformer.dtype))
                feats = raw[:, 1:, :].squeeze(0)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_patches.append(feats[::4].cpu())  # 每 4 个取 1 控制内存

    if not all_patches:
        print('没有提取到任何 patch！')
        return

    M = torch.cat(all_patches, dim=0)
    n_used = sum(1 for v in per_patient_counts.values() if v > 0)
    print(f'\nMemory bank (原始): {M.shape[0]} patches x {M.shape[1]} dims')

    # 降采样到 MAX_TOTAL_FEATURES（控制推理时余弦距离计算量）
    if M.shape[0] > MAX_TOTAL_FEATURES:
        idx = torch.randperm(M.shape[0])[:MAX_TOTAL_FEATURES]
        M = M[idx]
        print(f'降采样 → {M.shape[0]} patches')

    print(f'使用患者: {n_used}/{len(complete)}')
    print(f'每患者 patch 范围: {min(per_patient_counts.values())}~{max(per_patient_counts.values())}')

    torch.save({'M': M, 'organ': 'thymoma_chest_ct',
                'dtype': 'float32', 'per_patient': per_patient_counts},
               args.output)
    print(f'Saved → {args.output}')


if __name__ == '__main__':
    main()
