"""
eval_anomaly_localization.py — 用 233 例 ROI 金标准验证异常检测定位
================================================================
带 ROI 的胸腺瘤数据 (全部图像及ROI, 医生勾画) 作为金标准,
验证 Path B 异常检测 + 定位 + 阈值标定:

  1. 检测:  病灶 patch 的异常分 vs 正常 patch (patch 级 AUC)
  2. 定位:  14x14 异常热力图 vs ROI 掩膜 (IoU / 命中率)
  3. 标定:  最优"异常 patch"阈值 vs 当前 mean+1.5σ 启发式

流程 (每例):
  nii CT + ROI → 含病灶层面 → 纵隔窗 224 切片 → BiomedCLIP patch 特征
  → 与 memory_bank_mediastinum.pt 最近邻距离 → 14x14 热力图
  → 与 ROI 比较 + 可视化抽查

Usage:
  python eval_anomaly_localization.py --nii-dir "D:/图神经网络/异常检测/全部图像及ROI" \
      --bank memory_bank_mediastinum.pt --max-patients 60
"""

import argparse, os, sys, re, warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")

import numpy as np
import torch
import nibabel as nib
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image

from open_clip.src.open_clip import create_model_from_pretrained

WIN_W, WIN_L = 400, 45
SIZE = 224
GRID = 14
CELL = SIZE // GRID   # 16


def apply_window(ct_slice):
    low, high = WIN_L - WIN_W // 2, WIN_L + WIN_W // 2
    img = np.clip(ct_slice, low, high)
    img = ((img - low) / (high - low) * 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nii-dir', default=r'D:\图神经网络\异常检测\全部图像及ROI')
    ap.add_argument('--bank', default='memory_bank_mediastinum.pt')
    ap.add_argument('--max-patients', type=int, default=0, help='0=全部')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out-overlays', default='output/anomaly_overlays')
    args = ap.parse_args()

    print('Loading BiomedCLIP + memory bank ...')
    model, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.float().eval().to(args.device)
    bank = torch.load(args.bank, map_location='cpu', weights_only=True)['M']
    bank = bank.to(args.device)
    print(f'Bank: {bank.shape[0]} patches')

    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    # ---- 扫描 nii: CT(020-origin) + ROI(020) ----
    patients = {}
    for root, _, files in os.walk(args.nii_dir):
        for f in files:
            m = re.match(r'^(\d{3})020(-origin)?\.nii\.gz$', f)
            if not m:
                continue
            pid = int(m.group(1))
            if m.group(2):
                patients.setdefault(pid, {})['ct'] = os.path.join(root, f)
            else:
                patients.setdefault(pid, {})['roi'] = os.path.join(root, f)
    complete = {pid: fs for pid, fs in patients.items() if 'ct' in fs and 'roi' in fs}
    pids = sorted(complete.keys())
    if args.max_patients:
        pids = pids[:args.max_patients]
    print(f'ROI 患者: {len(complete)}, 本次验证: {len(pids)}')

    os.makedirs(args.out_overlays, exist_ok=True)

    all_scores, all_labels = [], []     # patch 级: 异常分 vs ROI标签
    hit_cnt = total_img = 0             # 定位命中: argmax patch 在 ROI 内
    ious = []                           # 图像级 IoU (阈值 mean+1.5σ)
    img_scores_tumor, img_scores_norm = [], []  # s_img 统计
    overlay_cnt = 0

    for pid in tqdm(pids, desc='Patients'):
        try:
            ct = nib.load(complete[pid]['ct']).get_fdata()
            roi = nib.load(complete[pid]['roi']).get_fdata()
        except Exception as e:
            print(f'  [SKIP] P{pid}: {e}')
            continue

        for z in range(roi.shape[2]):
            if (roi[:, :, z] > 0).sum() == 0:
                continue
            img = apply_window(ct[:, :, z])
            roi_z = roi[:, :, z]
            # ROI 上采样到 224
            roi_224 = np.array(Image.fromarray((roi_z > 0).astype(np.uint8) * 255)
                               .resize((SIZE, SIZE), Image.NEAREST)) > 127

            inp = tfm(Image.fromarray(np.stack([img] * 3, axis=-1)))
            with torch.no_grad():
                raw = model.visual.trunk.forward_features(
                    inp.unsqueeze(0).to(args.device).type(model.text.transformer.dtype))
                feats = raw[:, 1:, :].squeeze(0)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                cos_sim = feats @ bank.T
                dist = 1.0 - cos_sim.amax(dim=-1)      # [196]
            amap = dist.reshape(GRID, GRID).cpu().numpy()

            # patch 级标签: 16x16 单元内 ROI 占比 > 10% 视为病灶 patch
            labels = np.zeros((GRID, GRID), dtype=bool)
            for i in range(GRID):
                for j in range(GRID):
                    cell = roi_224[i*CELL:(i+1)*CELL, j*CELL:(j+1)*CELL]
                    labels[i, j] = cell.mean() > 0.1

            all_scores.append(amap.ravel())
            all_labels.append(labels.ravel())

            # 定位命中: argmax patch 落在 ROI patch
            argmax_in_roi = labels.ravel()[amap.ravel().argmax()]
            hit_cnt += int(argmax_in_roi)
            total_img += 1

            # 当前阈值 (mean+1.5σ) 下的 IoU
            thr = amap.mean() + 1.5 * amap.std()
            pred = amap > thr
            inter = (pred & labels).sum()
            union = (pred | labels).sum()
            ious.append(inter / max(union, 1))

            img_scores_tumor.append(amap[labels].mean() if labels.any() else 0.0)
            img_scores_norm.append(amap[~labels].mean() if (~labels).any() else 0.0)

            # 可视化抽查 (前 20 张)
            if overlay_cnt < 20:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img, cmap='gray'); axes[0].set_title('CT')
                axes[1].imshow(img, cmap='gray')
                axes[1].imshow(np.array(Image.fromarray((roi_224*255).astype(np.uint8)).resize((GRID, GRID), Image.NEAREST)) > 127, alpha=0.4, cmap='Reds')
                axes[1].set_title('ROI(降采样)')
                axes[2].imshow(amap, cmap='hot'); axes[2].set_title('异常热力图')
                plt.tight_layout()
                plt.savefig(os.path.join(args.out_overlays, f'P{pid}_z{z}.png'), dpi=80)
                plt.close()
                overlay_cnt += 1

    # ---- 汇总指标 ----
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels).astype(int)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, scores)
    print('\n' + '=' * 56)
    print(f'验证图像数: {total_img}')
    print(f'病灶 patch 平均异常分: {np.mean(img_scores_tumor):.4f}  '
          f'正常 patch 平均异常分: {np.mean(img_scores_norm):.4f}')
    print(f'patch 级检测 AUC: {auc:.4f}')
    print(f'定位命中率 (最亮patch在ROI内): {hit_cnt}/{total_img} = {hit_cnt/max(total_img,1)*100:.1f}%')
    print(f'当前阈值(mean+1.5σ) 平均 IoU: {np.mean(ious):.4f}')

    # 阈值标定: 扫 F1
    best_f1, best_t = 0, 0
    for t in np.linspace(0.02, 0.6, 60):
        pred = (scores > t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f'最优阈值: {best_t:.3f} (F1={best_f1:.4f}) — 当前启发式 mean+1.5σ 仅供参考')
    print(f'可视化抽查: {args.out_overlays}/ (前 20 张)')
    print('=' * 56)


if __name__ == '__main__':
    main()
