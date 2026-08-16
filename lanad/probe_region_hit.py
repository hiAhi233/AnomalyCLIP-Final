"""
lanad/probe_region_hit.py — 零模型区域级定位探针
================================================================
点级读数失败 (argmax 被骨头抢), 但分类器只需要"病灶大概在哪个区域"。
本探针测: NN 距离热力图 → 3×3 解剖格区域积分 → 最热格
         vs ROI 所在格 → 区域级命中率。

注意: 热力图与 ROI 在同一图像坐标系, 网格比较与图像朝向无关。

Usage:
  python lanad/probe_region_hit.py
"""

import os, sys, re, json, warnings, argparse
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")

import numpy as np
import torch
import nibabel as nib
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from open_clip.src.open_clip import create_model_from_pretrained

SIZE, GRID = 224, 14


def apply_window(ct_slice, ww=400, wl=45):
    low, high = wl - ww // 2, wl + ww // 2
    img = np.clip(ct_slice, low, high)
    return ((img - low) / (high - low) * 255).astype(np.uint8)


def find_patients(nii_dir):
    patients = {}
    for root, _, files in os.walk(nii_dir):
        for f in files:
            m = re.match(r'^(\d{3})020(-origin)?\.nii\.gz$', f)
            if not m:
                continue
            pid = int(m.group(1))
            if m.group(2):
                patients.setdefault(pid, {})['ct'] = os.path.join(root, f)
            else:
                patients.setdefault(pid, {})['roi'] = os.path.join(root, f)
    return {pid: fs for pid, fs in patients.items() if 'ct' in fs and 'roi' in fs}


def to_3x3(am_14, roi_grid):
    """am_14: [14,14] 热力图; roi_grid: [14,14] 0/1
    → 区域积分向量 [9] (每个 3x3 大格的热力图和), ROI 主格 (0-8)"""
    cell_sums = np.zeros(9)
    roi_cells = np.zeros(9)
    for i in range(14):
        for j in range(14):
            ci, cj = min(i * 3 // 14, 2), min(j * 3 // 14, 2)
            c = ci * 3 + cj
            cell_sums[c] += am_14[i, j]
            roi_cells[c] += roi_grid[i, j]
    return cell_sums, int(roi_cells.argmax()) if roi_cells.sum() > 0 else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nii-dir', default=r'D:\图神经网络\异常检测\全部图像及ROI')
    ap.add_argument('--bank', default='output/anomaly_eval/bank_leaveout.pt')
    ap.add_argument('--eval-patients', type=int, default=133)
    ap.add_argument('--max-slices', type=int, default=12)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='output/anomaly_eval')
    args = ap.parse_args()

    print('Loading BiomedCLIP + bank ...')
    model, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.float().eval().to(args.device)
    M = torch.load(args.bank, map_location='cpu', weights_only=True).to(args.device)
    print(f'Bank: {M.shape[0]}')

    tfm = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    complete = find_patients(args.nii_dir)
    pids = sorted(complete.keys())[100:100 + args.eval_patients]

    hit1 = hit3 = n = 0
    rank_dist = []   # 最热格与 ROI 格的格心距离
    roi_cell_ranks = []  # ROI 格在 9 格中的热度排名 (1=最热)
    for pid in tqdm(pids, desc='验证'):
        ct = nib.load(complete[pid]['ct']).get_fdata()
        roi = nib.load(complete[pid]['roi']).get_fdata()
        z_tumor = [z for z in range(roi.shape[2]) if (roi[:, :, z] > 0).sum() > 0]
        for z in z_tumor[::max(1, len(z_tumor)//args.max_slices)][:args.max_slices]:
            img = apply_window(ct[:, :, z])
            inp = tfm(Image.fromarray(np.stack([img] * 3, axis=-1)))
            with torch.no_grad():
                raw = model.visual.trunk.forward_features(
                    inp.unsqueeze(0).to(args.device).type(model.text.transformer.dtype))
                f = raw[:, 1:, :].squeeze(0)
                f = f / f.norm(dim=-1, keepdim=True)
                d = (1.0 - (f @ M.t()).amax(dim=-1)).cpu().numpy()
            am = d.reshape(GRID, GRID)
            roi_224 = np.array(Image.fromarray((roi[:, :, z] > 0).astype(np.uint8) * 255)
                               .resize((SIZE, SIZE), Image.NEAREST)) > 127
            roi_grid = np.array(Image.fromarray(roi_224.astype(np.uint8) * 255)
                                .resize((GRID, GRID), Image.NEAREST)) > 127
            cell_sums, roi_cell = to_3x3(am, roi_grid)
            if roi_cell < 0:
                continue
            order = np.argsort(cell_sums)[::-1]
            hot_cell = int(order[0])
            hit1 += int(hot_cell == roi_cell)
            hit3 += int(roi_cell in order[:3])
            cy, cx = divmod(hot_cell, 3)
            ry, rx = divmod(roi_cell, 3)
            rank_dist.append(np.hypot(cy - ry, cx - rx))
            roi_cell_ranks.append(int(np.where(order == roi_cell)[0][0]) + 1)
            n += 1

    results = {
        'n_slices': n,
        'region_hit_top1': hit1 / n,
        'region_hit_top3': hit3 / n,
        'mean_cell_dist': float(np.mean(rank_dist)),
        'roi_cell_median_rank': float(np.median(roi_cell_ranks)),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'probe_region_hit.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('\n' + '=' * 56)
    for k, v in results.items():
        print(f'{k:26s}: {v:.4f}')
    print('=' * 56)
    print('判定: region_hit_top1 > 0.6 → 区域级读数可用 (进分类器)')
    print('      否则 → 半监督分割头 (50 例掩膜)')


if __name__ == '__main__':
    main()
