"""
lanad/probe_centroid_localization.py — 热力图"读数"验证 (质心/区域/直径)
================================================================
用 233 例 ROI 金标准验证: 从 NN 距离热力图提取的定位读数准不准。
读数 (喂分类器前必须先过这关):
  · 病灶质心 (top-k 加权) → 区域词/位置
  · 直径估计 (高分区等效直径)
判定: 质心区域命中率、质心距离(mm)、直径误差(mm)

Usage:
  python lanad/probe_centroid_localization.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nii-dir', default=r'D:\图神经网络\异常检测\全部图像及ROI')
    ap.add_argument('--bank', default='output/anomaly_eval/bank_leaveout.pt')
    ap.add_argument('--eval-patients', type=int, default=133)
    ap.add_argument('--max-slices', type=int, default=12)
    ap.add_argument('--top-frac', type=float, default=0.2, help='top-k 高分区比例')
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

    hits, dists_mm, diam_errs, argmax_hits = [], [], [], []
    n_slices = 0
    for pid in tqdm(pids, desc='验证'):
        ct = nib.load(complete[pid]['ct']).get_fdata()
        roi = nib.load(complete[pid]['roi']).get_fdata()
        zoom = float(nib.load(complete[pid]['ct']).header.get_zooms()[0])  # mm/px (512空间)
        scale = zoom * 512 / SIZE                                       # mm/px (224空间)
        z_tumor = [z for z in range(roi.shape[2]) if (roi[:, :, z] > 0).sum() > 0]

        for z in z_tumor[::max(1, len(z_tumor)//args.max_slices)][:args.max_slices]:
            img = apply_window(ct[:, :, z])
            inp = tfm(Image.fromarray(np.stack([img] * 3, axis=-1)))
            with torch.no_grad():
                raw = model.visual.trunk.forward_features(
                    inp.unsqueeze(0).to(args.device).type(model.text.transformer.dtype))
                f = raw[:, 1:, :].squeeze(0)
                f = f / f.norm(dim=-1, keepdim=True)
                d = (1.0 - (f @ M.t()).amax(dim=-1)).cpu().numpy()   # [196]
            am = d.reshape(GRID, GRID)
            roi_224 = np.array(Image.fromarray((roi[:, :, z] > 0).astype(np.uint8) * 255)
                               .resize((SIZE, SIZE), Image.NEAREST)) > 127
            roi_grid = np.array(Image.fromarray(roi_224.astype(np.uint8) * 255)
                                .resize((GRID, GRID), Image.NEAREST)) > 127

            # ---- top-k 加权质心 (224 像素坐标) ----
            flat = am.ravel()
            k = max(3, int(len(flat) * args.top_frac))
            top_idx = np.argsort(flat)[-k:]
            w = flat[top_idx] - flat.min()
            w = w / (w.sum() + 1e-8)
            ys = (top_idx // GRID + 0.5) * (SIZE / GRID)
            xs = (top_idx % GRID + 0.5) * (SIZE / GRID)
            cy, cx = (w * ys).sum(), (w * xs).sum()

            # ---- ROI 质心与尺寸 (224 空间) ----
            ry, rx = np.nonzero(roi_224)
            roi_cy, roi_cx = ry.mean(), rx.mean()
            dist_mm = float(np.hypot(cy - roi_cy, cx - roi_cx)) * scale

            # ---- 直径估计: 高分区(超阈值)等效直径 ----
            thr = am.mean() + 1.5 * am.std()
            above = am > thr
            if above.any():
                yy, xx = np.nonzero(above)
                diam_px = np.hypot(yy.max() - yy.min(), xx.max() - xx.min()) * (SIZE / GRID)
            else:
                diam_px = 0.0
            # ROI 真实长径 (512空间 → mm)
            ryy, rxx = np.nonzero(roi[:, :, z] > 0)
            true_diam_mm = float(np.hypot(ryy.max() - ryy.min(), rxx.max() - rxx.min())) * zoom
            diam_errs.append(abs(diam_px * scale - true_diam_mm))

            hits.append(int(roi_224[int(cy), int(cx)]))
            argmax_hits.append(int(roi_224[int(am.argmax() // GRID + 0.5) * 16, int(am.argmax() % GRID + 0.5) * 16]))
            dists_mm.append(dist_mm)
            n_slices += 1

    results = {
        'n_slices': n_slices,
        'centroid_region_hit': float(np.mean(hits)),
        'argmax_hit_reference': float(np.mean(argmax_hits)),
        'centroid_dist_mm_median': float(np.median(dists_mm)),
        'centroid_dist_mm_mean': float(np.mean(dists_mm)),
        'diameter_mae_mm': float(np.mean(diam_errs)),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'probe_centroid.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('\n' + '=' * 56)
    for k, v in results.items():
        print(f'{k:28s}: {v:.4f}')
    print('=' * 56)
    print('判定: 质心区域命中率 > 0.7 且直径 MAE 可接受 → 读数可进分类器')


if __name__ == '__main__':
    main()
