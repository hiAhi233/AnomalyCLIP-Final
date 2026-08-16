"""
lanad/probe_image_score.py — 区域积分图级分数探针
================================================================
patch 级均值/最大值被解剖混杂压住 (图级 AUROC 倒挂)。
本探针: 用 3×3 区域积分统计量当图级分数, 测正负两组可分性:
  · max_cell:   最热格热量
  · conc:       热量集中度 = max_cell / 总热量
  · gap:        最热格 - 中位格
Usage:
  python lanad/probe_image_score.py
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
from lanad.metrics import image_level_metrics

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


def cells_of(am):
    cells = np.zeros(9)
    for i in range(14):
        for j in range(14):
            cells[min(i * 3 // 14, 2) * 3 + min(j * 3 // 14, 2)] += am[i, j]
    return cells


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

    tfm = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    complete = find_patients(args.nii_dir)
    pids = sorted(complete.keys())[100:100 + args.eval_patients]

    score_sets = {k: [] for k in ['max_cell', 'conc', 'gap']}
    labels = []
    for pid in tqdm(pids, desc='验证'):
        ct = nib.load(complete[pid]['ct']).get_fdata()
        roi = nib.load(complete[pid]['roi']).get_fdata()
        z_tumor = [z for z in range(roi.shape[2]) if (roi[:, :, z] > 0).sum() > 0]
        z_clean = [z for z in range(roi.shape[2]) if (roi[:, :, z] > 0).sum() == 0]

        def score_slice(z):
            img = apply_window(ct[:, :, z])
            inp = tfm(Image.fromarray(np.stack([img] * 3, axis=-1)))
            with torch.no_grad():
                raw = model.visual.trunk.forward_features(
                    inp.unsqueeze(0).to(args.device).type(model.text.transformer.dtype))
                f = raw[:, 1:, :].squeeze(0)
                f = f / f.norm(dim=-1, keepdim=True)
                d = (1.0 - (f @ M.t()).amax(dim=-1)).cpu().numpy()
            c = cells_of(d.reshape(GRID, GRID))
            c_sorted = np.sort(c)
            return {'max_cell': float(c.max()),
                    'conc': float(c.max() / (c.sum() + 1e-8)),
                    'gap': float(c.max() - np.median(c))}

        for z in z_tumor[::max(1, len(z_tumor)//args.max_slices)][:args.max_slices]:
            sc = score_slice(z)
            for k in score_sets:
                score_sets[k].append(sc[k])
            labels.append(1)
        step = max(1, len(z_clean)//args.max_slices)
        for z in z_clean[::step][:args.max_slices]:
            sc = score_slice(z)
            for k in score_sets:
                score_sets[k].append(sc[k])
            labels.append(0)

    labels = np.asarray(labels, dtype=np.int64)
    print('\n--- 区域分数分布 (正=病灶层, 负=无病灶层) ---')
    results = {}
    for k in score_sets:
        s = np.asarray(score_sets[k])
        print(f'{k:10s}: 正 {s[labels==1].mean():.4f}±{s[labels==1].std():.4f} | '
              f'负 {s[labels==0].mean():.4f}±{s[labels==0].std():.4f}')
        a, ap_ = image_level_metrics(s, labels)
        results[k] = {'auroc': a, 'ap': ap_}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'probe_image_score.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('\n' + '=' * 56)
    for k, v in results.items():
        print(f'image_auroc[{k:10s}]: {v["auroc"]:.4f}  ap: {v["ap"]:.4f}')
    print('=' * 56)


if __name__ == '__main__':
    main()
