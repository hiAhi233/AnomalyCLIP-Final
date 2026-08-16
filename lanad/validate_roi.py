"""
lanad/validate_roi.py — 233 例 ROI 金标准 同域留出验证
================================================================
同域留出: 前 N_bank 例 (ROI 反选正常区) 建库, 其余验证。
阳性图 = 含病灶层 (ROI>0), 阴性图 = 无病灶层 (ROI==0, 同批扫描)。
指标 (lanad/metrics.py, MVTec 标准公式):
  Image-level AUROC / AP, Pixel-level AUROC, AUPRO, argmax 命中率

基线: 单层 patch + 记忆库最近邻 (当前 Path B, 无训练)

Usage:
  python lanad/validate_roi.py --nii-dir "D:/图神经网络/异常检测/全部图像及ROI" \
      --bank-patients 100 --eval-patients 133
"""

import argparse, os, re, sys, warnings, json
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
from lanad.metrics import (image_level_metrics, pixel_level_auroc,
                           auroc_pro, argmax_hit_rate)

WIN_W, WIN_L = 400, 45
SIZE = 224
GRID = 14
MAX_CROPS_PER_SLICE = 20
MAX_BANK_FEATURES = 50000


def apply_window(ct_slice):
    low, high = WIN_L - WIN_W // 2, WIN_L + WIN_W // 2
    img = np.clip(ct_slice, low, high)
    return ((img - low) / (high - low) * 255).astype(np.uint8)


def crop_normal_regions(ct_slice, roi_slice, max_crops=MAX_CROPS_PER_SLICE):
    """含病灶层面中 ROI==0 的正常区域网格裁剪"""
    h, w = ct_slice.shape
    crops = []
    step = 112  # 50% 重叠 (patch 224)
    for y in range(0, h - SIZE + 1, step):
        for x in range(0, w - SIZE + 1, step):
            if len(crops) >= max_crops:
                return crops
            roi_patch = roi_slice[y:y+SIZE, x:x+SIZE]
            if (roi_patch > 0).mean() < 0.05:
                ct_patch = ct_slice[y:y+SIZE, x:x+SIZE]
                if ct_patch.std() > 5:
                    crops.append(ct_patch)
    return crops


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
    ap.add_argument('--bank-patients', type=int, default=100)
    ap.add_argument('--eval-patients', type=int, default=133)
    ap.add_argument('--max-slices-per-patient', type=int, default=12)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='output/anomaly_eval')
    args = ap.parse_args()

    print('Loading BiomedCLIP ...')
    model, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.float().eval().to(args.device)

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    complete = find_patients(args.nii_dir)
    pids = sorted(complete.keys())
    bank_pids = pids[:args.bank_patients]
    eval_pids = pids[args.bank_patients:args.bank_patients + args.eval_patients]
    print(f'总 {len(pids)} 例 | 建库 {len(bank_pids)} | 验证 {len(eval_pids)}')

    # ================= 建同域库 (ROI 反选正常区, 缓存) =================
    bank_cache = os.path.join(args.out, 'bank_leaveout.pt')
    if os.path.exists(bank_cache):
        M = torch.load(bank_cache, map_location='cpu', weights_only=True)
        print(f'同域库 (缓存): {M.shape[0]} patches')
    else:
        all_patches = []
        for pid in tqdm(bank_pids, desc='建库'):
            ct = nib.load(complete[pid]['ct']).get_fdata()
            roi = nib.load(complete[pid]['roi']).get_fdata()
            z_clean = [z for z in range(roi.shape[2]) if (roi[:, :, z] > 0).sum() == 0]

            # 整层无病灶切片 (与验证同表征: 512→224 降采样整层), 覆盖全解剖范围
            for z in z_clean[::2]:
                img = apply_window(ct[:, :, z])
                inp = tfm(Image.fromarray(np.stack([img] * 3, axis=-1)))
                with torch.no_grad():
                    raw = model.visual.trunk.forward_features(
                        inp.unsqueeze(0).to(args.device).type(model.text.transformer.dtype))
                    feats = raw[:, 1:, :].squeeze(0)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                all_patches.append(feats[::4].cpu())
        M = torch.cat(all_patches, dim=0)
        if M.shape[0] > MAX_BANK_FEATURES:
            idx = torch.randperm(M.shape[0])[:MAX_BANK_FEATURES]
            M = M[idx]
        os.makedirs(args.out, exist_ok=True)
        torch.save(M, bank_cache)
        print(f'同域库: {M.shape[0]} patches')
    M = M.to(args.device)

    # ================= 验证 (多图像级分数公式对比) =================
    amaps, masks = [], []
    score_sets = {k: [] for k in ['mean', 'max', 'top10', 'frac_025']}
    labels = []
    for pid in tqdm(eval_pids, desc='验证'):
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
                feats = raw[:, 1:, :].squeeze(0)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                d = 1.0 - (feats @ M.t()).amax(dim=-1)   # [196]
            dd = d.cpu().numpy()
            return {'mean': float(dd.mean()),
                    'max': float(dd.max()),
                    'top10': float(np.sort(dd)[-10:].mean()),
                    'frac_025': float((dd > 0.25).mean())}, \
                   dd.reshape(GRID, GRID)

        for z in z_tumor[::max(1, len(z_tumor)//args.max_slices_per_patient)][:args.max_slices_per_patient]:
            sc, am = score_slice(z)
            for k in score_sets:
                score_sets[k].append(sc[k])
            labels.append(1)
            roi_224 = np.array(Image.fromarray((roi[:, :, z] > 0).astype(np.uint8) * 255)
                               .resize((SIZE, SIZE), Image.NEAREST)) > 127
            am_224 = np.array(Image.fromarray(am.astype(np.float32)).resize((SIZE, SIZE), Image.BILINEAR))
            amaps.append(am_224)
            masks.append(roi_224)
        # 阴性层 (同批无病灶层)
        step = max(1, len(z_clean)//args.max_slices_per_patient)
        for z in z_clean[::step][:args.max_slices_per_patient]:
            sc, _ = score_slice(z)
            for k in score_sets:
                score_sets[k].append(sc[k])
            labels.append(0)

    labels = np.asarray(labels, dtype=np.int64)

    # 诊断: 正负两组的分数分布
    print('\n--- 分数分布诊断 (正=有病灶层, 负=无病灶层) ---')
    for k in score_sets:
        s = np.asarray(score_sets[k])
        sp, sn = s[labels == 1], s[labels == 0]
        print(f'{k:8s}: 正 {sp.mean():.4f}±{sp.std():.4f} | 负 {sn.mean():.4f}±{sn.std():.4f}')

    pix_auroc = pixel_level_auroc(amaps, masks)
    pro = auroc_pro(amaps, masks)
    hit = argmax_hit_rate(amaps, masks)

    results = {'bank_patients': len(bank_pids), 'eval_patients': len(eval_pids),
               'n_pos': int(labels.sum()), 'n_neg': int((1 - labels).sum()),
               'pixel_auroc': pix_auroc, 'aupro': pro, 'argmax_hit': hit,
               'image_level': {}}
    for k in score_sets:
        auroc, ap = image_level_metrics(score_sets[k], labels)
        results['image_level'][k] = {'auroc': auroc, 'ap': ap}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'baseline_singlelayer.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('\n' + '=' * 56)
    print(f"正/负样本: {int(labels.sum())}/{int((1-labels).sum())}")
    for k, v in results['image_level'].items():
        print(f"image_auroc[{k:8s}]: {v['auroc']:.4f}  ap: {v['ap']:.4f}")
    print(f'pixel_auroc: {pix_auroc:.4f}  aupro: {pro:.4f}  argmax_hit: {hit:.4f}')
    print('=' * 56)


if __name__ == '__main__':
    main()
