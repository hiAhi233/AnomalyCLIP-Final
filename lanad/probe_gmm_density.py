"""
lanad/probe_gmm_density.py — GMM 密度探针 (LADAR 通道①, CS-Flow 式)
================================================================
假设: 最近邻距离对"忙碌但正常"的解剖敏感 (图级倒挂的根因);
密度似然 -log p(x) 会把它们判为高似然 → 低异常分 → 修倒挂。

探针: 正常 patch 特征 (PCA 768→128) → GMM (64 成分, 对角协方差)
→ patch 异常分 = -log p → 233 例同域留出四指标, 与 NN 距离并排对比。

Usage:
  python lanad/probe_gmm_density.py
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
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from open_clip.src.open_clip import create_model_from_pretrained
from lanad.metrics import (image_level_metrics, pixel_level_auroc,
                           auroc_pro, argmax_hit_rate)

SIZE, GRID = 224, 14
PCA_DIM, N_COMP = 128, 64


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
    ap.add_argument('--bank-patients', type=int, default=100)
    ap.add_argument('--eval-patients', type=int, default=133)
    ap.add_argument('--max-slices', type=int, default=12)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='output/anomaly_eval')
    ap.add_argument('--n-components', type=int, default=N_COMP)
    args = ap.parse_args()

    print('Loading BiomedCLIP ...')
    model, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.float().eval().to(args.device)
    tfm = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    complete = find_patients(args.nii_dir)
    pids = sorted(complete.keys())
    bank_pids = pids[:args.bank_patients]
    eval_pids = pids[args.bank_patients:args.bank_patients + args.eval_patients]

    # ============ 拟合阶段: 正常 patch 特征 (整层无病灶切片) ============
    cache = os.path.join(args.out, 'gmm.pkl')
    nn_bank_path = os.path.join(args.out, 'nn_bank.npy')
    if os.path.exists(cache):
        pca, gmm = joblib.load(cache)
        print(f'GMM/PCA (缓存): {gmm.n_components} 成分')
    else:
        feats_all = []
        for pid in tqdm(bank_pids, desc='提取正常特征'):
            ct = nib.load(complete[pid]['ct']).get_fdata()
            roi = nib.load(complete[pid]['roi']).get_fdata()
            for z in range(roi.shape[2]):
                if (roi[:, :, z] > 0).sum() > 0:
                    continue
                if z % 2:
                    continue
                img = apply_window(ct[:, :, z])
                inp = tfm(Image.fromarray(np.stack([img] * 3, axis=-1)))
                with torch.no_grad():
                    raw = model.visual.trunk.forward_features(
                        inp.unsqueeze(0).to(args.device).type(model.text.transformer.dtype))
                    f = raw[:, 1:, :].squeeze(0)
                    f = f / f.norm(dim=-1, keepdim=True)
                feats_all.append(f[::4].cpu().numpy())
        X = np.concatenate(feats_all, axis=0)
        print(f'拟合样本: {X.shape[0]} patches')
        idx = np.random.default_rng(0).permutation(X.shape[0])[:50000]
        X = X[idx]
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        pca = PCA(n_components=PCA_DIM, random_state=0).fit(X)
        Xr = pca.transform(X)
        gmm = GaussianMixture(n_components=args.n_components, covariance_type='diag',
                              max_iter=200, random_state=0).fit(Xr)
        os.makedirs(args.out, exist_ok=True)
        joblib.dump((pca, gmm), cache)
        _idx2 = np.random.default_rng(1).permutation(X.shape[0])[:5000]
        np.save(nn_bank_path, X[_idx2])   # NN 对照库 (5k 子样)
        print('GMM 拟合完成')

    global _bank
    _bank = np.load(nn_bank_path)

    # ============ 验证: NN 距离 vs GMM 密度 并排 ============
    amaps_nn, amaps_gmm, masks = [], [], []
    score_sets = {k: [] for k in ['nn_mean', 'nn_max', 'gmm_mean', 'gmm_max']}
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
                f = raw[:, 1:, :].squeeze(0)
                f = f / f.norm(dim=-1, keepdim=True)
            fn = f.cpu().numpy()
            # NN 距离 (全库, 但探针只对比密度 — 用拟合集子样 5k 当库)
            sim = fn @ _bank[:5000].T
            d_nn = 1.0 - sim.max(axis=1)
            # GMM 密度
            zr = pca.transform(fn[::4])
            logp = gmm.score_samples(zr)          # [49]
            nll = -logp                             # 异常分
            nll_full = np.repeat(nll, 4)[:196]      # 对齐 196 patch (feats[::4] 还原)
            return d_nn, nll_full, nll_full.reshape(GRID, GRID)

        for z in z_tumor[::max(1, len(z_tumor)//args.max_slices)][:args.max_slices]:
            d_nn, nll, am_gmm = score_slice(z)
            score_sets['nn_mean'].append(d_nn.mean()); score_sets['nn_max'].append(d_nn.max())
            score_sets['gmm_mean'].append(nll.mean()); score_sets['gmm_max'].append(nll.max())
            labels.append(1)
            roi_224 = np.array(Image.fromarray((roi[:, :, z] > 0).astype(np.uint8) * 255)
                               .resize((SIZE, SIZE), Image.NEAREST)) > 127
            am_224 = np.array(Image.fromarray(am_gmm.astype(np.float32)).resize((SIZE, SIZE), Image.BILINEAR))
            amaps_gmm.append(am_224)
            masks.append(roi_224)
            amaps_nn.append(np.array(Image.fromarray(d_nn.reshape(GRID, GRID).astype(np.float32))
                                     .resize((SIZE, SIZE), Image.BILINEAR)))
        step = max(1, len(z_clean)//args.max_slices)
        for z in z_clean[::step][:args.max_slices]:
            d_nn, nll, _ = score_slice(z)
            score_sets['nn_mean'].append(d_nn.mean()); score_sets['nn_max'].append(d_nn.max())
            score_sets['gmm_mean'].append(nll.mean()); score_sets['gmm_max'].append(nll.max())
            labels.append(0)

    labels = np.asarray(labels, dtype=np.int64)
    print('\n--- 分数分布诊断 (正=有病灶层, 负=无病灶层) ---')
    for k in score_sets:
        s = np.asarray(score_sets[k])
        print(f'{k:10s}: 正 {s[labels==1].mean():.4f}±{s[labels==1].std():.4f} | 负 {s[labels==0].mean():.4f}±{s[labels==0].std():.4f}')

    results = {'image_level': {}}
    for k in score_sets:
        a, ap_ = image_level_metrics(score_sets[k], labels)
        results['image_level'][k] = {'auroc': a, 'ap': ap_}
    results['pixel_auroc_nn'] = pixel_level_auroc(amaps_nn, masks)
    results['pixel_auroc_gmm'] = pixel_level_auroc(amaps_gmm, masks)
    results['aupro_gmm'] = auroc_pro(amaps_gmm, masks)
    results['argmax_gmm'] = argmax_hit_rate(amaps_gmm, masks)
    with open(os.path.join(args.out, 'probe_gmm.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('\n' + '=' * 56)
    for k, v in results['image_level'].items():
        print(f"image_auroc[{k:10s}]: {v['auroc']:.4f}  ap: {v['ap']:.4f}")
    print(f'pixel_auroc: NN {results["pixel_auroc_nn"]:.4f} | GMM {results["pixel_auroc_gmm"]:.4f}')
    print(f'aupro_gmm: {results["aupro_gmm"]:.4f}  argmax_gmm: {results["argmax_gmm"]:.4f}')
    print('=' * 56)


# 模块级: NN 对照库 (拟合集正常 patch 特征子样), 在 main 内填充
_bank = None

if __name__ == '__main__':
    main()
