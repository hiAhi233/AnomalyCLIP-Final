"""
lanad/probe_ae_reconstruction.py — AE 重建探针 (Plan B, f-AnoGAN 式)
================================================================
只学正常切片的自编码器: 病灶结构没学过 → 重建残差 = 定位。
验证: 233 例 ROI 金标准 (同域留出) — 四指标 + 质心读数。

Usage:
  python lanad/probe_ae_reconstruction.py          # 训练+验证
  python lanad/probe_ae_reconstruction.py --eval-only   # 只验证(用已训模型)
"""

import os, sys, re, json, warnings, argparse
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lanad.metrics import (image_level_metrics, pixel_level_auroc,
                           auroc_pro, argmax_hit_rate)

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


class AE(nn.Module):
    """简单对称卷积自编码器: 3→32→64→128→256 下采样, latent 14x14x256, 镜像解码"""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nii-dir', default=r'D:\图神经网络\异常检测\全部图像及ROI')
    ap.add_argument('--normal-data', default='data/Mediastinum',
                    help='正常切片来源: 良性组(hyperplasia/benign_lesion) PNG')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--eval-patients', type=int, default=133)
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='output/anomaly_eval')
    args = ap.parse_args()

    model_path = os.path.join(args.out, 'ae.pt')
    tfm = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    # ================= 训练数据: 正常切片 =================
    normal_imgs = []
    for cls in ['hyperplasia', 'benign_lesion']:
        d = os.path.join(args.normal_data, cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith('.png'):
                normal_imgs.append(os.path.join(d, f))
    # 233 无病灶层也加入
    complete = find_patients(args.nii_dir)
    pids = sorted(complete.keys())
    for pid in pids[:80]:   # 只用前80例的干净层 (验证用后133例)
        ct = nib.load(complete[pid]['ct']).get_fdata()
        roi = nib.load(complete[pid]['roi']).get_fdata()
        for z in range(roi.shape[2]):
            if (roi[:, :, z] > 0).sum() > 0 or z % 4:
                continue
            img = apply_window(ct[:, :, z])
            normal_imgs.append(('np', img))
    print(f'正常切片: {len(normal_imgs)}')

    class NormalDS(torch.utils.data.Dataset):
        def __len__(self):
            return len(normal_imgs)

        def __getitem__(self, i):
            item = normal_imgs[i]
            if isinstance(item, str):
                img = np.array(Image.open(item).convert('RGB'))
            else:
                img = np.stack([item[1]] * 3, axis=-1)
            return tfm(Image.fromarray(img.astype(np.uint8)))

    ae = AE().to(args.device)
    if os.path.exists(model_path):
        ae.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f'AE 加载: {model_path}')
    if not args.eval_only:
        opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
        loader = torch.utils.data.DataLoader(NormalDS(), batch_size=args.batch, shuffle=True,
                                             num_workers=0)
        ae.train()
        for ep in range(1, args.epochs + 1):
            tot = 0.0
            for x in tqdm(loader, desc=f'E{ep}', leave=False):
                x = x.to(args.device)
                xr = ae(x)
                loss = nn.functional.mse_loss(xr, x)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(x)
            print(f'epoch {ep:3d}: loss {tot/len(normal_imgs):.5f}')
        os.makedirs(args.out, exist_ok=True)
        torch.save(ae.state_dict(), model_path)
        print('AE 保存完成')

    # ================= 验证 =================
    ae.eval()
    amaps, masks = [], []
    s_pos, s_neg = [], []
    labels = []
    for pid in tqdm(pids[100:100 + args.eval_patients], desc='验证'):
        ct = nib.load(complete[pid]['ct']).get_fdata()
        roi = nib.load(complete[pid]['roi']).get_fdata()
        z_tumor = [z for z in range(roi.shape[2]) if (roi[:, :, z] > 0).sum() > 0]
        z_clean = [z for z in range(roi.shape[2]) if (roi[:, :, z] > 0).sum() == 0]

        def score_slice(z):
            img = apply_window(ct[:, :, z])
            x = tfm(Image.fromarray(np.stack([img] * 3, axis=-1))).unsqueeze(0).to(args.device)
            with torch.no_grad():
                xr = ae(x)
            resid = (x - xr).abs().mean(dim=1).squeeze(0).cpu().numpy()   # [224,224]
            return float(resid.mean()), resid

        for z in z_tumor[::max(1, len(z_tumor)//12)][:12]:
            s, am = score_slice(z)
            s_pos.append(s); labels.append(1)
            roi_224 = np.array(Image.fromarray((roi[:, :, z] > 0).astype(np.uint8) * 255)
                               .resize((SIZE, SIZE), Image.NEAREST)) > 127
            amaps.append(am); masks.append(roi_224)
        step = max(1, len(z_clean)//12)
        for z in z_clean[::step][:12]:
            s, _ = score_slice(z)
            s_neg.append(s); labels.append(0)

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(s_pos + s_neg)
    ia, ip = image_level_metrics(scores, labels)
    pa = pixel_level_auroc(amaps, masks)
    pro = auroc_pro(amaps, masks)
    hit = argmax_hit_rate(amaps, masks)
    results = {'n_pos': len(s_pos), 'n_neg': len(s_neg),
               'image_auroc': ia, 'image_ap': ip,
               'pixel_auroc': pa, 'aupro': pro, 'argmax_hit': hit,
               'pos_mean_resid': float(scores[labels == 1].mean()),
               'neg_mean_resid': float(scores[labels == 0].mean())}
    with open(os.path.join(args.out, 'probe_ae.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('\n' + '=' * 56)
    for k, v in results.items():
        print(f'{k:20s}: {v}')
    print('=' * 56)


if __name__ == '__main__':
    main()
