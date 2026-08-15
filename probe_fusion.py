"""
probe_fusion.py — 图像特征 + 结构化征象 融合探针
================================================================
背景 (已测基线):
  图像冻结特征:       11类 CV 16.8%  | 二分类 CV 56.5%
  图像微调特征:       二分类 test F1 ~55%
  结构化特征(19维):   11类 CV 55.2%  | 二分类 CV 77.8% / F1 72.1%
  融合 = 两者拼接 → 逻辑回归, 看是否互补

Usage:
  python probe_fusion.py
"""

import os, sys, json, re, warnings, argparse
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")

import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from open_clip.src.open_clip import create_model_from_pretrained

CSV_PATH = r'D:\图神经网络\异常检测\代码demo3\数据集2\胸腺区占位\胸腺区占位_AI3.csv'

NUM = ['长径mm', '短径mm', '年龄', '胸大肌平扫密度CT值', '肿块平扫密度CT值',
       '病变动脉期CT值', '病变静脉期CT值', 'AFP', 'HCG', 'LDH', 'HCT红细胞压积']
CAT = ['性别', '钙化', '形态', '边缘边界', '囊变坏死', '周围情况', '增强情况', '偏侧性']
LABEL_MAP = {'胸腺瘤': 0, '胸腺癌': 1, '良性囊肿': 2, '成熟型囊性畸胎瘤': 3, '淋巴瘤': 4,
             '生殖细胞肿瘤': 5, '神经内分泌肿瘤': 6, '胸腺增生': 7, '非肿瘤性病变': 8,
             '转移瘤': 9, '肉瘤': 10, '混合性生殖细胞肿瘤': 5, '精原细胞瘤': 5,
             '卵黄囊瘤': 5, '绒毛膜癌': 5, '淋巴上皮瘤样癌': 1, '未成熟型畸胎瘤': 3,
             'Castleman病': 8, '纤维性肿瘤': 8, '结节性甲状腺肿': 8, '神经鞘瘤': 8,
             '血管瘤/淋巴管瘤': 8, '胸腺脂肪瘤': 8, '纵隔型肺癌': 10, '甲状腺癌': 10}


def load_data():
    csv = pd.read_csv(CSV_PATH)
    csv['影像号'] = csv['影像号'].astype(int)
    X = pd.DataFrame()
    for c in NUM:
        X[c] = pd.to_numeric(csv[c], errors='coerce')
    for c in CAT:
        X[c] = csv[c].astype(str).astype('category').cat.codes
    X = X.fillna(X.median())
    y11 = csv['病理诊断_标准化'].map(LABEL_MAP).values
    yb = (csv['病理诊断_标准化'] == '胸腺瘤').astype(int).values
    return csv, X, y11, yb


def load_rep_split(data_dir='data/Mediastinum'):
    reps = set()
    for dp, _, fns in os.walk(data_dir):
        for f in fns:
            if f.endswith('_rep.txt'):
                with open(os.path.join(dp, f), encoding='utf-8') as fp:
                    reps.add(os.path.normpath(os.path.join(dp, fp.read().strip())))
    split = json.load(open(os.path.join(data_dir, 'split_Mediastinum.json'), encoding='utf-8'))
    out = {part: [] for part in split}
    for part in split:
        for rel, lab, cname in split[part]:
            p = os.path.normpath(os.path.join(data_dir, rel))
            if p in reps:
                out[part].append((p, lab))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tuned-model', default='output/probe_lora_binary/probe_model.pt',
                    help='微调过的 visual 检查点 (空=用冻结特征)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    csv, X_csv, y11_all, yb_all = load_data()
    pid2idx = {pid: i for i, pid in enumerate(csv['影像号'].values)}
    scaler = StandardScaler().fit(X_csv)
    X_csv_s = scaler.transform(X_csv)

    # ---- 图像特征 ----
    model, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.float().to(args.device)
    if args.tuned_model and os.path.exists(args.tuned_model):
        ckpt = torch.load(args.tuned_model, map_location='cpu', weights_only=True)
        model.visual.load_state_dict(ckpt['visual'])
        print(f'使用微调 visual: {args.tuned_model}')
    else:
        print('使用冻结 visual')
    model.eval()
    tfm = T.Compose([
        T.Resize(224), T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    split = load_rep_split()
    F_img, F_csv, Y11, YB, PIDS = {}, {}, {}, {}, {}
    for part in split:
        fi, fc, y11, yb, pids = [], [], [], [], []
        for p, lab in split[part]:
            pid = int(re.search(r'P(\d+)_', p).group(1))
            if pid not in pid2idx:
                continue
            img = Image.open(p).convert('RGB')
            with torch.no_grad():
                f = model.visual(tfm(img).unsqueeze(0).to(args.device)).squeeze(0).cpu().numpy()
            fi.append(f)
            fc.append(X_csv_s[pid2idx[pid]])
            y11.append(y11_all[pid2idx[pid]]); yb.append(yb_all[pid2idx[pid]])
            pids.append(pid)
        F_img[part] = np.stack(fi); F_csv[part] = np.stack(fc)
        Y11[part] = np.array(y11); YB[part] = np.array(yb)
        print(f'{part}: {len(fi)} 例 (有 CSV 记录)')

    def report(name, Xtr, Xte, ytr, yte):
        # 训练集内 CV + 测试集
        clf = LogisticRegression(max_iter=3000, C=1.0)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        cv_acc = cross_val_score(clf, Xtr, ytr, cv=cv, scoring='accuracy').mean()
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        te_acc = (pred == yte).mean()
        te_f1 = f1_score(yte, pred, average='macro')
        print(f'{name:28s}: CV acc {cv_acc*100:5.1f}% | test acc {te_acc*100:5.1f}%  test F1 {te_f1*100:5.1f}%')

    for tag, Y in [('11类', Y11), ('二分类', YB)]:
        print(f'\n===== {tag} =====')
        report('图像特征 only', F_img['train'], F_img['test'], Y['train'], Y['test'])
        report('结构化特征 only', F_csv['train'], F_csv['test'], Y['train'], Y['test'])
        report('图像+结构化 融合', np.concatenate([F_img['train'], F_csv['train']], 1),
               np.concatenate([F_img['test'], F_csv['test']], 1), Y['train'], Y['test'])


if __name__ == '__main__':
    main()
