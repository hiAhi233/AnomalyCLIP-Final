"""
lanad/probe_classify_fusion.py — 定位读数 → 分类器 融合探针 (路 A)
================================================================
热力图 3×3 区域热力分布 (9维) + 强度统计 (2维) 拼进 19 维结构化征象
→ LogisticRegression, 看 79.1%(二分类)/56.5%(11类) 涨不涨。

零新模型: 热力图来自现有记忆库 NN 距离, 分类器不变。

Usage:
  python lanad/probe_classify_fusion.py
"""

import os, sys, re, json, warnings, argparse
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='memory_bank_mediastinum.pt')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='output/anomaly_eval')
    args = ap.parse_args()

    print('Loading BiomedCLIP + bank ...')
    model, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.float().eval().to(args.device)
    M = torch.load(args.bank, map_location='cpu', weights_only=True)['M'].to(args.device)

    tfm = T.Compose([T.ToTensor(),
                     T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))])

    # ---- 结构化征象 (19维) ----
    csv = pd.read_csv(CSV_PATH)
    csv['影像号'] = csv['影像号'].astype(int)
    X = pd.DataFrame()
    for c in NUM:
        X[c] = pd.to_numeric(csv[c], errors='coerce')
    for c in CAT:
        X[c] = csv[c].astype(str).astype('category').cat.codes
    X = X.fillna(X.median()).values
    y11 = csv['病理诊断_标准化'].map(LABEL_MAP).values
    yb = (csv['病理诊断_标准化'] == '胸腺瘤').astype(int).values
    pid2i = {p: i for i, p in enumerate(csv['影像号'].values)}

    # ---- 患者级划分 + 代表层 ----
    reps = set()
    for dp, _, fns in os.walk('data/Mediastinum'):
        for f in fns:
            if f.endswith('_rep.txt'):
                with open(os.path.join(dp, f), encoding='utf-8') as fp:
                    reps.add(os.path.normpath(os.path.join(dp, fp.read().strip())))
    split = json.load(open('data/Mediastinum/split_Mediastinum.json', encoding='utf-8'))
    items = {p: [] for p in split}
    for part in split:
        for rel, lab, cn in split[part]:
            path = os.path.normpath(os.path.join('data/Mediastinum', rel))
            if path in reps:
                items[part].append((path, lab))

    # ---- 热力图区域特征 (每代表层) ----
    def heat_feats(path):
        img = Image.open(path).convert('RGB')
        inp = tfm(img).unsqueeze(0).to(args.device)
        with torch.no_grad():
            raw = model.visual.trunk.forward_features(
                inp.type(model.text.transformer.dtype))
            f = raw[:, 1:, :].squeeze(0)
            f = f / f.norm(dim=-1, keepdim=True)
            d = (1.0 - (f @ M.t()).amax(dim=-1)).cpu().numpy()
        am = d.reshape(14, 14)
        cells = np.zeros(9)
        for i in range(14):
            for j in range(14):
                cells[min(i * 3 // 14, 2) * 3 + min(j * 3 // 14, 2)] += am[i, j]
        cells = cells / (cells.sum() + 1e-8)          # 9维区域分布
        return np.concatenate([cells, [am.mean(), am.max()]])   # 11维

    parts = {}
    for part in items:
        Xs, Xh, Y11, YB = [], [], [], []
        for path, lab in tqdm(items[part], desc=part):
            pid = int(re.search(r'P(\d+)_', os.path.basename(path)).group(1))
            if pid not in pid2i:
                continue
            Xs.append(X[pid2i[pid]]); Xh.append(heat_feats(path))
            Y11.append(y11[pid2i[pid]]); YB.append(yb[pid2i[pid]])
        parts[part] = (np.stack(Xs), np.stack(Xh), np.array(Y11), np.array(YB))
    print(f"train {len(parts['train'][2])} / val {len(parts['val'][2])} / test {len(parts['test'][2])}")

    def train_eval(tag, use_heat, ykey):
        A = parts['train'][0]
        B = parts['train'][1] if use_heat else None
        if use_heat:
            A = np.hstack([A, B])
        sc = StandardScaler().fit(A)
        Atr, Ate = sc.transform(A), None
        te_s = parts['test'][0]
        if use_heat:
            te_s = np.hstack([te_s, parts['test'][1]])
        Ate = sc.transform(te_s)
        ytr, yte = parts['train'][2] if ykey == 'y11' else parts['train'][3], \
                   parts['test'][2] if ykey == 'y11' else parts['test'][3]
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(Atr, ytr)
        pred = clf.predict(Ate)
        return accuracy_score(yte, pred) * 100, f1_score(yte, pred, average='macro') * 100

    results = {}
    for ykey, name in [('yb', '二分类'), ('y11', '11类')]:
        a0, f0 = train_eval(name, False, ykey)
        a1, f1 = train_eval(name, True, ykey)
        results[name] = {'only19': {'acc': a0, 'f1': f0},
                         'plus_heat': {'acc': a1, 'f1': f1}}
        print(f'{name}: 19维 alone  acc {a0:.1f}% F1 {f0:.1f}%  |  +热力11维  acc {a1:.1f}% F1 {f1:.1f}%')

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'probe_classify_fusion.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('已保存 → output/anomaly_eval/probe_classify_fusion.json')


if __name__ == '__main__':
    main()
