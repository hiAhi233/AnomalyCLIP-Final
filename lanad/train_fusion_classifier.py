"""
lanad/train_fusion_classifier.py — 训练并保存融合分类器 (19维征象 + 热力11维)
================================================================
与 probe_classify_fusion 同逻辑, 额外把模型+scaler 保存到 output/classification,
供 generate_report.py 推理加载 (二分类 84.3% / 11类 58.3%)。

Usage:
  python lanad/train_fusion_classifier.py
"""

import os, sys, re, json, warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import joblib
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
OUT = 'output/classification'


def heat11(path, model, M, tfm, device):
    img = Image.open(path).convert('RGB')
    inp = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        raw = model.visual.trunk.forward_features(inp.type(model.text.transformer.dtype))
        f = raw[:, 1:, :].squeeze(0)
        f = f / f.norm(dim=-1, keepdim=True)
        d = (1.0 - (f @ M.t()).amax(dim=-1)).cpu().numpy()
    am = d.reshape(14, 14)
    cells = np.zeros(9)
    for i in range(14):
        for j in range(14):
            cells[min(i * 3 // 14, 2) * 3 + min(j * 3 // 14, 2)] += am[i, j]
    cells = cells / (cells.sum() + 1e-8)
    return np.concatenate([cells, [am.mean(), am.max()]])


def main():
    print('Loading BiomedCLIP + bank ...')
    model, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.float().eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    M = torch.load('memory_bank_mediastinum.pt', map_location='cpu')['M'].to(device)
    tfm = T.Compose([T.ToTensor(),
                     T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))])

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

    parts = {}
    for part in items:
        Xs, Xh, Y11, YB = [], [], [], []
        for path, lab in tqdm(items[part], desc=part):
            pid = int(re.search(r'P(\d+)_', os.path.basename(path)).group(1))
            if pid not in pid2i:
                continue
            Xs.append(X[pid2i[pid]]); Xh.append(heat11(path, model, M, tfm, device))
            Y11.append(y11[pid2i[pid]]); YB.append(yb[pid2i[pid]])
        parts[part] = (np.stack(Xs), np.stack(Xh), np.array(Y11), np.array(YB))

    os.makedirs(OUT, exist_ok=True)
    for tag, ykey in [('binary', 'yb'), ('class11', 'y11')]:
        ytr = parts['train'][2] if ykey == 'y11' else parts['train'][3]
        yte = parts['test'][2] if ykey == 'y11' else parts['test'][3]
        Atr = np.hstack([parts['train'][0], parts['train'][1]])
        Ate = np.hstack([parts['test'][0], parts['test'][1]])
        sc = StandardScaler().fit(Atr)
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Atr), ytr)
        pred = clf.predict(sc.transform(Ate))
        print(f'{tag}: acc {accuracy_score(yte, pred)*100:.1f}%  '
              f'macro_f1 {f1_score(yte, pred, average="macro")*100:.1f}%')
        joblib.dump(clf, os.path.join(OUT, f'{tag}_fusion_model.joblib'))
        joblib.dump(sc, os.path.join(OUT, f'scaler_fusion.joblib'))
    print(f'已保存 → {OUT}/ (fusion models + scaler_fusion.joblib)')


if __name__ == '__main__':
    main()
