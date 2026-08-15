"""
train_structured_classifier.py — 分类正式版: 结构化影像征象分类器
================================================================
依据探针结论: 19 维结构化征象 (大小/CT值/强化/钙化/形态等, 放射科读片测量值)
是纵隔占位分类的最强单模态 (11类 test 56.5%, 二分类 79.1%)。

本脚本 = 正式训练与评估:
  - 患者级 70/15/15 划分 (与 split_Mediastinum.json 一致)
  - LogisticRegression (11类 多分类 + 二分类 胸腺瘤vs其他)
  - 输出: 每类 P/R/F1、混淆矩阵、模型 (joblib)、预测明细 (论文用)

Usage:
  python train_structured_classifier.py
"""

import os, sys, json, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

CSV_PATH = r'D:\图神经网络\异常检测\代码demo3\数据集2\胸腺区占位\胸腺区占位_AI3.csv'
OUT_DIR = 'output/classification'

NUM = ['长径mm', '短径mm', '年龄', '胸大肌平扫密度CT值', '肿块平扫密度CT值',
       '病变动脉期CT值', '病变静脉期CT值', 'AFP', 'HCG', 'LDH', 'HCT红细胞压积']
CAT = ['性别', '钙化', '形态', '边缘边界', '囊变坏死', '周围情况', '增强情况', '偏侧性']

CLASSES = ['thymoma', 'thymic_carcinoma', 'cyst', 'teratoma', 'lymphoma',
           'germ_cell_tumor', 'neuroendocrine_tumor', 'hyperplasia',
           'benign_lesion', 'metastasis', 'other_malignant']
LABEL_MAP = {'胸腺瘤': 0, '胸腺癌': 1, '良性囊肿': 2, '成熟型囊性畸胎瘤': 3, '淋巴瘤': 4,
             '生殖细胞肿瘤': 5, '神经内分泌肿瘤': 6, '胸腺增生': 7, '非肿瘤性病变': 8,
             '转移瘤': 9, '肉瘤': 10, '混合性生殖细胞肿瘤': 5, '精原细胞瘤': 5,
             '卵黄囊瘤': 5, '绒毛膜癌': 5, '淋巴上皮瘤样癌': 1, '未成熟型畸胎瘤': 3,
             'Castleman病': 8, '纤维性肿瘤': 8, '结节性甲状腺肿': 8, '神经鞘瘤': 8,
             '血管瘤/淋巴管瘤': 8, '胸腺脂肪瘤': 8, '纵隔型肺癌': 10, '甲状腺癌': 10}
CLASS_CN = {0: '胸腺瘤', 1: '胸腺癌', 2: '良性囊肿', 3: '畸胎瘤', 4: '淋巴瘤',
            5: '生殖细胞肿瘤', 6: '神经内分泌肿瘤', 7: '胸腺增生', 8: '良性病变',
            9: '转移瘤', 10: '其他恶性'}


def load_split_pids():
    """从 split_Mediastinum.json 提取患者级划分 (每患者一行)"""
    split = json.load(open('data/Mediastinum/split_Mediastinum.json', encoding='utf-8'))
    out = {part: [] for part in split}
    for part in split:
        seen = set()
        for rel, lab, cname in split[part]:
            pid = int(re.search(r'P(\d+)_', rel).group(1))
            if pid not in seen:
                seen.add(pid)
                out[part].append(pid)
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs(OUT_DIR, exist_ok=True)

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
    pid2idx = {p: i for i, p in enumerate(csv['影像号'].values)}

    split_pids = load_split_pids()
    print(f'患者划分: train {len(split_pids["train"])} / val {len(split_pids["val"])} / test {len(split_pids["test"])}')

    results = {}
    for part in split_pids:
        idx = [pid2idx[p] for p in split_pids[part] if p in pid2idx]
        results[part] = {
            'X': X.iloc[idx].values, 'y11': y11[idx], 'yb': yb[idx], 'pids': [p for p in split_pids[part] if p in pid2idx]
        }

    scaler = StandardScaler().fit(results['train']['X'])
    for part in results:
        results[part]['X'] = scaler.transform(results[part]['X'])

    def train_eval(name, ykey, n_cls):
        print(f'\n===== {name} =====')
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(results['train']['X'], results['train'][ykey])
        for part in ['train', 'val', 'test']:
            y, pred = results[part][ykey], clf.predict(results[part]['X'])
            acc = accuracy_score(y, pred) * 100
            mf1 = f1_score(y, pred, average='macro') * 100
            print(f'  {part:5s}: acc {acc:5.1f}%  macro-F1 {mf1:5.1f}%')
        # 测试集每类明细
        y, pred = results['test'][ykey], clf.predict(results['test']['X'])
        print('  测试集每类 (P/R/F1/样本数):')
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(n_cls), zero_division=0)
        for i in range(n_cls):
            print(f'    {CLASSES[i]:22s}: P={p[i]:.2f} R={r[i]:.2f} F1={f[i]:.2f} n={s[i]}')
        cm = confusion_matrix(y, pred, labels=range(n_cls))
        np.save(os.path.join(OUT_DIR, f'{name}_confusion.npy'), cm)
        joblib.dump(clf, os.path.join(OUT_DIR, f'{name}_model.joblib'))
        # 预测明细 (论文用)
        rows = [{'pid': pid, 'label': int(yy), 'pred': int(pp)}
                for pid, yy, pp in zip(results['test']['pids'], y, pred)]
        with open(os.path.join(OUT_DIR, f'{name}_test_predictions.json'), 'w', encoding='utf-8') as fp:
            json.dump(rows, fp, ensure_ascii=False, indent=1)
        return clf

    train_eval('binary', 'yb', 2)
    train_eval('class11', 'y11', 11)
    joblib.dump(scaler, os.path.join(OUT_DIR, 'scaler.joblib'))
    print(f'\n已保存 → {OUT_DIR}/')


if __name__ == '__main__':
    main()
