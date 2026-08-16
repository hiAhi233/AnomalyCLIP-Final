"""
lanad/metrics.py — 异常检测定位标准指标
================================================================
按 MVTec-AD 基准的标准公式实现, 保证评分正确:

  Image-level AUROC: 每图异常分 s_img vs 图级标签 (0=无病灶/1=有病灶)
  Image-level AP:    average precision (PR 曲线下面积)
  Pixel-level AUROC: 逐像素异常值 vs 逐像素 GT 掩膜
  AUPRO:             按 GT 连通域计算 per-region overlap (PRO 曲线,
                     FPR∈[0,0.3] 下面积), 阈值集 200 档
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import label


def image_level_metrics(scores, labels):
    """scores: [N] float; labels: [N] int 0/1 → (auroc, ap)"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return float('nan'), float('nan')
    return (float(roc_auc_score(labels, scores)),
            float(average_precision_score(labels, scores)))


def pixel_level_auroc(amaps, masks):
    """amaps: list[np.ndarray [H,W]] 异常图; masks: list[np.ndarray [H,W]] 0/1 GT"""
    scores = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in amaps])
    targets = np.concatenate([(np.asarray(m) > 0).astype(np.int64).ravel() for m in masks])
    if targets.sum() == 0 or (targets == 1).all():
        return float('nan')
    return float(roc_auc_score(targets, scores))


def _normalize(amap):
    a = np.asarray(amap, dtype=np.float64)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def auroc_pro(amaps, masks, max_fpr=0.3, gt_component_min_area=4, num_th=200):
    """
    AUPRO — per-region overlap (MVTec 标准公式):
      1) GT 掩膜取面积 ≥ gt_component_min_area 的连通域 (3x3 全连通)
      2) 每张异常图 min-max 归一化到 [0,1], 200 档阈值
      3) 每阈值: 逐图像 per-pixel FPR = (pred&~gt)/~gt (无负像素 → 0)
                逐连通域 overlap = |pred∩comp| / |comp|
      4) PRO 曲线: 平均 overlap vs 平均 FPR → 梯形积分 FPR∈[0, max_fpr]
    """
    amaps = [_normalize(a) for a in amaps]
    masks = [np.asarray(m) > 0 for m in masks]

    # 收集所有 GT 连通域 (每图独立标号, 面积过滤)
    comps = []           # [(img_idx, comp_bool)]
    for i, m in enumerate(masks):
        lab, n = label(m, structure=np.ones((3, 3)))
        for j in range(1, n + 1):
            c = lab == j
            if c.sum() >= gt_component_min_area:
                comps.append((i, c))
    if not comps:
        return 0.0

    # 预计算每图的负像素数
    neg_counts = [int((~m).sum()) for m in masks]
    # 阈值: 全局分数分位数线性扫描
    all_scores = np.concatenate([a.ravel() for a in amaps])
    ths = np.linspace(float(all_scores.min()), float(all_scores.max()), num_th)

    fpr_curve, pro_curve = [], []
    for t in ths:
        preds = [a >= t for a in amaps]
        # 平均逐图像 FPR
        fprs = []
        for a, m, nneg in zip(preds, masks, neg_counts):
            if nneg == 0:
                fprs.append(0.0)
            else:
                fprs.append(float((a & ~m).sum()) / nneg)
        # 平均逐连通域 overlap
        overlaps = []
        for i, c in comps:
            inter = int((preds[i] & c).sum())
            overlaps.append(inter / int(c.sum()))
        fpr_curve.append(float(np.mean(fprs)))
        pro_curve.append(float(np.mean(overlaps)))

    fpr_curve = np.asarray(fpr_curve)
    pro_curve = np.asarray(pro_curve)
    sel = fpr_curve <= max_fpr
    if sel.sum() < 2:
        return 0.0
    # 确保左端点在 0
    f = np.concatenate([[0.0], fpr_curve[sel]])
    p = np.concatenate([[pro_curve[sel][0]], pro_curve[sel]])
    return float(np.trapz(p, f) / max_fpr)


def argmax_hit_rate(amaps, masks):
    """定位命中: 最亮像素落在 GT 内 → 命中 (参考指标)"""
    hit = tot = 0
    for a, m in zip(amaps, masks):
        a = np.asarray(a)
        m = np.asarray(m) > 0
        if not m.any():
            continue
        hit += int(m[a.argmax() // a.shape[1], a.argmax() % a.shape[1]])
        tot += 1
    return hit / tot if tot else float('nan')
