"""
胸腺瘤 CT nii → 2D 切片数据集构建
================================================================
输入: nii CT + ROI + Excel 报告
输出: Dassl 兼容的数据集目录（data/Thymoma/）

每例患者:
  平扫 CT (xxx000-origin.nii.gz) → 取肿瘤最大层 + 上下各 2 层
  增强 CT (xxx020-origin.nii.gz) → 取肿瘤最大层 + 上下各 2 层
  共 ~10 张 PNG 切片 / 患者

标签: 从 Excel 报告结论提取（胸腺瘤 / 非胸腺瘤）

用法:
  python build_thymoma_dataset.py
"""

import os, re, json
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from collections import defaultdict

# ============================================================
# 配置
# ============================================================

NII_DIR = r"D:\图神经网络\异常检测\全部图像及ROI"
EXCEL_PATH = r"D:\图神经网络\异常检测\一致性图像（可重复勾画的实验）\5.胸腺瘤CT报告233例（别外传，报告还是算隐私的）.xlsx"
OUT_DIR = r"D:\图神经网络\异常检测\代码demo4\BiomedCoOp-main\data\Thymoma"

WINDOW_WIDTH = 400   # CT 窗宽
WINDOW_LEVEL = 45    # CT 窗位
CONTEXT_SLICES = 2   # 肿瘤最大层上下各取 N 层


# ============================================================
# 工具函数
# ============================================================

def apply_window(ct_slice: np.ndarray, width: int, level: int) -> np.ndarray:
    """CT 窗宽窗位 → 0-255"""
    low = level - width // 2
    high = level + width // 2
    img = np.clip(ct_slice, low, high)
    img = (img - low) / (high - low) * 255
    return img.astype(np.uint8)


def extract_tumor_slices(ct_path: str, roi_path: str,
                         context: int = CONTEXT_SLICES) -> list:
    """
    从 CT + ROI 提取肿瘤切片。

    Returns:
        [(slice_idx, ct_slice_uint8), ...]  按 z 轴排序
    """
    ct = nib.load(ct_path).get_fdata()    # [H, W, D]
    roi = nib.load(roi_path).get_fdata()

    # 找 ROI 面积最大的层
    roi_areas = [(roi[:, :, z] > 0).sum() for z in range(roi.shape[2])]
    if max(roi_areas) == 0:
        # 无 ROI，取中心层
        best_z = roi.shape[2] // 2
    else:
        best_z = int(np.argmax(roi_areas))

    # 取 best_z ± context
    start = max(0, best_z - context)
    end = min(roi.shape[2], best_z + context + 1)
    z_indices = list(range(start, end))

    slices = []
    for z in z_indices:
        ct_slice = ct[:, :, z]
        roi_slice = roi[:, :, z]
        has_tumor = (roi_slice > 0).any()

        img = apply_window(ct_slice, WINDOW_WIDTH, WINDOW_LEVEL)
        slices.append((z, img, has_tumor))

    return slices


def extract_label(conclusion: str) -> str:
    """从报告结论提取简化标签"""
    if pd.isna(conclusion):
        return "unknown"
    c = str(conclusion)
    if "胸腺瘤" in c or "胸腺" in c:
        if "恶性" in c:
            return "malignant_thymoma"
        return "thymoma"
    if "良性" in c:
        return "benign"
    if "淋巴瘤" in c:
        return "lymphoma"
    if "生殖源" in c:
        return "germ_cell_tumor"
    if "甲状腺" in c:
        return "thyroid_mass"
    return "other"


# ============================================================
# 主流程
# ============================================================

def main():
    # ---- 读 Excel ----
    df = pd.read_excel(EXCEL_PATH)
    df['自编号'] = df['自编号'].astype(int)
    print(f"Excel: {len(df)} 条报告")

    # ---- 递归扫描 nii 文件（233 例分布在多个子目录: 003-65/ 124-220/ ...）----
    nii_files = defaultdict(dict)
    for root, dirs, files in os.walk(NII_DIR):
        for f in files:
            m = re.match(r'^(\d{3})(\d{3})(-origin)?\.nii\.gz$', f)
            if not m:
                continue
            pid = int(m.group(1))
            suffix = m.group(2)   # "000" or "020"
            is_origin = m.group(3) is not None

            if is_origin:
                nii_files[pid][f"{suffix}_ct"] = os.path.join(root, f)
            else:
                nii_files[pid][f"{suffix}_roi"] = os.path.join(root, f)

    print(f"nii: {len(nii_files)} 例患者")

    # ---- 过滤：必须四件套齐全 ----
    valid_pids = []
    for pid in sorted(nii_files.keys()):
        fs = nii_files[pid]
        if all(k in fs for k in ["000_ct", "000_roi", "020_ct", "020_roi"]):
            valid_pids.append(pid)

    print(f"四件套齐全: {len(valid_pids)} 例 → {valid_pids}")

    # 输出目录在保存时按需创建，不预建空目录

    # ---- 切片提取 ----
    all_samples = []
    for pid in valid_pids:
        fs = nii_files[pid]

        # 标签
        row = df[df['自编号'] == pid]
        if len(row) == 0:
            print(f"  [SKIP] 自编号={pid}: Excel 无记录")
            continue
        conclusion = row.iloc[0]['报告结论']
        label = extract_label(conclusion)
        label_dir = os.path.join(OUT_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        print(f"  自编号={pid} → {label}: {str(conclusion)[:60]}")

        patient_slices = []

        for modality, name in [("000", "平扫"), ("020", "增强")]:
            ct_path = fs[f"{modality}_ct"]
            roi_path = fs[f"{modality}_roi"]

            slices = extract_tumor_slices(ct_path, roi_path)
            for z, img, has_tumor in slices:
                # 文件名: P{pid}_{modality}_z{z}_{tumor}.png
                fname = f"P{pid:03d}_{name}_z{z:02d}.png"
                fpath = os.path.join(label_dir, fname)
                Image.fromarray(img).save(fpath)
                patient_slices.append({
                    "impath": fpath,
                    "label": label,
                    "classname": label,
                })
                all_samples.append(fpath)

        # 保存报告原文
        findings = str(row.iloc[0].get('影像所见', ''))
        conclusion_full = str(conclusion)
        report_txt = f"影像所见：{findings if findings != 'nan' else ''}\n报告结论：{conclusion_full if conclusion_full != 'nan' else ''}"
        rpt_path = os.path.join(label_dir, f"P{pid:03d}_报告.txt")
        with open(rpt_path, 'w', encoding='utf-8') as f:
            f.write(report_txt)

    print(f"\n总计: {len(all_samples)} 张切片 → {OUT_DIR}")

    # ---- 打印各类别分布 ----
    print("\n类别分布:")
    for cls_name in os.listdir(OUT_DIR):
        cls_dir = os.path.join(OUT_DIR, cls_name)
        if os.path.isdir(cls_dir):
            pngs = [f for f in os.listdir(cls_dir) if f.endswith('.png')]
            reports = [f for f in os.listdir(cls_dir) if f.endswith('.txt')]
            print(f"  {cls_name}: {len(pngs)} 张切片, {len(reports)} 份报告")


if __name__ == "__main__":
    main()
