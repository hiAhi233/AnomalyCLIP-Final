"""
演示脚本：医学超声影像报告生成
================================================================
用法:
    # 使用模拟数据演示报告生成
    python generate_report.py --demo

    # 使用真实模型推理（需先训练或加载检查点）
    python generate_report.py --config-file configs/trainers/AnomalyDetect/few_shot/busi.yaml \
                              --dataset-config-file configs/datasets/busi.yaml \
                              --model-dir <checkpoint_dir> \
                              --image-path <image.jpg>
"""

import argparse
import json
import os
import sys
import random
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")

# 修复 Windows 控制台 GBK 编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn.functional as F
import numpy as np


def demo_mode():
    """
    使用模拟的模型输出演示报告生成，无需加载真实模型。
    模拟一个 BUSI 恶性病变的典型场景。
    """
    print("=" * 64)
    print("  医学超声影像报告生成 — 演示模式（模拟数据）")
    print("=" * 64)
    print()

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "report_generator",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "trainers", "AnomalyDetect", "report_generator.py"))
    _rg = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_rg)
    MedicalReportGenerator = _rg.MedicalReportGenerator

    # ---- 模拟 BUSI 分类输出 (benign, malignant, normal) ----
    classnames = ["benign", "malignant", "normal"]

    # 模拟一个恶性病变的推理输出
    logits = torch.tensor([0.8, 4.2, 0.3])  # malignant 得分最高

    # 模拟异常分数
    s_img = torch.tensor(0.72)               # 较显著异常

    # 模拟 patch 级异常分数 (14x14=196 patches)
    np.random.seed(42)
    anomaly_scores = np.random.beta(2, 3, size=(14, 14)).astype(np.float32)
    # 让中心区域异常更高
    for i in range(14):
        for j in range(14):
            dist = ((i - 7) ** 2 + (j - 7) ** 2) ** 0.5
            if dist < 4:
                anomaly_scores[i, j] = 0.6 + random.random() * 0.4
    anomaly_scores = torch.from_numpy(anomaly_scores.flatten())

    # 模拟概念分数
    concept_scores = {
        "normal": {
            "normal_tissue": torch.tensor(0.12),
            "regular_architecture": torch.tensor(0.18),
            "isoechoic_texture": torch.tensor(0.10),
        },
        "anomaly": {
            "tumor_mass": torch.tensor(0.85),
            "inflammation": torch.tensor(0.55),
            "necrosis": torch.tensor(0.38),
            "calcification": torch.tensor(0.22),
            "cyst_fluid": torch.tensor(0.05),
            "fibrosis_scar": torch.tensor(0.15),
        },
    }

    outputs = (logits, s_img, anomaly_scores, concept_scores, None)

    # ---- 生成报告 ----
    generator = MedicalReportGenerator(
        classnames=classnames,
        dataset_name="BUSI",
        modality="乳腺超声",
    )

    result = generator.generate(
        outputs=outputs,
        patient_id="DEMO001",
    )

    # ---- 打印报告 ----
    print(result["raw_text"])
    print()

    # ---- 展示 LLM 润色接口 ----
    print("─" * 64)
    print("  LLM 润色接口演示")
    print("─" * 64)
    print()
    print(">>> 以下是可以发送给大模型的润色 prompt：")
    print()
    print(generator.get_llm_polish_prompt())
    print()
    print("─" * 64)
    print("  使用方法:")
    print("  1. 复制上方的 prompt 发送给 ChatGPT / Claude / DeepSeek 等大模型")
    print("  2. 将大模型返回的润色后报告文本传给 report_generator.apply_llm_polish()")
    print("  3. 调用 report_generator.get_final_report() 获取润色后的报告")
    print("─" * 64)

    # ---- 展示评分模块 ----
    print()
    print("─" * 64)
    print("  报告评分演示")
    print("─" * 64)
    from trainers.AnomalyDetect.report_scorer import ReportScorer
    scorer = ReportScorer()
    # 用这份模拟报告做单份评分展示
    single_score = scorer.score_single(result, "malignant")
    print(f"\n  单份报告评分 (真实标签=malignant):")
    print(f"    诊断准确性:     {single_score['diagnosis_correct']:.0f} / 100")
    print(f"    异常分级对齐:   {single_score['anomaly_alignment']:.0f} / 100")
    print(f"    征象覆盖度:     {single_score['concept_coverage']:.0f} / 100")
    print(f"    安全性指标:     {single_score['safety']:.0f} / 100")
    print(f"    结构完整性:     {single_score['structure_complete']:.0f} / 100")
    print(f"    {'─'*40}")
    print(f"    综合得分:       {single_score['overall']:.1f} / 100")

    return result


def real_mode(config_file, dataset_config_file, model_dir, image_path):
    """
    加载真实模型进行推理并生成报告。
    """
    from dassl.config import get_cfg_default
    from dassl.engine import build_trainer
    from dassl.utils import setup_logger, set_random_seed

    import trainers.AnomalyDetect.anomaly_detect

    # ---- 加载配置 ----
    from train import setup_cfg, extend_cfg

    _model_dir = model_dir  # 捕获函数参数，class body 中无法直接引用

    class Args:
        root = "data"
        output_dir = ""
        resume = ""
        seed = 1
        source_domains = None
        target_domains = None
        transforms = None
        trainer = "AnomalyDetect_BiomedCLIP"
        backbone = "ViT-B/16"
        head = ""
        eval_only = True
        model_dir = _model_dir
        load_epoch = None
        no_train = True
        opts = []

    if config_file:
        Args.config_file = config_file
    else:
        Args.config_file = "configs/trainers/AnomalyDetect/few_shot/busi.yaml"

    if dataset_config_file:
        Args.dataset_config_file = dataset_config_file
    else:
        Args.dataset_config_file = "configs/datasets/busi.yaml"

    from train import main as train_main

    print("=" * 64)
    print("  医学超声影像报告生成 — 真实推理模式")
    print("=" * 64)
    print(f"  配置文件: {Args.config_file}")
    print(f"  模型目录: {model_dir}")
    print()

    # 构建 trainer
    import argparse as _argparse
    parser = _argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source-domains", type=str, nargs="+")
    parser.add_argument("--target-domains", type=str, nargs="+")
    parser.add_argument("--transforms", type=str, nargs="+")
    parser.add_argument("--config-file", type=str, default=Args.config_file)
    parser.add_argument("--dataset-config-file", type=str, default=Args.dataset_config_file)
    parser.add_argument("--trainer", type=str, default="AnomalyDetect_BiomedCLIP")
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--head", type=str, default="")
    parser.add_argument("--eval-only", action="store_true", default=True)
    parser.add_argument("--model-dir", type=str, default=model_dir)
    parser.add_argument("--load-epoch", type=int)
    parser.add_argument("--no-train", action="store_true", default=True)
    parser.add_argument("opts", nargs=_argparse.REMAINDER)

    known_args = parser.parse_args()

    from train import setup_cfg
    cfg = setup_cfg(known_args)

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    trainer = build_trainer(cfg)

    if model_dir:
        # 自动检测最新 checkpoint
        ckpt_dir = os.path.join(model_dir, "anomaly_clip")
        if os.path.isdir(ckpt_dir):
            epochs = []
            for f in os.listdir(ckpt_dir):
                if f.startswith("model.pth.tar-"):
                    try:
                        epochs.append(int(f.split("-")[-1]))
                    except ValueError:
                        pass
            epoch = max(epochs) if epochs else None
        else:
            epoch = None
        trainer.load_model(model_dir, epoch=epoch)

    # 获取 classnames
    classnames = trainer.dm.dataset.classnames

    # 创建报告输出目录
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "output", "generated_reports")
    os.makedirs(report_dir, exist_ok=True)

    # 加载真实医生报告作为 NLG 参考
    import re, json as _json
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", cfg.DATASET.NAME)
    real_refs = {}  # patient_id → report text
    for cls_dir in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        for f in os.listdir(cls_path):
            if f.endswith("_报告.txt"):
                m = re.match(r'P(\d+)_报告\.txt', f)
                if m:
                    pid = int(m.group(1))
                    with open(os.path.join(cls_path, f), "r", encoding="utf-8") as fp:
                        real_refs[pid] = fp.read().strip()
    print(f"  加载了 {len(real_refs)} 份真实报告作为 NLG 参考")

    # ---- 结构化征象查询表 (与训练侧同逻辑: 19维标准化 + 尺寸文本) ----
    import torch as _torch
    _struct_map = {}
    _size_map = {}
    _csv_path = getattr(cfg.TRAINER.ANOMALY_DETECT, 'STRUCT_CSV', '')
    if _csv_path and os.path.exists(_csv_path):
        import pandas as _pd
        from sklearn.preprocessing import StandardScaler as _SS
        from trainers.AnomalyDetect.report_generator_qwen import build_size_text as _bst
        _df = _pd.read_csv(_csv_path)
        _df['影像号'] = _df['影像号'].astype(int)
        _NUM_F = ['长径mm', '短径mm', '年龄', '胸大肌平扫密度CT值', '肿块平扫密度CT值',
                  '病变动脉期CT值', '病变静脉期CT值', 'AFP', 'HCG', 'LDH', 'HCT红细胞压积']
        _CAT_F = ['性别', '钙化', '形态', '边缘边界', '囊变坏死', '周围情况', '增强情况', '偏侧性']
        _X = _pd.DataFrame()
        for _c in _NUM_F:
            _X[_c] = _pd.to_numeric(_df[_c], errors='coerce')
        for _c in _CAT_F:
            _X[_c] = _df[_c].astype(str).astype('category').cat.codes
        _X = _X.fillna(_X.median())
        _Xs = _SS().fit(_X.values).transform(_X.values)
        for _i, _row in _df.iterrows():
            _pid = int(_row['影像号'])
            _struct_map[_pid] = _torch.tensor(_Xs[_i], dtype=_torch.float)
            _size_map[_pid] = _bst({'长径mm': _row.get('长径mm'), '短径mm': _row.get('短径mm')})
        print(f"  加载结构化征象: {len(_struct_map)} 患者, 尺寸文本: {len(_size_map)} 患者")

    def _struct_lookup(pid):
        return _struct_map.get(pid, _torch.zeros(19))

    def _size_lookup(pid):
        return _size_map.get(pid, "大小未测")

    # ---- 结构化分类器 (11类, 已融合热力读数: 19维征象 + 11维区域热力) ----
    import joblib as _jl
    _cls_model, _cls_scaler = None, None
    _cls_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'classification')
    try:
        _cls_model = _jl.load(os.path.join(_cls_dir, 'class11_model.joblib'))
        _cls_scaler = _jl.load(os.path.join(_cls_dir, 'scaler.joblib'))
        print('  已加载结构化分类器 (11类, 19维征象)')
    except Exception as _e:
        print(f'  结构化分类器加载失败 ({_e})')
    _CLS_EN = ['thymoma', 'thymic_carcinoma', 'cyst', 'teratoma', 'lymphoma',
               'germ_cell_tumor', 'neuroendocrine_tumor', 'hyperplasia',
               'benign_lesion', 'metastasis', 'other_malignant']
    _CLS_CN = ['胸腺瘤', '胸腺癌', '良性囊肿', '畸胎瘤', '淋巴瘤',
               '生殖细胞肿瘤', '神经内分泌肿瘤', '胸腺增生', '良性病变', '转移瘤', '其他恶性']

    # ---- 融合分类器 (19维 + 热力11维, 探针验证 84.3%/58.3%) ----
    _fus_model, _fus_scaler = None, None
    try:
        _fus_model = _jl.load(os.path.join(_cls_dir, 'class11_fusion_model.joblib'))
        _fus_scaler = _jl.load(os.path.join(_cls_dir, 'scaler_fusion.joblib'))
        print('  已加载融合分类器 (19维+热力11维, 84.3%/58.3%)')
    except Exception as _e:
        print(f'  融合分类器未找到 ({_e}), 用 19 维分类器')

    # 原始 19 维特征表 (分类器自带 scaler, 与训练一致)
    _raw_map = {}
    if _csv_path and os.path.exists(_csv_path):
        _df2 = _pd.read_csv(_csv_path)
        _df2['影像号'] = _df2['影像号'].astype(int)
        for _i, _row in _df2.iterrows():
            _pid = int(_row['影像号'])
            _raw_map[_pid] = _X.iloc[_i].values  # _X 已在上面构建 (19维, fillna后)

    def _heat11_from_amap(amap_14):
        """14×14 热力图 → 9维区域分布 + 2维强度 (与探针同逻辑)"""
        c = np.zeros(9)
        for i in range(14):
            for j in range(14):
                c[min(i * 3 // 14, 2) * 3 + min(j * 3 // 14, 2)] += amap_14[i, j]
        c = c / (c.sum() + 1e-8)
        return np.concatenate([c, [float(np.mean(amap_14)), float(np.max(amap_14))]])

    def _classify(pid, heat_amap=None):
        """返回 (英文类名, 概率, 中文提示)
        heat_amap: [14,14] 热力图 (若有 → 优先用融合分类器)"""
        if pid not in _raw_map:
            return None, None, ""
        base = _raw_map[pid].reshape(1, -1)
        # 融合分类器优先 (有热力图时)
        if heat_amap is not None and _fus_model is not None:
            _v = _fus_scaler.transform(np.hstack([base, _heat11_from_amap(heat_amap).reshape(1, -1)]))
            _probs = _fus_model.predict_proba(_v)[0]
            _k = int(_probs.argmax())
            return _CLS_EN[_k], float(_probs[_k]), f"分类参考：{_CLS_CN[_k]}。"
        if _cls_model is not None:
            _v = _cls_scaler.transform(base)
            _probs = _cls_model.predict_proba(_v)[0]
            _k = int(_probs.argmax())
            return _CLS_EN[_k], float(_probs[_k]), f"分类参考：{_CLS_CN[_k]}。"
        return None, None, ""

    # 读 split JSON 获取测试集路径→患者ID映射
    split_path = os.path.join(data_dir, f"split_{cfg.DATASET.NAME}.json")
    test_paths = []
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            split = _json.load(f)
        test_paths = [item[0] for item in split.get("test", [])]
        print(f"  测试集 {len(test_paths)} 个样本")

    test_loader = trainer.test_loader
    trainer.model.eval()

    # ---- 预扫描: 每患者选代表切片 ----
    # 首选: _rep.txt 标记 (软组织最多层 = 病灶最大层, 与分类/生成训练完全一致)
    # 回退: 增强期 + z 最大 (无 _rep.txt 的数据集, 如 BUSI: 逐张处理)
    chosen_indices = set()   # 被选中的切片下标
    rep_names = {}           # {basename: 1} 所有 _rep.txt 标记的切片
    for dp, _, fns in os.walk(data_dir):
        for f in fns:
            if f.endswith('_rep.txt'):
                with open(os.path.join(dp, f), encoding='utf-8') as fp:
                    rep_names[fp.read().strip()] = 1
    if rep_names:
        for idx, path in enumerate(test_paths):
            if os.path.basename(path) in rep_names:
                chosen_indices.add(idx)
        print(f"  测试集代表切片: {len(chosen_indices)} 个患者 (_rep.txt 病灶最大层, 与训练一致)")
    else:
        # 无 _rep.txt: 回退 增强期+z最大; 再无 → 逐张处理
        best_slice = {}
        patient_named = False
        for idx, path in enumerate(test_paths):
            m = re.match(r'P(\d+)_(\S+)_z(\d+)\.png', os.path.basename(path))
            if not m:
                continue
            patient_named = True
            pid = int(m.group(1))
            seq = m.group(2)
            z = int(m.group(3))
            key = (1 if "增强" in seq else 0, z)
            if pid not in best_slice or key > best_slice[pid][0]:
                best_slice[pid] = (key, idx)
        if patient_named:
            for pid, (_, idx) in best_slice.items():
                chosen_indices.add(idx)
            print(f"  测试集代表切片: {len(best_slice)} 个患者 (增强期肿瘤最大层)")
        else:
            chosen_indices = set(range(len(test_paths)))
            print(f"  测试集: {len(test_paths)} 张图像 (无患者命名, 逐张处理)")

    patient_idx = 0
    slice_idx = 0
    correct_total = 0
    total_samples = 0
    all_results = []
    all_labels = []
    all_ref_texts = []
    _gen_cache = [None]  # GEN_MODE 词表缓存

    with torch.no_grad():
        for batch in test_loader:
            image = batch["img"].to(trainer.device)
            label = batch["label"]
            outputs = trainer.model(image)

            B = image.shape[0]
            for i in range(B):
                # 只处理代表切片, 其余跳过 (一份报告/患者)
                if slice_idx not in chosen_indices:
                    slice_idx += 1
                    continue
                slice_idx += 1

                # 匹配真实报告 (从代表切片的路径解析患者 ID)
                ref_text = ""
                path = test_paths[slice_idx - 1]
                m = re.match(r'P(\d+)_', os.path.basename(path))
                if m:
                    pid = int(m.group(1))
                    ref_text = real_refs.get(pid, "")

                # 模型 eval 输出: (s_img, anomaly_scores, masked_image)
                s_img_b = outputs[0]          # [B]
                amap_b = outputs[1]           # [B, 196]
                masked_b = outputs[2]         # [B, 3, 224, 224] or None

                # ---- 预测类别: 融合分类器 (19维征象 + 热力区域读数) ----
                _amap_i = amap_b[i].reshape(14, 14).cpu().numpy() if amap_b is not None else None
                _cls_en, _cls_prob, _cls_hint = _classify(pid, _amap_i)
                if _cls_en is not None:
                    top_class = _cls_en
                    top_prob = _cls_prob
                else:
                    top_class = "未知"
                    top_prob = 0.0
                s_img_val = s_img_b[i]
                if isinstance(s_img_val, torch.Tensor):
                    s_img_val = s_img_val.item() if s_img_val.numel() == 1 else s_img_val.mean().item()

                def _lvl(x):
                    if x >= 0.75: return "显著异常"
                    if x >= 0.55: return "较显著异常"
                    if x >= 0.35: return "中度异常"
                    if x >= 0.15: return "轻度异常"
                    return "未见异常"

                result = {
                    "report_id": f"AI-US-P{patient_idx+1:04d}",
                    "patient_id": f"P{patient_idx+1:04d}",
                    "report_time": "",
                    "raw_text": "",
                    "polished_text": None,
                    "is_polished": False,
                    "findings": {
                        "top_class": top_class,
                        "top_probability": round(top_prob, 4),
                        "confidence": _lvl(top_prob) if top_prob >= 0.85 else "较高" if top_prob >= 0.7 else "中等",
                        "anomaly_level": _lvl(s_img_val),
                        "anomaly_score": round(s_img_val, 4),
                        "anomaly_concepts": [],
                        "normal_concepts": [],
                    },
                }

                # ---- GEN_MODE: 用 ReportGenNet 生成中文报告 (唯一的文本来源) ----
                gen_mode = getattr(cfg.TRAINER.ANOMALY_DETECT, 'GEN_MODE', False)
                if patient_idx == 0:
                    print(f'[DEBUG] gen_mode={gen_mode} has_report_gen={hasattr(trainer.model, "report_gen")}')
                if gen_mode and hasattr(trainer.model, 'report_gen'):
                    gen_backend = getattr(cfg.TRAINER.ANOMALY_DETECT, 'GEN_BACKEND', 'qwen')

                    # --- ① 先用生成器写报告文本 (qwen / lstm 接口不同) ---
                    if gen_backend == 'qwen':
                        # 全图 patch (去掩膜) + 分类提示 + 尺寸文本 + 结构化征象 → 报告
                        full_patches = trainer.model._extract_patch_tokens(image[i:i+1])[1]
                        struct = _struct_lookup(pid)   # [19] 标准化征象
                        _size_text = (_cls_hint or "") + _size_lookup(pid)
                        gen_texts = trainer.model.report_gen.generate(
                            full_patches, [_size_text],
                            struct.unsqueeze(0).to(trainer.device))
                    else:
                        masked_img = masked_b[i:i+1] if masked_b is not None else image[i:i+1]
                        masked_feat, masked_patches = trainer.model._extract_patch_tokens(masked_img)
                        masked_feat_norm = masked_feat / masked_feat.norm(dim=-1, keepdim=True)
                        if _gen_cache[0] is None:
                            import pickle as _pkl
                            vp = getattr(cfg.TRAINER.ANOMALY_DETECT, 'VOCAB_PATH', 'data/Thymoma/vocab.pkl')
                            with open(vp, 'rb') as f:
                                _gen_cache[0] = _pkl.load(f)
                        gen_texts = trainer.model.report_gen.generate(
                            masked_feat_norm, _gen_cache[0],
                            max_sentences=getattr(cfg.TRAINER.ANOMALY_DETECT, 'S_MAX', 8),
                            max_words=getattr(cfg.TRAINER.ANOMALY_DETECT, 'N_MAX', 50))
                    gen_text = gen_texts[0] if gen_texts else ""
                    # 清理 tokenizer 解码残渣 (U+FFFD 等, GBK 日志写不了)
                    gen_text = gen_text.replace('�', '').strip()
                    if gen_text:
                        result["raw_text"] = f"医学影像AI辅助分析报告\n\n{gen_text}\n\n免责声明:\n本报告由AI辅助分析系统自动生成，仅供临床参考。"
                        result["gen_text"] = gen_text

                    # --- ② 从生成文本提取 CE 征象 (否定过滤关键词匹配, CT 纵隔版) ---
                    finding_order = ["tumor_mass", "enhancement", "necrosis", "calcification",
                                     "cystic_change", "lobulation", "fat_density",
                                     "lymphadenopathy", "pleural_effusion"]
                    finding_keywords = {
                        "tumor_mass": ["肿块","占位","结节","肿物","软组织影","团块","病灶"],
                        "enhancement": ["强化","增强"],
                        "necrosis": ["坏死","液化"],
                        "calcification": ["钙化","钙灶"],
                        "cystic_change": ["囊肿","囊性","囊变","囊状","液性"],
                        "lobulation": ["分叶"],
                        "fat_density": ["脂肪密度","脂性","脂质"],
                        "lymphadenopathy": ["淋巴结"],
                        "pleural_effusion": ["胸腔积液","积液","胸水"],
                    }
                    negation_words = ["未见","未发现","无明显","未见明确","未见明显","排除",
                                      "不除外","不考虑","未见异常","无异常","无明确","未见肿物","无肿物"]
                    finding_cn_map = {
                        "tumor_mass": "实性占位征象","enhancement":"强化征象","necrosis":"坏死液化征象",
                        "calcification":"钙化灶征象","cystic_change":"囊变征象","lobulation":"分叶征象",
                        "fat_density":"脂肪密度征象","lymphadenopathy":"淋巴结征象","pleural_effusion":"胸腔积液征象"}
                    norm_names = set()
                    anom_c, norm_c = [], []
                    for fn in finding_order:
                        kws = finding_keywords.get(fn, [])
                        level = "未见"
                        for kw in kws:
                            idx = result["raw_text"].find(kw)
                            if idx >= 0:
                                prefix = result["raw_text"][max(0,idx-15):idx]
                                if any(n in prefix for n in negation_words):
                                    continue
                                level = "显著"  # 找到即显著
                                break
                        item = {"name":finding_cn_map.get(fn,fn), "level":level, "desc":"文本匹配"}
                        if fn in norm_names:
                            norm_c.append(item)
                        else:
                            anom_c.append(item)
                    result["findings"]["anomaly_concepts"] = anom_c
                    result["findings"]["normal_concepts"] = norm_c

                true_label = classnames[label[i].item()]
                all_results.append(result)
                all_labels.append(true_label)
                all_ref_texts.append(ref_text)

                pred_label = result["findings"]["top_class"]
                total_samples += 1
                if pred_label == true_label:
                    correct_total += 1

                report_path = os.path.join(report_dir, f"P{patient_idx+1:04d}_超声报告.txt")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(result["raw_text"])

                patient_idx += 1

                if patient_idx <= 3:
                    print(f"\n{'─'*64}")
                    print(f"  患者 {patient_idx} / 真实标签: {true_label} / 预测: {pred_label}")
                    if ref_text:
                        print(f"  参考报告(前80字): {ref_text[:80]}...")
                    print(f"{'─'*64}")
                    print(result["raw_text"])

    # 打印汇总
    acc = correct_total / total_samples * 100 if total_samples > 0 else 0
    print(f"\n{'═'*64}")
    print(f"  报告生成完成")
    print(f"  总计: {total_samples} 份报告")
    print(f"  预测正确: {correct_total}/{total_samples} ({acc:.1f}%)")
    print(f"  存放路径: {report_dir}")
    print(f"{'═'*64}")

    # ---- 评分（PromptMRG 风格）----
    from trainers.AnomalyDetect.report_scorer import ReportScorer
    scorer = ReportScorer()
    has_real_refs = any(t for t in all_ref_texts)
    metrics = scorer.score_batch(all_results, all_labels,
        reference_texts=all_ref_texts if has_real_refs else None)

    # 打印简要汇总
    print(f"\n  CE: P={metrics['ce_precision']:.3f}  R={metrics['ce_recall']:.3f}  F1={metrics['ce_f1']:.3f}")
    print(f"  NLG: BLEU-1={metrics['BLEU_1']:.1f}  BLEU-4={metrics['BLEU_4']:.1f}  ROUGE_L={metrics['ROUGE_L']:.1f}")
    print(f"  Accuracy: {metrics['accuracy']:.1f}%  (N={metrics['count']})")

    # 保存所有评分数据 (dataset_name + 真实参考报告)
    score_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "output", "evaluation")
    scorer.save(all_results, all_labels, metrics, score_dir,
                method_name="AnomalyCLIP", dataset_name=cfg.DATASET.NAME,
                reference_texts=all_ref_texts)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="医学超声影像报告生成")
    parser.add_argument("--demo", action="store_true", default=True,
                        help="使用模拟数据演示（默认）")
    parser.add_argument("--config-file", type=str, default="",
                        help="方法配置文件路径")
    parser.add_argument("--dataset-config-file", type=str, default="",
                        help="数据集配置文件路径")
    parser.add_argument("--model-dir", type=str, default="",
                        help="模型检查点目录（用于真实推理）")
    parser.add_argument("--image-path", type=str, default="",
                        help="待分析的图像路径")
    args = parser.parse_args()

    # 默认使用演示模式
    if args.model_dir:
        real_mode(args.config_file, args.dataset_config_file,
                  args.model_dir, args.image_path)
    else:
        demo_mode()
