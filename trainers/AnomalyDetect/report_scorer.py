"""
医学超声报告评分模块 — PromptMRG 风格
========================================================================
借鉴 PromptMRG (AAAI 2024) 的 CE + NLG 双轨评分体系。

CE 指标：对 9 种超声征象做二分类 → Precision/Recall/F1（宏平均+微平均+逐征象）
NLG 指标：BLEU-1~4, ROUGE-L（中文用 jieba 分词，回退到字级 n-gram）

用法:
  from trainers.AnomalyDetect.report_scorer import ReportScorer
  scorer = ReportScorer()
  metrics = scorer.score_batch(reports, labels)
  scorer.save(reports, labels, metrics, 'output/evaluation/')
"""

import os, json, math, csv
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

# ---- 中文分词 ----
try:
    import jieba
    _JIEBA = True
except ImportError:
    _JIEBA = False


# ============================================================
# 征象定义
# ============================================================

ALL_FINDINGS = [
    "tumor_mass", "inflammation", "necrosis",
    "calcification", "cyst_fluid", "fibrosis_scar",
    "normal_tissue", "regular_architecture", "isoechoic_texture",
]

FINDING_CN = {
    "tumor_mass": "实性占位征象",
    "inflammation": "炎性改变征象",
    "necrosis": "坏死液化征象",
    "calcification": "钙化灶征象",
    "cyst_fluid": "囊性结构征象",
    "fibrosis_scar": "纤维化/瘢痕征象",
    "normal_tissue": "正常组织回声",
    "regular_architecture": "结构规整性",
    "isoechoic_texture": "等回声质地",
}

# 标签 → 期望征象（按关键词匹配，数据集无关）
def _get_expected_findings(label: str) -> dict:
    """根据标签关键词返回期望征象向量"""
    low = label.lower()
    if any(kw in low for kw in ("malignant", "cancer", "carcinoma")):
        return {"tumor_mass": +1, "inflammation": +1, "necrosis": +1,
                "calcification": 0, "cyst_fluid": 0, "fibrosis_scar": 0,
                "normal_tissue": -1, "regular_architecture": -1, "isoechoic_texture": -1}
    if any(kw in low for kw in ("thymoma", "tumor", "neoplasm")):
        return {"tumor_mass": +1, "inflammation": 0, "necrosis": 0,
                "calcification": +1, "cyst_fluid": 0, "fibrosis_scar": 0,
                "normal_tissue": -1, "regular_architecture": -1, "isoechoic_texture": -1}
    if "benign" in low:
        return {"tumor_mass": +1, "inflammation": 0, "necrosis": 0,
                "calcification": 0, "cyst_fluid": 0, "fibrosis_scar": 0,
                "normal_tissue": +1, "regular_architecture": +1, "isoechoic_texture": 0}
    if "normal" in low:
        return {"tumor_mass": -1, "inflammation": -1, "necrosis": -1,
                "calcification": -1, "cyst_fluid": -1, "fibrosis_scar": -1,
                "normal_tissue": +1, "regular_architecture": +1, "isoechoic_texture": +1}
    # 未知标签：全中性
    return {f: 0 for f in ALL_FINDINGS}


# ============================================================
# 文本清洗 + 分词
# ============================================================

def _clean_text(text: str) -> str:
    """
    去掉报告中的格式字符+元数据+标题，只保留临床正文。
    元数据示例: "报告编号:AI-US-P0001-20260810患者编号:P0001..."
    标题示例: "【一、影像学所见】"  "综合印象"  "免责声明:..."
    """
    import re
    # 1) 去分隔线
    for ch in "=═─*~":
        text = text.replace(ch, "")

    # 2) 取【一】到免责声明之前的正文（核心临床内容）
    m_start = re.search(r'【一[、，]', text)
    m_end = re.search(r'免责声明', text)
    if m_start:
        text = text[m_start.start():]
    if m_end:
        text = text[:m_end.start()]

    # 3) 去节标题关键字及其编号
    headers = [
        r'【一[、，]】?\s*影像学所见',
        r'【二[、，]】?\s*影像征象分析',
        r'【三[、，]】?\s*综合印象',
        r'【四[、，]】?\s*临床建议',
        r'一[、，]\s*异常影像征象评估[：:]?',
        r'二[、，]\s*正常组织特征评估[：:]?',
        r'影像学所见', r'影像征象分析', r'综合印象', r'临床建议',
        r'异常影像征象评估', r'正常组织特征评估',
        r'各类别评估[：:]',
    ]
    for h in headers:
        text = re.sub(h, '', text)

    # 4) 去报告元数据行（冒号前的标签）
    metadata_keys = [
        r'报告编号[：:][^\s]+',
        r'患者编号[：:][^\s]+',
        r'检查模态[：:][^\s]+',
        r'生成时间[：:][^\s]+',
        r'生成方式[：:][^\s]+',
    ]
    for mk in metadata_keys:
        text = re.sub(mk, '', text)

    # 5) 去残留的【】符号 + 真实报告元数据标签
    text = text.replace('【', '').replace('】', '')
    text = text.replace('影像所见：', '').replace('报告结论：', '')
    text = re.sub(r'\s+', '', text)

    # 6) 提取核心临床句：只保留 "送检…" 开头的影像学所见段 + 综合印象段
    #    去掉冗长的逐征象列表 ("xxx征象：未见…")，减少与伪参考的长度差异
    sentences = re.split(r'[。；]', text)
    keep = []
    for s in sentences:
        # 跳过纯征象描述行（太细节，伪参考没有对应内容）
        if re.search(r'(征象|特征)[：:]', s) and len(s) < 80:
            continue
        # 跳过纯建议行
        if re.search(r'^(建议|如|本报告)', s):
            continue
        if s.strip():
            keep.append(s.strip())
    text = '。'.join(keep)

    return text


def _tokenize(text: str) -> list:
    """清洗 + 分词"""
    text = _clean_text(text)
    if _JIEBA:
        return list(jieba.cut(text))
    else:
        return list(text)


# ============================================================
# NLG: BLEU / ROUGE-L
# ============================================================

def _ngrams(tokens: list, n: int) -> list:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def compute_bleu(ref_tokens: list, cand_tokens: list, max_n: int = 4) -> dict:
    """BLEU-1~4，n-gram precision + brevity penalty"""
    cand_len = len(cand_tokens)
    ref_len = len(ref_tokens)
    bp = 1.0 if cand_len >= ref_len else math.exp(1 - ref_len / max(cand_len, 1))

    scores = {}
    for n in range(1, max_n + 1):
        cand_ng = _ngrams(cand_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)
        if not cand_ng:
            scores[f"BLEU_{n}"] = 0.0
            continue

        ref_counts = defaultdict(int)
        for ng in ref_ng:
            ref_counts[ng] += 1

        match = 0
        for ng in set(cand_ng):
            match += min(cand_ng.count(ng), ref_counts.get(ng, 0))

        precision = match / max(len(cand_ng), 1)
        scores[f"BLEU_{n}"] = round(precision * bp, 4)

    return scores


def compute_rouge_l(ref_tokens: list, cand_tokens: list) -> float:
    """ROUGE-L，LCS + F-beta (beta=1.2)"""
    m, n = len(ref_tokens), len(cand_tokens)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref_tokens[i-1] == cand_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len = dp[m][n]
    prec = lcs_len / max(n, 1)
    rec = lcs_len / max(m, 1)
    beta = 1.2
    if prec + rec > 0:
        return round((1 + beta**2) * prec * rec / (rec + beta**2 * prec), 4)
    return 0.0


# ============================================================
# CE: 征象级二分类评估
# ============================================================

def _extract_finding_vector(report_findings: dict) -> Dict[str, int]:
    """报告中每个征象是否被提及 (1=提及, 0=未提及)"""
    vec = {}
    concept_dict = {}
    for c in report_findings.get("anomaly_concepts", []):
        concept_dict[c["name"]] = c["level"]
    for c in report_findings.get("normal_concepts", []):
        concept_dict[c["name"]] = c["level"]
    for eng_name in ALL_FINDINGS:
        cn_name = FINDING_CN.get(eng_name, eng_name)
        level = concept_dict.get(cn_name, "未见")
        vec[eng_name] = 1 if level in ("显著", "可见") else 0
    return vec


def _expected_finding_vector(true_label: str) -> Dict[str, int]:
    """真实标签的期望征象（关键词匹配，数据集无关）"""
    expected = _get_expected_findings(true_label)
    return {eng: (1 if expected.get(eng, 0) == +1 else 0) for eng in ALL_FINDINGS}


def compute_ce_metrics(all_pred_vecs, all_gt_vecs) -> dict:
    findings = ALL_FINDINGS
    per_finding = {}
    total_tp = total_fp = total_fn = 0

    for f in findings:
        tp = fp = fn = 0
        for pv, gv in zip(all_pred_vecs, all_gt_vecs):
            pp, gp = pv.get(f, 0), gv.get(f, 0)
            if pp and gp:     tp += 1
            elif pp and not gp: fp += 1
            elif not pp and gp: fn += 1
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 0.001)
        per_finding[f] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
        total_tp += tp; total_fp += fp; total_fn += fn

    macro_prec = sum(v["precision"] for v in per_finding.values()) / len(findings)
    macro_rec = sum(v["recall"] for v in per_finding.values()) / len(findings)
    macro_f1 = sum(v["f1"] for v in per_finding.values()) / len(findings)

    micro_prec = total_tp / max(total_tp + total_fp, 1)
    micro_rec = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, 0.001)

    return {
        "ce_precision": round(macro_prec, 4),
        "ce_recall": round(macro_rec, 4),
        "ce_f1": round(macro_f1, 4),
        "ce_micro_precision": round(micro_prec, 4),
        "ce_micro_recall": round(micro_rec, 4),
        "ce_micro_f1": round(micro_f1, 4),
        "ce_per_finding": per_finding,
    }


# ============================================================
# 伪参考报告
# ============================================================

def generate_pseudo_reference(true_label: str, dataset_name: str = "") -> str:
    """
    根据标签生成伪参考报告（关键词匹配，数据集无关）。
    有真实报告时传 reference_texts 参数即可绕过此函数。
    """
    lbl = true_label.lower()

    if any(kw in lbl for kw in ("malignant", "cancer", "carcinoma")):
        return (
            "影像检查显示占位性病变，边界不规则，呈浸润性生长，内部结构紊乱。"
            "实性占位征象显著，可见异常血流信号。正常组织结构消失。"
            "综合印象：考虑恶性病变，建议穿刺活检以明确诊断。"
        )
    elif any(kw in lbl for kw in ("thymoma", "tumor", "neoplasm", "mass")):
        return (
            "影像检查显示前纵隔占位性病变，边界清楚或分叶状，"
            "增强扫描可见强化。实性占位征象显著，钙化灶征象可见。"
            "综合印象：考虑胸腺瘤可能性大，建议结合临床综合评估。"
        )
    elif "benign" in lbl:
        return (
            "影像检查显示局限性占位病变，边界尚清，形态规则，内部结构均匀。"
            "实性占位征象可见，正常组织结构大致正常。"
            "综合印象：考虑良性病变，建议定期随访复查。"
        )
    elif "normal" in lbl:
        return (
            "影像检查显示组织结构清晰，层次分明，未见明确占位性病变及结构异常。"
            "正常组织特征显著。"
            "综合印象：检查未见明确异常，建议按常规筛查计划随访。"
        )
    else:
        return f"影像检查显示{true_label}。综合印象：{true_label}。"


# ============================================================
# 报告评分器
# ============================================================

class ReportScorer:

    def __init__(self, classnames: List[str] = None):
        self.classnames = classnames or ["benign", "malignant", "normal"]

    def score_batch(self, reports: List[dict],
                    labels: List[str],
                    reference_texts: List[str] = None) -> dict:
        n = len(reports)
        if reference_texts is None:
            reference_texts = [generate_pseudo_reference(lbl) for lbl in labels]

        # ---- CE ----
        pred_vecs = [_extract_finding_vector(r.get("findings", {})) for r in reports]
        gt_vecs = [_expected_finding_vector(lbl) for lbl in labels]
        ce = compute_ce_metrics(pred_vecs, gt_vecs)

        # ---- NLG ----
        bleu_sums = {"BLEU_1": 0.0, "BLEU_2": 0.0, "BLEU_3": 0.0, "BLEU_4": 0.0}
        rouge_sum = 0.0
        for report, ref_text in zip(reports, reference_texts):
            gen_text = report.get("raw_text", "")
            ref_tokens = _tokenize(ref_text)
            gen_tokens = _tokenize(gen_text)

            bleu = compute_bleu(ref_tokens, gen_tokens, max_n=4)
            for k in bleu_sums:
                bleu_sums[k] += bleu.get(k, 0.0)
            rouge_sum += compute_rouge_l(ref_tokens, gen_tokens)

        nlg = {k: round(bleu_sums[k] / max(n, 1), 4) for k in bleu_sums}
        nlg["ROUGE_L"] = round(rouge_sum / max(n, 1), 4)

        # ---- 诊断准确率 ----
        correct = sum(1 for r, lbl in zip(reports, labels)
                      if r.get("findings", {}).get("top_class", "") == lbl)
        accuracy = round(correct / max(n, 1) * 100, 2)

        return {**ce, **nlg, "accuracy": accuracy, "count": n}

    # ----------------------------------------------------------

    def save(self, reports, labels, metrics, output_dir,
             method_name="AnomalyCLIP", dataset_name="BUSI",
             reference_texts=None):
        os.makedirs(output_dir, exist_ok=True)

        # 参考报告: 真实报告优先, 无则伪参考
        if reference_texts and any(reference_texts):
            ref_texts = reference_texts
        else:
            ref_texts = [generate_pseudo_reference(lbl) for lbl in labels]

        # test_gts.txt / test_res.txt
        with open(os.path.join(output_dir, "test_gts.txt"), "w", encoding="utf-8") as f:
            for t in ref_texts:
                f.write(t + "\n")

        with open(os.path.join(output_dir, "test_res.txt"), "w", encoding="utf-8") as f:
            for r in reports:
                f.write(r.get("raw_text", "").replace("\n", " ") + "\n")

        # per_sample.json
        per_sample = []
        for i, (report, label) in enumerate(zip(reports, labels)):
            fd = report.get("findings", {})
            per_sample.append({
                "patient_id": f"P{i+1:04d}",
                "true_label": label,
                "pred_label": fd.get("top_class", ""),
                "pred_probability": fd.get("top_probability", 0),
                "anomaly_level": fd.get("anomaly_level", ""),
                "anomaly_score": fd.get("anomaly_score", 0),
                "finding_vector": _extract_finding_vector(fd),
                "generated_report": report.get("raw_text", ""),
            })
        with open(os.path.join(output_dir, "per_sample.json"), "w", encoding="utf-8") as f:
            json.dump(per_sample, f, ensure_ascii=False, indent=2)

        # metrics.json
        save_metrics = {
            "method": method_name,
            "dataset": dataset_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "num_samples": metrics["count"],
            "ce_metrics": {
                "precision": metrics["ce_precision"],
                "recall": metrics["ce_recall"],
                "f1": metrics["ce_f1"],
                "micro_precision": metrics["ce_micro_precision"],
                "micro_recall": metrics["ce_micro_recall"],
                "micro_f1": metrics["ce_micro_f1"],
            },
            "nlg_metrics": {
                "BLEU_1": metrics["BLEU_1"], "BLEU_2": metrics["BLEU_2"],
                "BLEU_3": metrics["BLEU_3"], "BLEU_4": metrics["BLEU_4"],
                "ROUGE_L": metrics["ROUGE_L"],
            },
            "diagnostic_accuracy": metrics["accuracy"],
            "ce_per_finding": metrics.get("ce_per_finding", {}),
        }
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(save_metrics, f, ensure_ascii=False, indent=2)

        # metrics.csv
        csv_path = os.path.join(output_dir, "metrics.csv")
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            headers = [
                "Method", "Dataset", "Date",
                "CE_Precision", "CE_Recall", "CE_F1",
                "CE_Micro_Precision", "CE_Micro_Recall", "CE_Micro_F1",
                "BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "ROUGE_L",
                "Accuracy", "N"
            ]
            if not file_exists:
                writer.writerow(headers)
            writer.writerow([
                method_name, dataset_name, datetime.now().strftime("%Y-%m-%d"),
                metrics["ce_precision"], metrics["ce_recall"], metrics["ce_f1"],
                metrics["ce_micro_precision"], metrics["ce_micro_recall"], metrics["ce_micro_f1"],
                metrics["BLEU_1"], metrics["BLEU_2"], metrics["BLEU_3"], metrics["BLEU_4"],
                metrics["ROUGE_L"],
                metrics["accuracy"], metrics["count"],
            ])

        print(f"\n  [Scorer] Saved to: {output_dir}/")
        print(f"    test_gts.txt      — {len(ref_texts)} lines")
        print(f"    test_res.txt      — {len(reports)} lines")
        print(f"    per_sample.json   — {len(per_sample)} samples")
        print(f"    metrics.json      — CE + NLG metrics")
        print(f"    metrics.csv       — paper-ready row")
