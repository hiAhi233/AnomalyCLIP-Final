"""
医学超声影像报告生成器
========================================================================
将 AnomalyCLIP 推理输出转化为结构化中文医学报告。

核心设计（借鉴 HepaPathGPT 的桥梁模式）：
  模型输出(连续值/向量) → [1] 量化分级 → [2] 医学描述映射 → [3] 模板填充

LLM 润色接口:
  - get_llm_polish_prompt(): 输出结构化发现 + 润色指令，可发送给大模型
  - apply_llm_polish():      接收大模型润色后的报告文本

Author: Generated via Claude Code
"""

from datetime import datetime
import torch
import torch.nn.functional as F


# ============================================================
# Layer 1 & 2: 量化分级 + 医学描述映射
# ============================================================

# ---- 异常程度分级 ----
ANOMALY_THRESHOLDS = [
    (0.75, "显著异常"),
    (0.55, "较显著异常"),
    (0.35, "中度异常"),
    (0.15, "轻度异常"),
    (0.00, "未见异常"),
]

# ---- 异常程度 → 完整医学描述（影像学所见，通用模板） ----
ANOMALY_IMAGING_DESC = {
    "显著异常": (
        "影像检查显示明显不规则占位性病变，边界模糊，呈浸润性生长表现，"
        "内部结构紊乱，高度怀疑恶性病变。"
        "病变区域与周围正常组织分界不清。"
    ),
    "较显著异常": (
        "影像检查显示形态不规则占位性病变，边界欠清晰，"
        "内部结构不均匀，需警惕恶性可能。"
    ),
    "中度异常": (
        "影像检查显示不规则异常区域，范围中等，边界部分欠清晰，"
        "内部结构欠均匀。建议进一步影像学评估。"
    ),
    "轻度异常": (
        "影像检查显示局限性异常改变，范围较小，边界尚清，"
        "形态规则。考虑良性改变可能，建议随访观察。"
    ),
    "未见异常": (
        "影像检查显示组织结构层次清晰，未见明确占位性病变及结构异常。"
    ),
}

# ---- 异常程度 → 综合印象 ----
ANOMALY_IMPRESSION = {
    "显著异常": "高度怀疑恶性病变，建议尽快行穿刺活检明确诊断。",
    "较显著异常": "考虑可疑恶性病变，建议行穿刺活检或短期密切随访。",
    "中度异常": "性质待定占位性病变，建议结合其他影像学检查综合评估，必要时行穿刺活检。",
    "轻度异常": "良性改变可能，建议短期随访复查。",
    "未见异常": "影像检查未见明确异常，建议按常规筛查计划随访。",
}

# ---- 分类依据（BI-RADS 分级） ----
BIRADS_LEVELS = {
    2: "BI-RADS 2类：良性发现",
    3: "BI-RADS 3类：可能良性",
    4: "BI-RADS 4类：可疑恶性",
    5: "BI-RADS 5类：高度怀疑恶性",
}

# ---- 各影像征象（concept）→ 分级的完整医学描述 ----
CONCEPT_MEDICAL_DESC = {
    "tumor_mass": {
        "name_cn": "实性占位征象",
        "显著": "可见明确实性占位病变，形态不规则，对周围结构产生推挤效应。",
        "可见": "可见实性结节样异常区域，边界尚可辨认。",
        "不明显": "局部密度/回声轻度不均匀，未见明确占位效应。",
        "未见": "未见明确实性占位性病变征象。",
    },
    "inflammation": {
        "name_cn": "炎性改变征象",
        "显著": "病变区域可见明显炎性改变，伴组织水肿，符合急性炎症表现。",
        "可见": "局部可见炎性改变特征，组织轻度水肿。",
        "不明显": "局部密度/回声轻度改变，未见明确炎性特征。",
        "未见": "未见明确炎性改变征象。",
    },
    "necrosis": {
        "name_cn": "坏死液化征象",
        "显著": "占位病变内部可见大片不规则低密度/无回声区，符合坏死液化改变。",
        "可见": "病变内部可见小片状异常区域，考虑局灶性坏死或液化。",
        "不明显": "病变内部结构欠均匀，未见明确坏死液化区。",
        "未见": "病变内部结构均匀，未见坏死液化征象。",
    },
    "calcification": {
        "name_cn": "钙化灶征象",
        "显著": "病变内部及周边可见多发簇状钙化灶，呈砂砾样分布。",
        "可见": "病变内部可见散在点状高密度/强回声灶，提示钙化可能。",
        "不明显": "局部密度/回声略增强，未见明确钙化灶。",
        "未见": "未见明确钙化灶征象。",
    },
    "cyst_fluid": {
        "name_cn": "囊性结构征象",
        "显著": "可见边界清晰的含液囊性结构，壁薄光滑，符合囊肿表现。",
        "可见": "可见囊性结构，边界较清晰。",
        "不明显": "局部密度/回声偏低，未见明确囊性结构。",
        "未见": "未见明确囊性结构征象。",
    },
    "fibrosis_scar": {
        "name_cn": "纤维化/瘢痕征象",
        "显著": "实质内可见条索状纤维组织增生，呈放射状分布。",
        "可见": "局部可见线状或条索状纤维化改变。",
        "不明显": "局部实质略增强，未见明确纤维化改变。",
        "未见": "未见明确纤维化或瘢痕性改变征象。",
    },
}

# ---- 正常组织征象（concept）→ 分级的完整医学描述 ----
NORMAL_CONCEPT_MEDICAL_DESC = {
    "normal_tissue": {
        "name_cn": "正常组织回声/密度",
        "显著": "组织结构层次清晰，符合正常组织表现。",
        "可见": "大部分区域组织结构大致正常。",
        "不明显": "组织结构欠均匀，正常结构显示欠清。",
        "未见": "正常组织结构显示不清，被异常组织替代。",
    },
    "regular_architecture": {
        "name_cn": "结构规整性",
        "显著": "组织层次分明，结构规整，组织界面清晰。",
        "可见": "组织结构大致规整，层次可辨。",
        "不明显": "组织结构紊乱，层次显示不清。",
        "未见": "正常结构消失，被异常病变组织替代。",
    },
    "isoechoic_texture": {
        "name_cn": "质地均匀性",
        "显著": "组织呈均匀表现，符合正常质地。",
        "可见": "大部分区域质地均匀，局部略有差异。",
        "不明显": "质地不均匀，局部偏离正常范围。",
        "未见": "质地显著异常，与正常组织差异明显。",
    },
}

# ---- 可信度 → 描述 ----
CONFIDENCE_DESC = {
    "高":     "诊断可信度高，与典型影像学特征高度吻合。",
    "较高":   "诊断可信度较高，影像学特征较为典型。",
    "中等":   "诊断可信度中等，部分影像学特征不典型，建议结合其他检查综合判断。",
    "低":     "诊断可信度偏低，影像学特征不典型，需进一步检查明确。",
}

# ---- 临床建议模板（按严重程度，通用） ----
RECOMMENDATIONS = {
    "显著异常": [
        "鉴于影像学高度怀疑恶性病变，建议尽快行穿刺活检以明确病理诊断。",
        "建议行增强影像检查，进一步评估病变范围。",
        "建议结合临床查体及肿瘤标志物综合评估。",
    ],
    "较显著异常": [
        "建议行穿刺活检，或短期（1-3个月）密切随访复查。",
        "建议结合其他影像学检查进一步评估。",
        "建议结合肿瘤标志物及临床查体综合评估。",
    ],
    "中度异常": [
        "建议在3个月内复查，动态观察病变变化。",
        "建议结合其他影像学检查进一步定性。",
        "如随访期间病变增大或形态改变，应及时行穿刺活检。",
    ],
    "轻度异常": [
        "建议在3-6个月内随访复查，观察病变变化。",
        "如随访期间病变稳定或缩小，可延长复查间隔。",
        "如随访期间病变增大或形态改变，应及时进一步评估。",
    ],
    "未见异常": [
        "建议按常规筛查计划定期随访。",
        "保持健康生活方式。",
    ],
}

# ---- 免责声明 ----
DISCLAIMER = (
    "本报告由AI辅助分析系统自动生成，仅供临床参考，不能替代执业医师的诊断意见。"
    "最终诊断需由具有执业资质的超声医师或放射科医师结合完整临床资料（包括但不限于"
    "病史、体格检查、其他影像学检查及病理结果）综合判断。"
)


# ============================================================
# 辅助函数
# ============================================================

def _quantize_anomaly_score(s_img: float) -> str:
    """连续异常分数 → 离散医学分级"""
    for threshold, level in ANOMALY_THRESHOLDS:
        if s_img >= threshold:
            return level
    return "未见异常"


def _quantize_confidence(prob: float) -> str:
    """概率 → 可信度等级"""
    if prob >= 0.85:
        return "高"
    elif prob >= 0.70:
        return "较高"
    elif prob >= 0.50:
        return "中等"
    else:
        return "低"


# ============================================================
# 核心类
# ============================================================

class MedicalReportGenerator:
    """
    医学超声影像报告生成器

    将 AnomalyCLIP 推理输出转化为结构化中文医学报告，
    支持 LLM 润色接口。
    """

    def __init__(self, classnames: list, dataset_name: str = "BUSI",
                 modality: str = "乳腺超声"):
        """
        Args:
            classnames: 数据集类别名称列表，如 ['benign', 'malignant', 'normal']
            dataset_name: 数据集名称
            modality: 影像模态名称
        """
        self.classnames = classnames
        self.dataset_name = dataset_name
        self.modality = modality
        self.report_time = datetime.now()

        # 内部状态
        self._raw_outputs = None
        self._findings = {}          # 结构化发现
        self._raw_report_text = ""   # 模板生成的原始报告
        self._polished_text = None   # LLM 润色后的报告

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------

    def generate(self, outputs: dict, image_path: str = None,
                 patient_id: str = None) -> dict:
        """
        根据模型推理输出生成完整医学报告。

        Args:
            outputs: 模型 eval 模式 forward() 返回的元组或字典
                     (logits, s_img, anomaly_scores, concept_scores, masked_image)
            image_path: 原始图像路径（可选）
            patient_id: 患者编号（可选，默认自动生成）

        Returns:
            dict: {
                'report_id': str,
                'patient_id': str,
                'report_time': str,
                'raw_text': str,          # 模板生成的原始报告
                'polished_text': str|None, # LLM润色后报告（如有）
                'findings': dict,          # 结构化发现
                'is_polished': bool,
            }
        """
        self._raw_outputs = outputs
        self.report_time = datetime.now()

        # 解析输出
        if isinstance(outputs, (tuple, list)):
            logits, s_img, anomaly_scores, concept_scores, masked_image = outputs
        else:
            logits = outputs.get('logits')
            s_img = outputs.get('s_img')
            anomaly_scores = outputs.get('anomaly_scores')
            concept_scores = outputs.get('concept_scores')
            masked_image = outputs.get('masked_image')

        # 处理 batch 维度（取第一个样本）
        if logits.dim() == 2:
            logits = logits[0]
        if isinstance(s_img, torch.Tensor) and s_img.dim() >= 1:
            s_img = s_img[0].item() if s_img.numel() == 1 else s_img.mean().item()
        elif isinstance(s_img, torch.Tensor):
            s_img = s_img.item()
        if isinstance(anomaly_scores, torch.Tensor) and anomaly_scores.dim() >= 2:
            anomaly_scores = anomaly_scores[0]

        # --- [1] 量化分级 ---
        probs = F.softmax(logits, dim=-1)
        top_idx = probs.argmax(dim=-1).item()
        top_prob = probs[top_idx].item()
        top_class = self.classnames[top_idx] if top_idx < len(self.classnames) else "未知"

        anomaly_level = _quantize_anomaly_score(s_img)
        confidence = _quantize_confidence(top_prob)

        # 异常分数统计
        anomaly_mean = anomaly_scores.mean().item()
        anomaly_max = anomaly_scores.max().item()
        above_thresh = (anomaly_scores > 0.5).float().mean().item()

        # 概念征象推导 — 基于分类结果 + 异常分数倒推
        concept_findings_anomaly, concept_findings_normal = \
            self._derive_findings(top_class)

        # --- [2] 医学描述映射 ---
        imaging_desc = ANOMALY_IMAGING_DESC.get(anomaly_level, "")

        # 异常征象描述（按激活程度排序）
        anomaly_concept_descs = []
        sorted_anomaly = sorted(concept_findings_anomaly.items(),
                                key=lambda x: x[1]["value"], reverse=True)
        for cname, info in sorted_anomaly:
            cfg = CONCEPT_MEDICAL_DESC.get(cname, {})
            name_cn = cfg.get("name_cn", cname)
            desc = cfg.get(info["level"], "")
            if desc:
                anomaly_concept_descs.append({
                    "name": name_cn,
                    "level": info["level"],
                    "desc": desc,
                })

        # 正常组织征象描述
        normal_concept_descs = []
        sorted_normal = sorted(concept_findings_normal.items(),
                               key=lambda x: x[1]["value"], reverse=True)
        for cname, info in sorted_normal:
            cfg = NORMAL_CONCEPT_MEDICAL_DESC.get(cname, {})
            name_cn = cfg.get("name_cn", cname)
            desc = cfg.get(info["level"], "")
            if desc:
                normal_concept_descs.append({
                    "name": name_cn,
                    "level": info["level"],
                    "desc": desc,
                })

        # 综合印象
        impression = ANOMALY_IMPRESSION.get(anomaly_level, "")

        # 临床建议
        recs = RECOMMENDATIONS.get(anomaly_level, RECOMMENDATIONS["未见异常"])

        # --- 保存结构化发现 ---
        self._findings = {
            "top_class": top_class,
            "top_probability": round(top_prob, 4),
            "confidence": confidence,
            "anomaly_level": anomaly_level,
            "anomaly_score": round(s_img, 4),
            "anomaly_mean": round(anomaly_mean, 4),
            "anomaly_max": round(anomaly_max, 4),
            "lesion_coverage_ratio": round(above_thresh, 4),
            "anomaly_concepts": anomaly_concept_descs,
            "normal_concepts": normal_concept_descs,
            "impression": impression,
            "recommendations": recs,
        }

        # --- [3] 模板填充 ---
        report_id = f"AI-US-{patient_id or 'XXXX'}-{self.report_time.strftime('%Y%m%d')}"
        pid_display = patient_id or "未指定"

        # 分类结果映射为中文（自动从类名推断）
        class_cn = self._classname_to_cn(top_class)

        # 组装报告
        raw_text = self._compose_report(
            report_id=report_id,
            patient_id=pid_display,
            imaging_desc=imaging_desc,
            class_cn=class_cn,
            confidence=confidence,
            confidence_desc=CONFIDENCE_DESC.get(confidence, ""),
            top_prob=top_prob,
            anomaly_level=anomaly_level,
            anomaly_concept_descs=anomaly_concept_descs,
            normal_concept_descs=normal_concept_descs,
            impression=impression,
            recommendations=recs,
            classnames_cn=[self._classname_to_cn(c) for c in self.classnames],
            probs=probs.tolist(),
        )

        self._raw_report_text = raw_text
        self._polished_text = None

        return {
            "report_id": report_id,
            "patient_id": pid_display,
            "report_time": self.report_time.strftime("%Y年%m月%d日 %H:%M"),
            "raw_text": raw_text,
            "polished_text": None,
            "findings": self._findings,
            "is_polished": False,
        }

    # ----------------------------------------------------------
    # 通用辅助方法（数据集无关）
    # ----------------------------------------------------------

    @staticmethod
    def _classname_to_cn(name: str) -> str:
        """类名 → 中文（关键词匹配，不依赖硬编码映射表）"""
        low = name.lower()
        if "malignant" in low or "cancer" in low or "carcinoma" in low:
            return "恶性病变"
        if "benign" in low:
            return "良性病变"
        if "normal" in low:
            return "正常"
        if "thymoma" in low:
            return "胸腺瘤"
        # 未知类名直接返回原文
        return name

    @staticmethod
    def _derive_findings(top_class: str) -> tuple:
        """根据分类结果推导征象等级（关键词匹配，适用于任何数据集）"""
        cls_lower = top_class.lower()

        NORMAL_CONCEPTS = {"normal_tissue", "regular_architecture", "isoechoic_texture"}
        ANOMALY_CONCEPTS = {"tumor_mass", "inflammation", "necrosis",
                            "calcification", "cyst_fluid", "fibrosis_scar"}

        if any(kw in cls_lower for kw in ("malignant", "cancer", "carcinoma")):
            findings = {
                "tumor_mass": ("显著", 0.90), "inflammation": ("可见", 0.55),
                "necrosis": ("可见", 0.50), "calcification": ("不明显", 0.30),
                "cyst_fluid": ("不明显", 0.25), "fibrosis_scar": ("不明显", 0.20),
                "normal_tissue": ("未见", 0.10),
                "regular_architecture": ("未见", 0.08),
                "isoechoic_texture": ("未见", 0.12),
            }
        elif any(kw in cls_lower for kw in ("thymoma", "tumor", "mass", "lesion", "neoplasm")):
            findings = {
                "tumor_mass": ("显著", 0.85), "inflammation": ("不明显", 0.25),
                "necrosis": ("不明显", 0.25), "calcification": ("可见", 0.50),
                "cyst_fluid": ("不明显", 0.25), "fibrosis_scar": ("不明显", 0.30),
                "normal_tissue": ("未见", 0.10),
                "regular_architecture": ("未见", 0.10),
                "isoechoic_texture": ("未见", 0.15),
            }
        elif any(kw in cls_lower for kw in ("benign",)):
            findings = {
                "tumor_mass": ("可见", 0.55), "inflammation": ("不明显", 0.30),
                "necrosis": ("未见", 0.10), "calcification": ("不明显", 0.25),
                "cyst_fluid": ("不明显", 0.25), "fibrosis_scar": ("不明显", 0.20),
                "normal_tissue": ("可见", 0.60),
                "regular_architecture": ("可见", 0.55),
                "isoechoic_texture": ("不明显", 0.35),
            }
        elif any(kw in cls_lower for kw in ("normal",)):
            findings = {
                "tumor_mass": ("未见", 0.05), "inflammation": ("未见", 0.05),
                "necrosis": ("未见", 0.02), "calcification": ("未见", 0.05),
                "cyst_fluid": ("未见", 0.05), "fibrosis_scar": ("未见", 0.05),
                "normal_tissue": ("显著", 0.90),
                "regular_architecture": ("显著", 0.88),
                "isoechoic_texture": ("显著", 0.85),
            }
        else:
            # 未知类别：全部中性
            findings = {c: ("不明显", 0.30) for c in ANOMALY_CONCEPTS | NORMAL_CONCEPTS}

        anomaly_findings = {}
        normal_findings = {}
        for cname, (level, value) in findings.items():
            if cname in NORMAL_CONCEPTS:
                normal_findings[cname] = {"value": value, "level": level}
            else:
                anomaly_findings[cname] = {"value": value, "level": level}
        return anomaly_findings, normal_findings

    # ----------------------------------------------------------
    # 报告组装（私有）
    # ----------------------------------------------------------

    def _compose_report(self, report_id, patient_id, imaging_desc,
                        class_cn, confidence, confidence_desc, top_prob,
                        anomaly_level, anomaly_concept_descs,
                        normal_concept_descs, impression, recommendations,
                        classnames_cn, probs):
        """组装完整报告文本"""

        top_line = f"报告编号: {report_id}"
        lines = [
            "=" * 64,
            f"        医学影像AI辅助分析报告",
            "=" * 64,
            "",
            f"报告编号: {report_id}",
            f"患者编号: {patient_id}",
            f"检查模态: {self.modality}",
            f"生成时间: {self.report_time.strftime('%Y年%m月%d日 %H:%M')}",
            f"生成方式: 基于超声影像的AI辅助自动分析",
            "─" * 64,
            "",
        ]

        # ---- 【一、影像学所见】----
        lines.append("【一、影像学所见】")
        lines.append(f"    送检{self.modality}图像，经AI辅助分析系统评估，")
        # 每行缩进4个空格，自动换行
        for i, line in enumerate(self._wrap_text(imaging_desc, width=56)):
            prefix = "    " if i == 0 else "      "
            lines.append(f"{prefix}{line}")
        lines.append("")

        # ---- 【二、影像征象分析】----
        lines.append("【二、影像征象分析】")

        # 异常征象
        if anomaly_concept_descs:
            lines.append("  一、异常影像征象评估：")
            lines.append("")
            for item in anomaly_concept_descs:
                level_mark = {"显著": "**", "可见": "*~", "不明显": "~ ", "未见": "  "}.get(item["level"], "  ")
                lines.append(f"  {level_mark} {item['name']}：{item['level']}")
                for line in self._wrap_text(item["desc"], width=54):
                    lines.append(f"      {line}")
                lines.append("")
        else:
            lines.append("  未见明确异常影像征象。")
            lines.append("")

        # 正常征象
        if normal_concept_descs:
            lines.append("  二、正常组织特征评估：")
            lines.append("")
            for item in normal_concept_descs:
                level_mark = {"显著": "**", "可见": "*~", "不明显": "~ ", "未见": "  "}.get(item["level"], "  ")
                lines.append(f"  {level_mark} {item['name']}：{item['level']}")
                for line in self._wrap_text(item["desc"], width=54):
                    lines.append(f"      {line}")
                lines.append("")

        # ---- 【三、综合印象】----
        lines.append("【三、综合印象】")
        lines.append(f"    结合{self.modality}影像学特征及AI辅助分析结果，初步考虑：")
        lines.append(f"    （{self.modality}）{class_cn}（{confidence_desc.rstrip('。')}）。")
        lines.append(f"    异常程度评估：{anomaly_level}。")
        lines.append("")
        lines.append(f"    {impression}")
        lines.append("")

        # 鉴别诊断（多分类时列出其他可能）
        if len(probs) > 2:
            sorted_probs = sorted(zip(classnames_cn, probs), key=lambda x: x[1], reverse=True)
            lines.append("    各类别评估：")
            for name, p in sorted_probs:
                bar_len = int(p * 20)
                bar = "=" * bar_len + "-" * (20 - bar_len)
                lines.append(f"      {name:8s}  [{bar}]  {p:.1%}")
            lines.append("")

        # ---- 【四、临床建议】----
        lines.append("【四、临床建议】")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"    {i}. {rec}")
        lines.append(f"    {len(recommendations) + 1}. 本报告为AI辅助生成，仅供临床参考，最终诊断需结合临床表现及其他检查结果综合判断。")
        lines.append("")

        # ---- 免责声明 ----
        lines.append("=" * 64)
        lines.append("免责声明:")
        for line in self._wrap_text(DISCLAIMER, width=60):
            lines.append(f"  {line}")
        lines.append("=" * 64)

        return "\n".join(lines)

    @staticmethod
    def _wrap_text(text: str, width: int = 56) -> list:
        """按宽度自动换行，保留中文标点粘连"""
        result = []
        current = ""
        for char in text:
            current += char
            # 中文字符和中文标点每个算2个宽度
            w = sum(2 if ord(c) > 127 else 1 for c in current)
            if w >= width:
                result.append(current)
                current = ""
        if current:
            result.append(current)
        return result

    # ----------------------------------------------------------
    # LLM 润色接口
    # ----------------------------------------------------------

    def get_llm_polish_prompt(self) -> str:
        """
        生成一段可发送给大语言模型的润色指令。

        Returns:
            str: 包含结构化发现和润色要求的 prompt 文本。
                 可直接复制发送给 ChatGPT/Claude/DeepSeek 等大模型。
        """
        if not self._findings:
            return "（尚未生成报告，请先调用 generate() 方法）"

        f = self._findings

        # 构建异常征象摘要
        anomaly_lines = []
        for item in f["anomaly_concepts"]:
            anomaly_lines.append(f"  - {item['name']}: {item['level']}（{item['desc']}）")

        normal_lines = []
        for item in f["normal_concepts"]:
            normal_lines.append(f"  - {item['name']}: {item['level']}（{item['desc']}）")

        # 映射分类名为中文
        _class_cn = self._classname_to_cn(f['top_class'])

        prompt = f"""你是一位资深的超声影像科医师，请将以下AI系统生成的乳腺超声分析结果润色为一份专业、流畅的中文医学影像报告。

【关键发现】
- 主要诊断: {_class_cn}（可信度: {f['confidence']}，概率: {f['top_probability']:.1%}）
- 异常程度: {f['anomaly_level']}（异常评分: {f['anomaly_score']:.2f}）
- 病变区域占比: 约{f['lesion_coverage_ratio']:.0%}

【异常影像征象】
{chr(10).join(anomaly_lines) if anomaly_lines else '  无明确异常征象'}

【正常组织特征】
{chr(10).join(normal_lines) if normal_lines else '  无明确正常征象'}

【综合印象（草稿）】
{f['impression']}

【润色要求】
1. 保持医学专业性和严谨性，使用规范的超声影像学术语。
2. 语言自然流畅，避免生硬的模板感，像是一位资深医师在口述报告。
3. 结构保持为四个板块：【一、影像学所见】【二、影像征象分析】【三、综合印象】【四、临床建议】。
4. 不要在报告中出现"评分"、"概率"、"置信度"等计算机/数学术语，应转化为临床描述。
5. 报告末尾保留免责声明。
6. 仅输出润色后的报告正文，不要添加额外说明。"""

        return prompt

    def apply_llm_polish(self, polished_text: str):
        """
        应用大语言模型润色后的报告文本。

        Args:
            polished_text: 大模型返回的润色后报告文本

        Note:
            调用此方法后，generate() 返回的字典中 is_polished 将变为 True，
            polished_text 字段将包含润色后的内容。
        """
        self._polished_text = polished_text

    def get_final_report(self) -> str:
        """获取最终报告文本（优先返回润色版，无润色则返回原始版）"""
        return self._polished_text or self._raw_report_text

    def is_polished(self) -> bool:
        """报告是否已经过大模型润色"""
        return self._polished_text is not None
