"""
build_report_corpus.py — 报告语料构建
================================================================
从胸腺瘤 233 例切片 + 报告生成训练语料:

  1. captions.json:  {切片文件名: 患者报告全文}
     格式仿 Vinyals 2017 (Medical-Report-Generation-master)

  2. vocab.pkl:      中文词表 (jieba 分词 + 词频 top N)
     特殊 token: <pad> <start> <end> <unk>

  3. thymoma_tags.json: 9 征象标签 one-hot
     从报告文本自动匹配关键词 (仿 Vinyals 的 tag in text)

Usage:
  python build_report_corpus.py
"""

import os, re, json, pickle
import numpy as np
from collections import Counter, defaultdict

# ============================================================
# 配置
# ============================================================

DATA_DIR = r"D:\图神经网络\异常检测\代码demo4\BiomedCoOp-main\data\Thymoma"
VOCAB_SIZE = 5000          # 词表大小 (阈值=1 时确保所有词收录)
WORD_FREQ_THRESHOLD = 1    # 词频阈值 (消灭<unk>)

# 9 征象关键词（从报告文本匹配，仿 Vinyals 的 static_tags）
FINDING_KEYWORDS = {
    "tumor_mass":        ["肿块", "占位", "结节", "肿物", "软组织影", "团块"],
    "inflammation":      ["炎性", "炎症", "炎变", "感染"],
    "necrosis":          ["坏死", "液化"],
    "calcification":     ["钙化", "钙灶"],
    "cyst_fluid":        ["囊肿", "囊性", "囊变", "囊状", "液性"],
    "fibrosis_scar":     ["纤维", "瘢痕", "条索"],
    "normal_tissue":     ["未见异常", "未见明确", "未见明显", "正常", "清晰", "无异常"],
    "regular_architecture": ["规则", "规整", "光整", "清晰", "分界清", "边界清"],
    "isoechoic_texture": ["均匀", "一致", "密度均匀"],
}

FINDING_ORDER = ["tumor_mass", "inflammation", "necrosis", "calcification",
                 "cyst_fluid", "fibrosis_scar", "normal_tissue",
                 "regular_architecture", "isoechoic_texture"]


# ============================================================
# 分词 + 词表
# ============================================================

try:
    import jieba
except ImportError:
    jieba = None


def tokenize(text: str) -> list:
    """中文分词，去标点"""
    if jieba is not None:
        tokens = list(jieba.cut(text))
    else:
        tokens = list(text)
    tokens = [t for t in tokens if t.strip() and t not in "，。、；：？！（）【】《》\"'…—·\n "]
    return tokens


# Vocabulary 类定义在 report_generator_net.py (pickle 加载需要)
from trainers.AnomalyDetect.report_generator_net import Vocabulary


def build_vocab(all_texts, size=VOCAB_SIZE, threshold=WORD_FREQ_THRESHOLD):
    counter = Counter()
    for text in all_texts:
        counter.update(tokenize(text))

    words = [w for w, cnt in counter.most_common(size)
             if cnt >= threshold and w not in ('<pad>', '<start>', '<end>', '<unk>')]
    vocab = Vocabulary()
    for w in words:
        vocab.add_word(w)
    return vocab


# ============================================================
# 征象标签
# ============================================================

# 否定词: 匹配到的关键词如果前面有这些词，视为假阳性 (征象不存在)
NEGATION_PATTERNS = [
    "未见", "未发现", "无明显", "未见明确", "未见明显", "排除",
    "不除外", "不考虑", "不考虑为", "未见异常", "未见明确",
    "无异常", "无明确", "未见肿块", "未见占位", "无占位",
    "未见肿物", "无肿物", "未见明显异常", "未见明确异常",
]


def extract_tags(text: str) -> dict:
    """从报告文本匹配 9 征象 → one-hot (含否定过滤)"""
    tags = {}
    for finding, keywords in FINDING_KEYWORDS.items():
        found = False
        for kw in keywords:
            idx = text.find(kw)
            if idx < 0:
                continue
            # 检查关键词前面 15 字内是否有否定词
            prefix = text[max(0, idx - 15):idx]
            if any(neg in prefix for neg in NEGATION_PATTERNS):
                continue  # 被否定 → 假阳性, 跳过
            found = True
            break
        tags[finding] = 1 if found else 0
    return tags


# ============================================================
# 主流程
# ============================================================

def main():
    # ---- 1. 收集所有 (切片路径, 报告) ----
    samples = []        # [(image_name, report_text)]
    reports_by_patient = {}

    for cls_dir in os.listdir(DATA_DIR):
        cls_path = os.path.join(DATA_DIR, cls_dir)
        if not os.path.isdir(cls_path):
            continue

        # 先收集该类的报告
        cls_reports = {}
        for f in os.listdir(cls_path):
            if f.endswith("_报告.txt"):
                m = re.match(r'P(\d+)_报告\.txt', f)
                if m:
                    with open(os.path.join(cls_path, f), "r", encoding="utf-8") as fp:
                        cls_reports[int(m.group(1))] = fp.read().strip()

        # 切片 → 报告
        for f in os.listdir(cls_path):
            if not f.endswith(".png"):
                continue
            m = re.match(r'P(\d+)_', f)
            if m:
                pid = int(m.group(1))
                report = cls_reports.get(pid, "")
                if report:
                    samples.append((f, report))
                    reports_by_patient[pid] = report

    print(f"图像-报告对: {len(samples)}  (患者: {len(reports_by_patient)})")

    # ---- 2. captions.json ----
    captions = {img: report for img, report in samples}
    with open(os.path.join(DATA_DIR, "captions.json"), "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    print(f"captions.json: {len(captions)} 条")

    # ---- 3. 词表 ----
    vocab = build_vocab(list(reports_by_patient.values()))
    with open(os.path.join(DATA_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    print(f"vocab.pkl: {len(vocab)} 词")

    # ---- 4. 征象标签 ----
    tag_stats = defaultdict(int)
    tag_data = {}
    for pid, report in reports_by_patient.items():
        tags = extract_tags(report)
        tag_data[str(pid)] = tags
        for k, v in tags.items():
            if v == 1:
                tag_stats[k] += 1

    with open(os.path.join(DATA_DIR, "thymoma_tags.json"), "w", encoding="utf-8") as f:
        json.dump(tag_data, f, ensure_ascii=False, indent=2)
    print("征象标签分布:")
    for k in FINDING_ORDER:
        print(f"  {k:20s}: {tag_stats[k]}/{len(reports_by_patient)} 例阳性")

    # ---- 5. 统计句子长度（报告分句）----
    sent_lens = []
    for report in reports_by_patient.values():
        sentences = [s for s in report.replace("\n", "").split("。") if len(s) > 1]
        sent_lens.append(len(sentences))
    arr = np.array(sent_lens)
    print(f"报告句子数: mean={arr.mean():.1f}  p90={np.percentile(arr, 90):.0f}  max={arr.max()}")


if __name__ == "__main__":
    main()
