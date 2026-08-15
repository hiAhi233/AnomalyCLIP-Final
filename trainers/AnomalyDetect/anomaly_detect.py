"""
AnomalyCLIP: Memory Bank Anomaly Detection + Lesion Masking
============================================================
Path A (Classification):    BiomedCoOp prompt learning → disease logits
Path B (Anomaly Detection): Normal patch Memory Bank → distance → anomaly map
Path C (Image Processing):  anomaly map → mask → suppress background → Path A

M: [N, 768] — all normal patches from normal_scan images (built offline)

score_i = min_j ||patch_i - M_j||  →  anomaly map
mask    = score > threshold        →  suppress background in input image
"""

import os
import os.path as osp
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

from trainers.BiomedCoOp.biomedcoop_biomedclip import PromptLearner, TextEncoder
from trainers.AnomalyDetect.text_descriptions import get_text_descriptions


# ============================================================
#  Utility: cosine distance to memory bank
# ============================================================

def cosine_distance(patches, M):
    """patches: [B, N, D], M: [K, D] (L2-normalized both)
       Returns: [B, N]  minimum distance to M for each patch"""
    # cos_sim: [B, N, K]
    cos_sim = patches @ M.T
    # distance = 1 - max cosine similarity
    max_sim = cos_sim.amax(dim=-1)    # [B, N]
    return 1.0 - max_sim               # [B, N]


# ============================================================
#  Core Model
# ============================================================

class AnomalyCLIP(nn.Module):

    def __init__(self, cfg, classnames, biomedclip_model, text_descriptions):
        super().__init__()
        n_cls = len(classnames)

        # ---- Path A: BiomedCoOp Classification ----
        self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = biomedclip_model.visual
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype
        self.n_cls = n_cls
        self.cfg = cfg

        # ---- Path B: Anomaly Detection ----
        # BiomedCLIP: ViT patch features = 768-dim, text features = 512-dim
        # We project text anchors → 768-dim image space for direct comparison
        self.patch_dim = 768
        self.text_dim = 512

        self.patch_proj = nn.Sequential(
            nn.Linear(self.patch_dim, self.patch_dim),
            nn.LayerNorm(self.patch_dim),
        )

        # Projection: text space (512) → image patch space (768)
        self.text_to_patch = nn.Linear(self.text_dim, self.patch_dim, bias=False)

        # ---- text anchors (for concept description only) ----
        tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        anchors_normal, names_normal = self._encode_concept_anchors(
            text_descriptions['normal'], biomedclip_model, tokenizer)
        anchors_anomaly, names_anomaly = self._encode_concept_anchors(
            text_descriptions['anomaly'], biomedclip_model, tokenizer)

        # Project text anchors to patch space, detach from autograd graph
        with torch.no_grad():
            self.register_buffer('anchors_normal',
                self.text_to_patch(anchors_normal).detach())
            self.register_buffer('anchors_anomaly',
                self.text_to_patch(anchors_anomaly).detach())
        self.concept_names_normal = names_normal
        self.concept_names_anomaly = names_anomaly

        # ---- Memory Bank (loaded separately) ----
        mbank_path = getattr(cfg.TRAINER.ANOMALY_DETECT, 'MEMORY_BANK_PATH', '')
        self.memory_bank = None
        if mbank_path and osp.exists(mbank_path):
            data = torch.load(mbank_path, map_location='cpu', weights_only=True)
            self.register_buffer('M', data['M'])   # [K, 768], frozen
            self.memory_bank = True
            print(f'Memory Bank loaded: {self.M.shape[0]} patches')
        else:
            print('No memory bank found — using text anchors for anomaly scoring')

        # ---- hyperparameters ----
        self.lambda_consist = cfg.TRAINER.ANOMALY_DETECT.LAMBDA_CONSIST
        self.threshold_std = cfg.TRAINER.ANOMALY_DETECT.THRESHOLD_STD
        self.patch_temperature = cfg.TRAINER.ANOMALY_DETECT.PATCH_TEMPERATURE
        self.mask_mode = getattr(cfg.TRAINER.ANOMALY_DETECT, 'MASK_MODE', True)
        self.bg_factor = getattr(cfg.TRAINER.ANOMALY_DETECT, 'BG_FACTOR', 0.3)
        # ---- dual-stream fusion ----
        self.lambda_fusion = getattr(cfg.TRAINER.ANOMALY_DETECT, 'LAMBDA_FUSION', 0.7)
        # ---- report generator ----
        # gen_backend: "qwen" (预训练语言模型, 推荐) | "lstm" (层级LSTM, 旧版)
        self.gen_mode = getattr(cfg.TRAINER.ANOMALY_DETECT, 'GEN_MODE', False)
        self.gen_backend = getattr(cfg.TRAINER.ANOMALY_DETECT, 'GEN_BACKEND', 'qwen')
        self.lambda_tag = getattr(cfg.TRAINER.ANOMALY_DETECT, 'LAMBDA_TAG', 0.5)
        self.lambda_stop = getattr(cfg.TRAINER.ANOMALY_DETECT, 'LAMBDA_STOP', 0.5)
        self.lambda_word = getattr(cfg.TRAINER.ANOMALY_DETECT, 'LAMBDA_WORD', 1.0)
        if self.gen_mode:
            if self.gen_backend == 'qwen':
                from trainers.AnomalyDetect.report_generator_qwen import QwenReportGenerator
                qwen_path = getattr(cfg.TRAINER.ANOMALY_DETECT, 'QWEN_PATH',
                                    'models/qwen2.5-0.5b')
                self.report_gen = QwenReportGenerator(
                    model_path=qwen_path,
                    patch_dim=768,   # trunk 原始 patch 特征空间
                    vis_tokens=getattr(cfg.TRAINER.ANOMALY_DETECT, 'VIS_TOKENS', 64),
                    struct_dim=19,   # 结构化征象维度
                    struct_tokens=getattr(cfg.TRAINER.ANOMALY_DETECT, 'STRUCT_TOKENS', 4),
                    lora_r=getattr(cfg.TRAINER.ANOMALY_DETECT, 'LORA_R', 4),
                    lora_alpha=getattr(cfg.TRAINER.ANOMALY_DETECT, 'LORA_ALPHA', 8),
                    lora_dropout=getattr(cfg.TRAINER.ANOMALY_DETECT, 'LORA_DROPOUT', 0.1),
                    gen_max_tokens=getattr(cfg.TRAINER.ANOMALY_DETECT, 'GEN_MAX_TOKENS', 110),
                    temperature=getattr(cfg.TRAINER.ANOMALY_DETECT, 'GEN_TEMPERATURE', 0.4),
                )
                print(f'Report generator (Qwen2.5-0.5B, {getattr(cfg.TRAINER.ANOMALY_DETECT, "VIS_TOKENS", 64)} 视觉词元, LoRA r={getattr(cfg.TRAINER.ANOMALY_DETECT, "LORA_R", 4)}) initialized from {qwen_path}')
            else:
                from trainers.AnomalyDetect.report_generator_net import ReportGenNet
                vocab_path = getattr(cfg.TRAINER.ANOMALY_DETECT, 'VOCAB_PATH', '')
                vocab_size = 1008  # fallback (thymoma corpus default)
                if vocab_path and os.path.exists(vocab_path):
                    import pickle as _pkl
                    with open(vocab_path, 'rb') as f:
                        vocab_size = len(_pkl.load(f))
                self.report_gen = ReportGenNet(
                    vocab_size=vocab_size,
                    s_max=getattr(cfg.TRAINER.ANOMALY_DETECT, 'S_MAX', 8),
                    n_max=getattr(cfg.TRAINER.ANOMALY_DETECT, 'N_MAX', 50),
                )
                print(f'Report generator (LSTM) initialized: vocab={vocab_size}')
        self.anomaly_indices = list(range(n_cls - 1))

    # ---- anchor encoding (shared with original) ----
    def _encode_anchor(self, descs, model, tok):
        dev = next(model.parameters()).device
        embs = []
        for d in descs:
            t = tok(d).to(dev)
            with torch.no_grad():
                e = model.encode_text(t)
            e = e.squeeze(0)
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e)
        a = torch.stack(embs).mean(dim=0)
        return a / a.norm(dim=-1, keepdim=True)

    def _encode_concept_anchors(self, concepts_dict, model, tok):
        anchors, names = [], []
        for name, descs in concepts_dict.items():
            a = self._encode_anchor(descs, model, tok)
            anchors.append(a)
            names.append(name)
        return torch.stack(anchors, dim=0), names

    def _extract_patch_tokens(self, image):
        x = image.type(self.dtype)
        raw = self.image_encoder.trunk.forward_features(x)
        return self.image_encoder(x), raw[:, 1:, :]

    def _get_lambda(self, s_img):
        """
        双流融合权重: 掩膜可信(异常分高) → 偏向病灶聚焦流;
        s_img 低(疑似无病灶, 掩膜不可信) → 回落偏向全图流。
        """
        lam = self.lambda_fusion
        if isinstance(s_img, torch.Tensor):
            s = s_img.mean().item()
        else:
            s = float(s_img)
        if s < 0.05:  # 原始特征空间: 正常 s_img≈0.0 (旧标定 0.2 针对饱和分数)
            lam = min(lam, 0.3)
        return lam

    def forward(self, image, label=None, captions=None, prob_real=None, tag_target=None, struct=None):
        logit_scale = self.logit_scale.exp()

        # ---- extract features ----
        image_features, patch_tokens = self._extract_patch_tokens(image)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        # ---- Path A (classification) ----
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_full = logit_scale * image_features_norm @ text_features.t()

        # ---- Path B (anomaly detection) ----
        # 注意: 记忆库/文本锚点存的是 trunk 原始 patch 特征, 这里直接比较原始特征,
        # 不再过 patch_proj (投影后的空间与库不一致 → s_img 饱和 0.99 的根因)
        patch_features = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        if self.memory_bank is not None:
            # Memory Bank mode: distance to nearest normal patch
            anomaly_scores = cosine_distance(patch_features, self.M)  # [B, 196]
        else:
            # Text anchor mode: max anomaly - max normal (fallback)
            sim_anom = (patch_features @ self.anchors_anomaly.T).amax(dim=-1)
            sim_norm = (patch_features @ self.anchors_normal.T).amax(dim=-1)
            anomaly_scores = torch.sigmoid(sim_anom - sim_norm)

        # ---- image-level anomaly score ----
        L = anomaly_scores
        thr = L.mean(dim=1, keepdim=True) + self.threshold_std * L.std(dim=1, keepdim=True)
        mask = L > thr
        n_abov = mask.sum(dim=1).float().clamp(min=1)
        s_img = ((L * mask.float()).sum(dim=1) / n_abov).clamp(0, 1)

        # ---- concept scores (for report) ----
        concept_scores = {
            "normal": dict(zip(self.concept_names_normal,
                (patch_features @ self.anchors_normal.T).unbind(dim=-1))),
            "anomaly": dict(zip(self.concept_names_anomaly,
                (patch_features @ self.anchors_anomaly.T).unbind(dim=-1))),
        }

        # ---- Path C: image masking (suppress background for classification) ----
        masked_image = image
        if self.mask_mode and self.memory_bank is not None:
            # upsample anomaly map [B, 196] → [B, 1, 224, 224]
            B = image.shape[0]
            amap_2d = anomaly_scores.reshape(B, 14, 14)          # [B, 14, 14]
            # thr: [B, 1] → [B, 1, 1] for broadcast with [B, 14, 14]
            mask_2d = (amap_2d > thr.view(B, 1, 1)).float().unsqueeze(1)  # [B, 1, 14, 14]
            mask_224 = F.interpolate(mask_2d, size=224, mode='bilinear')  # [B, 1, 224, 224]

            # suppress background, keep lesion region
            apply = s_img > 0.1  # 原始特征空间距离: 正常≈0, 病灶 0.1~0.25 (0.4 是旧投影空间的标定)
            if apply.any():
                bg_suppressed = image * (mask_224 + self.bg_factor * (1 - mask_224))
                masked_image = torch.where(apply.view(-1,1,1,1), bg_suppressed, image)

        # ---- dual-stream fusion: lesion-focused (masked) + full-image ----
        if self.mask_mode and self.memory_bank is not None:
            masked_feat, _ = self._extract_patch_tokens(masked_image)
            masked_feat_norm = masked_feat / masked_feat.norm(dim=-1, keepdim=True)
            logits_masked = logit_scale * masked_feat_norm @ text_features.t()
            lam = self._get_lambda(s_img)
            logits = lam * logits_masked + (1 - lam) * logits_full
        else:
            logits = logits_full

        if self.prompt_learner.training:
            # ---- BiomedCoOp losses (unchanged) ----
            femb = self.prompt_learner.fixed_embeddings
            femb = femb / femb.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                zs_feat = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zs_feat = zs_feat / zs_feat.norm(dim=-1, keepdim=True)

                scores = []
                for i in range(femb.shape[1]):
                    tmp = logit_scale * image_features_norm @ femb[:, i, :].cuda().t()
                    scores.append(torch.mean(torch.max(tmp, dim=1).values).item())

                s_bar = torch.nan_to_num(torch.median(torch.tensor(scores)), nan=0.0)
                d_bar = torch.nan_to_num(torch.median(torch.abs(torch.tensor(scores) - s_bar)), nan=1e-8) + 1e-8
                z = (torch.tensor(scores) - s_bar) / d_bar
                tau = self.cfg.TRAINER.BIOMEDCOOP.TAU
                z_std = torch.nan_to_num(torch.std(z), nan=1.0).clamp(min=1e-8)
                mt = torch.abs((z - torch.mean(z)) / z_std) <= tau
                sel = femb[:, mt].mean(dim=1)
                sel = sel / sel.norm(dim=-1, keepdim=True)

            femb_mean = femb.mean(dim=1)
            femb_mean = femb_mean / femb_mean.norm(dim=-1, keepdim=True)
            zs_logits = logit_scale * zs_feat.cuda() @ sel.cuda().t()

            loss_ce = F.cross_entropy(logits, label)
            loss_sccm = torch.nan_to_num(
                F.mse_loss(text_features, femb_mean.cuda()) * self.cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA,
                nan=0.0)
            loss_kdsp = torch.nan_to_num(
                F.kl_div(F.log_softmax(logits, dim=1), F.log_softmax(zs_logits, dim=1),
                         reduction='sum', log_target=True) / logits.numel()
                * self.cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA,
                nan=0.0)

            # consistency loss: s_img should match classification
            s_cls_anom = F.softmax(logits, dim=1)[:, self.anomaly_indices].sum(dim=1)
            loss_consist = F.l1_loss(s_img, s_cls_anom.detach()) * self.lambda_consist

            # ---- report generation branch (teacher forcing) ----
            gen_losses = None
            if self.gen_mode and captions is not None:
                # 生成器视觉输入 = 全图 patch (不再用掩膜图 — 掩膜质量不绑架报告)
                # 代表层本就以病灶为最大软组织层, 病灶信息在全图 patch 中可读
                if self.gen_backend == 'qwen':
                    # Qwen: [视觉词元 | 征象词元 | 报告 token] → 标量 loss
                    # (非代表层全 pad → 批内自动不计损失)
                    gen_loss = self.report_gen(
                        patch_tokens, struct, captions, prob_real)
                    gen_losses = {'loss_word': gen_loss}
                    loss_ce = loss_ce + self.lambda_word * gen_loss
                else:
                    masked_feat, masked_patches = self._extract_patch_tokens(masked_image)
                    masked_feat_norm = masked_feat / masked_feat.norm(dim=-1, keepdim=True)
                    gen_losses = self.report_gen(
                        masked_feat_norm, captions, prob_real, tag_target)
                    loss_ce = loss_ce \
                        + self.lambda_tag * gen_losses['loss_tag'] \
                        + self.lambda_stop * gen_losses['loss_stop'] \
                        + self.lambda_word * gen_losses['loss_word']

            return logits, loss_ce, loss_sccm, loss_kdsp, loss_consist, s_img, gen_losses
        else:
            return logits, s_img, anomaly_scores, concept_scores, masked_image


# ============================================================
#  报告数据包装器 (Dassl DatasetWrapper + captions/tags)
# ============================================================

try:
    from dassl.data.data_manager import DatasetWrapper as DasslWrapper
except ImportError:
    DasslWrapper = None

from trainers.AnomalyDetect.report_generator_net import FINDING_ORDER as _GEN_FINDINGS


class ThymomaReportWrapper(DasslWrapper):
    """标准 Dassl wrapper + 报告 token (患者 ID → captions.json / thymoma_tags.json)"""

    def __init__(self, cfg, data_source, transform=None, is_train=False,
                 vocab=None, captions=None, tags=None, s_max=8, n_max=50,
                 tokenizer=None, qwen_backend=False, struct_feats=None):
        super().__init__(cfg, data_source, transform, is_train)
        self.vocab = vocab
        self.captions = captions      # {image_name: report_text}
        self.tags = tags              # {pid_str: {finding: 0/1}}
        self.s_max = s_max
        self.n_max = n_max
        self.tokenizer = tokenizer    # Qwen tokenizer (qwen 模式)
        self.qwen_backend = qwen_backend
        self.struct_feats = struct_feats  # {pid_str: tensor(19)} 结构化征象(标准化)

    def __getitem__(self, idx):
        output = super().__getitem__(idx)
        impath = output["impath"]

        # 从图像名解析患者 ID + 报告
        m = re.search(r'P(\d+)_', os.path.basename(impath))
        pid = m.group(1) if m else "0"
        report = self.captions.get(os.path.basename(impath), "") if self.captions else ""
        # 注意: captions 只挂代表层 → 非代表层 report 为空, 但仍要输出
        # 统一形状的 captions/prob_real (默认 collate 要求所有 item key 一致),
        # 空报告 → 全 pad + mask 0 → Qwen 侧 labels 全 -100, 不计损失

        if self.qwen_backend:
            # ---- Qwen 模式: 整段报告直接 tokenize (空报告 → 全 pad) ----
            if report:
                enc = self.tokenizer(
                    report, truncation=True, max_length=256,
                    padding='max_length', return_tensors='pt')
                output["captions"] = enc['input_ids'][0]        # [256]
                output["prob_real"] = enc['attention_mask'][0]  # [256] (当 mask 用)
            else:
                pad_id = self.tokenizer.pad_token_id
                output["captions"] = torch.full((256,), pad_id, dtype=torch.long)
                output["prob_real"] = torch.zeros(256, dtype=torch.long)
        else:
            # ---- LSTM 模式: 分句 → token 序列 ----
            try:
                import jieba
                def _tok(s):
                    toks = list(jieba.cut(s))
                    return [t for t in toks if t.strip() and t not in "，。、；：？！（）【】《》…—·"]
            except ImportError:
                def _tok(s):
                    return list(s)

            padded = torch.zeros(self.s_max, self.n_max, dtype=torch.long)
            prob_real = torch.zeros(self.s_max, dtype=torch.long)
            if report:
                sentences = [s for s in report.replace("\n", "").split("。") if len(s) > 1]
                sentences = sentences[:self.s_max]
                for i, sent in enumerate(sentences):
                    words = _tok(sent)[:self.n_max - 2]
                    toks = [self.vocab('<start>')] + [self.vocab(w) for w in words] + [self.vocab('<end>')]
                    padded[i, :len(toks)] = torch.tensor(toks[:self.n_max], dtype=torch.long)
                    prob_real[i] = 1
            output["captions"] = padded
            output["prob_real"] = prob_real

        output["tags"] = torch.tensor(
            [self.tags.get(pid, {}).get(f, 0) for f in _GEN_FINDINGS],
            dtype=torch.float) if self.tags else torch.zeros(len(_GEN_FINDINGS))

        # 结构化征象 (19 维, 标准化; 无记录 → 全 0)
        if self.struct_feats is not None:
            output["struct"] = self.struct_feats.get(pid, torch.zeros(19))
        else:
            output["struct"] = torch.zeros(19)

        return output


# ============================================================
#  Trainer
# ============================================================

@TRAINER_REGISTRY.register()
class AnomalyDetect_BiomedCLIP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.ANOMALY_DETECT.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader(self):
        super().build_data_loader()

        gen_mode = getattr(self.cfg.TRAINER.ANOMALY_DETECT, 'GEN_MODE', False)
        if not gen_mode:
            return

        # ---- 加载报告语料: vocab + captions + tags ----
        import json, pickle as _pkl
        from functools import partial
        from dassl.data.data_manager import build_data_loader as dassl_build_loader

        data_dir = os.path.join(self.cfg.DATASET.ROOT, self.cfg.DATASET.NAME)
        vocab_path = getattr(self.cfg.TRAINER.ANOMALY_DETECT, 'VOCAB_PATH', '')
        if not vocab_path:
            vocab_path = os.path.join(data_dir, 'vocab.pkl')
        captions_path = getattr(self.cfg.TRAINER.ANOMALY_DETECT, 'CAPTION_JSON', '')
        if not captions_path:
            captions_path = os.path.join(data_dir, 'captions.json')
        tags_path = getattr(self.cfg.TRAINER.ANOMALY_DETECT, 'TAGS_JSON', '')
        if not tags_path:
            tags_path = os.path.join(data_dir, 'thymoma_tags.json')

        with open(captions_path, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        tags = {}
        if os.path.exists(tags_path):
            with open(tags_path, 'r', encoding='utf-8') as f:
                tags = json.load(f)

        s_max = getattr(self.cfg.TRAINER.ANOMALY_DETECT, 'S_MAX', 8)
        n_max = getattr(self.cfg.TRAINER.ANOMALY_DETECT, 'N_MAX', 50)

        # Qwen 后端: 报告直接 tokenize, 不需要 vocab 分句
        gen_backend = getattr(self.cfg.TRAINER.ANOMALY_DETECT, 'GEN_BACKEND', 'qwen')
        qwen_tok = None
        if gen_backend == 'qwen':
            from transformers import AutoTokenizer as _AT
            qwen_path = getattr(self.cfg.TRAINER.ANOMALY_DETECT, 'QWEN_PATH',
                                'models/qwen2.5-0.5b')
            qwen_tok = _AT.from_pretrained(qwen_path, trust_remote_code=True)
            qwen_tok.pad_token = qwen_tok.eos_token
            self.vocab = None
            print(f'[ReportGen] Qwen tokenizer loaded from {qwen_path}')
        else:
            with open(vocab_path, 'rb') as f:
                self.vocab = _pkl.load(f)
        print(f'[ReportGen] captions={len(captions)} tags={len(tags)}')

        # ---- 结构化征象: 从 CSV 读 19 维测量值, 标准化后按患者索引 ----
        struct_feats = {}
        if gen_backend == 'qwen':
            csv_path = getattr(self.cfg.TRAINER.ANOMALY_DETECT, 'STRUCT_CSV', '')
            if csv_path and os.path.exists(csv_path):
                import pandas as _pd
                from sklearn.preprocessing import StandardScaler as _SS
                _df = _pd.read_csv(csv_path)
                _df['影像号'] = _df['影像号'].astype(int)
                NUM_F = ['长径mm', '短径mm', '年龄', '胸大肌平扫密度CT值', '肿块平扫密度CT值',
                         '病变动脉期CT值', '病变静脉期CT值', 'AFP', 'HCG', 'LDH', 'HCT红细胞压积']
                CAT_F = ['性别', '钙化', '形态', '边缘边界', '囊变坏死', '周围情况', '增强情况', '偏侧性']
                _X = _pd.DataFrame()
                for _c in NUM_F:
                    _X[_c] = _pd.to_numeric(_df[_c], errors='coerce')
                for _c in CAT_F:
                    _X[_c] = _df[_c].astype(str).astype('category').cat.codes
                _X = _X.fillna(_X.median())
                _sc = _SS().fit(_X.values)
                _Xs = _sc.transform(_X.values)
                for _i, _row in _df.iterrows():
                    struct_feats[str(int(_row['影像号']))] = torch.tensor(
                        _Xs[_i], dtype=torch.float)
                print(f'[ReportGen] 结构化征象: {len(struct_feats)} 患者 x {_Xs.shape[1]} 维')
            else:
                print('[ReportGen] 未找到 STRUCT_CSV, 结构化征象置零')

        # ---- 重建训练 loader (带报告) ----
        wrapper_cls = partial(ThymomaReportWrapper, vocab=self.vocab,
                              captions=captions, tags=tags, s_max=s_max, n_max=n_max,
                              tokenizer=qwen_tok, qwen_backend=(gen_backend == 'qwen'),
                              struct_feats=struct_feats)
        # 从 dm 内部取训练 transform (DataManager 未暴露, 用 _ 前缀属性)
        tfm_train = getattr(self.dm, 'tfm_train', None)
        if tfm_train is None:
            from dassl.data.transforms import build_transform
            tfm_train = build_transform(self.cfg, is_train=True)
        self.train_loader_x = dassl_build_loader(
            self.cfg,
            sampler_type=self.cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=self.dm.dataset.train_x,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=0,
            n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=wrapper_cls,
        )

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading BiomedCLIP ...')
        bclip, _ = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        bclip = bclip.float()  # keep on CPU for PromptLearner init, moved later

        dsname = cfg.DATASET.NAME
        tdescs = get_text_descriptions(dsname)
        print(f'Loaded {sum(len(v) for v in tdescs.values())} concept descriptions for {dsname}')

        self.model = AnomalyCLIP(cfg, classnames, bclip.eval(), tdescs)

        names_to_update = ["prompt_learner.ctx", "text_to_patch", "report_gen"]
        for n, p in self.model.named_parameters():
            p.requires_grad_(any(x in n for x in names_to_update))

        enabled = {n for n, p in self.model.named_parameters() if p.requires_grad}
        print(f'Trainable: {len(enabled)} — {enabled}')

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # ---- 分组学习率: 提示词按论文值, LoRA/投影按 LoRA 安全值 ----
        # (共享单 lr 时被迫取保守值 1e-4, 分类收敛不动; 分组后各自用文献验证值)
        prompt_lr = float(getattr(cfg.TRAINER.ANOMALY_DETECT, 'PROMPT_LR', 0.0025))
        gen_lr = float(getattr(cfg.TRAINER.ANOMALY_DETECT, 'GEN_LR', 0.0001))
        prompt_params, gen_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if 'prompt_learner' in n or 'text_to_patch' in n:
                prompt_params.append(p)
            elif 'report_gen' in n:
                gen_params.append(p)
            else:
                prompt_params.append(p)
        wd = float(getattr(cfg.OPTIM, 'WEIGHT_DECAY', 0.0005))
        self.optim = torch.optim.Adam([
            {"params": prompt_params, "lr": prompt_lr},
            {"params": gen_params, "lr": gen_lr},
        ], weight_decay=wd)
        print(f'Optimizer: Adam 分组 — 提示词/锚点 lr={prompt_lr} ({len(prompt_params)} params), '
              f'生成器 lr={gen_lr} ({len(gen_params)} params)')
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('anomaly_clip', self.model, self.optim, self.sched)
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.scaler = GradScaler() if cfg.TRAINER.ANOMALY_DETECT.PREC == 'amp' else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        prec = self.cfg.TRAINER.ANOMALY_DETECT.PREC

        # ---- 报告生成数据 (GEN_MODE 时存在) ----
        captions = batch.get("captions", None)
        prob_real = batch.get("prob_real", None)
        tag_target = batch.get("tags", None)
        struct = batch.get("struct", None)
        if captions is not None:
            captions = captions.to(self.device)
            prob_real = prob_real.to(self.device)
            tag_target = tag_target.to(self.device)
        if struct is not None:
            struct = struct.to(self.device)

        if prec == 'amp':
            with autocast():
                loss = model(image, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            out = model(image, label, captions, prob_real, tag_target, struct)
            logits, loss_ce, loss_sccm, loss_kdsp, loss_consist, s_img = out[:6]
            gen_losses = out[6] if len(out) > 6 else None
            loss = loss_ce + loss_sccm + loss_kdsp + loss_consist

            # Qwen fp16 梯度容易溢出 → 训练前裁剪生成器梯度
            if getattr(self.model, 'gen_backend', '') == 'qwen':
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.report_gen.parameters(), max_norm=1.0)
                self.optim.step()
            else:
                self.model_backward_and_update(loss)

        ret = {"loss": loss.item(),
               "loss_ce": loss_ce.item(), "loss_consist": loss_consist.item(),
               "s_img_mean": s_img.mean().item(),
               "acc": compute_accuracy(logits, label)[0].item()}
        if gen_losses is not None:
            for k, v in gen_losses.items():
                ret[f"gen_{k}"] = v.item()

        # 推进学习率调度器 (原版 BiomedCoOp 训练器同款钩子;
        # 缺失导致 lr 永远停在 warmup 值 1e-5, 分类学不动)
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return ret

    def parse_batch_train(self, batch):
        img, label = batch["img"].to(self.device), batch["label"].to(self.device)
        return img, label

    def model_inference(self, input):
        return self.model(input)[0]  # logits for evaluator

    def load_model(self, directory, epoch=None):
        if not directory:
            return
        for name in self.get_model_names():
            mfile = f'model.pth.tar-{epoch}' if epoch else 'model-best.pth.tar'
            path = osp.join(directory, name, mfile)
            if not osp.exists(path):
                raise FileNotFoundError(f'Model not found: {path}')
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            state = ckpt['state_dict']
            for k in ['prompt_learner.token_prefix', 'prompt_learner.token_suffix']:
                state.pop(k, None)
            print(f'Loading {name} from {path} (epoch={ckpt.get("epoch")})')
            self._models[name].load_state_dict(state, strict=False)
