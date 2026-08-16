"""
AnomalyCLIP: Memory Bank Anomaly Detection + Report Generation
============================================================
Path B (Anomaly Detection): Normal patch Memory Bank → distance → anomaly map → s_img
Path C (Localization):      anomaly map → threshold → lesion mask (定位输出, 不喂生成器)
Report:                     Qwen2.5-0.5B + LoRA, 全图 patch + 尺寸文本 + 征象词元

分类已从本模型移除 (提示词学习与文本原型提取退役) — 由结构化征象分类器负责
(train_structured_classifier.py, 二分类 79.1% / 11类 56.5%), 推理时注入生成器。

M: [N, 768] — all normal patches from benign-class images (built offline)
score_i = 1 - max_j cos(patch_i, M_j)  →  anomaly map
"""

import os
import os.path as osp
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from open_clip.src.open_clip import create_model_from_pretrained



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

    def __init__(self, cfg, biomedclip_model):
        super().__init__()

        self.image_encoder = biomedclip_model.visual
        self.dtype = biomedclip_model.text.transformer.dtype
        self.cfg = cfg

        # ---- Path B: Anomaly Detection (记忆库最近邻, 原始 patch 特征空间) ----
        self.patch_dim = 768

        # ---- Memory Bank (loaded separately) ----
        mbank_path = getattr(cfg.TRAINER.ANOMALY_DETECT, 'MEMORY_BANK_PATH', '')
        self.memory_bank = None
        if mbank_path and osp.exists(mbank_path):
            data = torch.load(mbank_path, map_location='cpu', weights_only=True)
            self.register_buffer('M', data['M'])   # [K, 768], frozen
            self.memory_bank = True
            print(f'Memory Bank loaded: {self.M.shape[0]} patches')
        else:
            print('No memory bank found — anomaly scores will be zero')

        # ---- hyperparameters ----
        self.threshold_std = cfg.TRAINER.ANOMALY_DETECT.THRESHOLD_STD
        self.mask_mode = getattr(cfg.TRAINER.ANOMALY_DETECT, 'MASK_MODE', True)
        self.bg_factor = getattr(cfg.TRAINER.ANOMALY_DETECT, 'BG_FACTOR', 0.3)
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
    def _extract_patch_tokens(self, image):
        x = image.type(self.dtype)
        raw = self.image_encoder.trunk.forward_features(x)
        return self.image_encoder(x), raw[:, 1:, :]

    def forward(self, image, captions=None, prob_real=None,
                struct=None, size_ids=None, size_mask=None):
        # ---- extract features ----
        _, patch_tokens = self._extract_patch_tokens(image)

        # ---- Path B: 记忆库最近邻 (原始 patch 特征空间, 与建库一致) ----
        patch_features = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if self.memory_bank is not None:
            anomaly_scores = cosine_distance(patch_features, self.M)  # [B, 196]
        else:
            anomaly_scores = torch.zeros(patch_features.shape[0], patch_features.shape[1],
                                         device=patch_features.device)

        # ---- image-level anomaly score ----
        L = anomaly_scores
        thr = L.mean(dim=1, keepdim=True) + self.threshold_std * L.std(dim=1, keepdim=True)
        mask = L > thr
        n_abov = mask.sum(dim=1).float().clamp(min=1)
        s_img = ((L * mask.float()).sum(dim=1) / n_abov).clamp(0, 1)

        # ---- Path C: 异常热力图 → 掩膜 (定位输出用, 不喂生成器) ----
        masked_image = image
        if self.mask_mode and self.memory_bank is not None:
            B = image.shape[0]
            amap_2d = anomaly_scores.reshape(B, 14, 14)          # [B, 14, 14]
            mask_2d = (amap_2d > thr.view(B, 1, 1)).float().unsqueeze(1)  # [B, 1, 14, 14]
            mask_224 = F.interpolate(mask_2d, size=224, mode='bilinear')  # [B, 1, 224, 224]
            apply = s_img > 0.1  # 原始特征空间距离: 正常≈0, 病灶 0.1~0.25
            if apply.any():
                bg_suppressed = image * (mask_224 + self.bg_factor * (1 - mask_224))
                masked_image = torch.where(apply.view(-1, 1, 1, 1), bg_suppressed, image)

        if self.training:
            # ---- report generation branch (teacher forcing) ----
            loss = torch.zeros((), device=image.device, requires_grad=True)
            gen_losses = None
            if self.gen_mode and captions is not None:
                if self.gen_backend == 'qwen':
                    # Qwen: [视觉词元 | 尺寸文本 | 征象词元 | 报告 token] → 标量 loss
                    # (非代表层全 pad → 批内自动不计损失)
                    gen_loss = self.report_gen(
                        patch_tokens, size_ids, size_mask, struct, captions, prob_real)
                    gen_losses = {'loss_word': gen_loss}
                    loss = loss + self.lambda_word * gen_loss
                else:
                    masked_feat, masked_patches = self._extract_patch_tokens(masked_image)
                    masked_feat_norm = masked_feat / masked_feat.norm(dim=-1, keepdim=True)
                    gen_losses = self.report_gen(
                        masked_feat_norm, captions, prob_real)
                    loss = loss + self.lambda_word * gen_losses['loss_word']

            return loss, s_img, gen_losses
        else:
            return s_img, anomaly_scores, masked_image


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
                 tokenizer=None, qwen_backend=False, struct_feats=None,
                 size_texts=None, size_max_len=32):
        super().__init__(cfg, data_source, transform, is_train)
        self.vocab = vocab
        self.captions = captions      # {image_name: report_text}
        self.tags = tags              # {pid_str: {finding: 0/1}}
        self.s_max = s_max
        self.n_max = n_max
        self.tokenizer = tokenizer    # Qwen tokenizer (qwen 模式)
        self.qwen_backend = qwen_backend
        self.struct_feats = struct_feats  # {pid_str: tensor(19)} 结构化征象(标准化)
        self.size_texts = size_texts      # {pid_str: str} 尺寸文本 ("长径18mm，短径35mm")
        self.size_max_len = size_max_len

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

        # 尺寸文本 token (精确数字走离散文字, 无 MLP 损耗)
        if self.qwen_backend and self.tokenizer is not None:
            size_str = (self.size_texts or {}).get(pid, "大小未测")
            enc = self.tokenizer(size_str, truncation=True,
                                 max_length=self.size_max_len,
                                 padding='max_length', return_tensors='pt')
            output["size_ids"] = enc['input_ids'][0]
            output["size_mask"] = enc['attention_mask'][0]
        else:
            output["size_ids"] = torch.zeros(self.size_max_len, dtype=torch.long)
            output["size_mask"] = torch.zeros(self.size_max_len, dtype=torch.long)

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
        # ---- 尺寸文本: 精确数字走文字 token (无 MLP 损耗) ----
        struct_feats, size_texts = {}, {}
        if gen_backend == 'qwen':
            from trainers.AnomalyDetect.report_generator_qwen import build_size_text as _bst
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
                    _pid = str(int(_row['影像号']))
                    struct_feats[_pid] = torch.tensor(_Xs[_i], dtype=torch.float)
                    size_texts[_pid] = _bst({'长径mm': _row.get('长径mm'),
                                             '短径mm': _row.get('短径mm')})
                print(f'[ReportGen] 结构化征象: {len(struct_feats)} 患者 x {_Xs.shape[1]} 维, '
                      f'尺寸文本: {len(size_texts)} 患者')
            else:
                print('[ReportGen] 未找到 STRUCT_CSV, 结构化征象/尺寸文本置空')

        # ---- 重建训练 loader (带报告) ----
        wrapper_cls = partial(ThymomaReportWrapper, vocab=self.vocab,
                              captions=captions, tags=tags, s_max=s_max, n_max=n_max,
                              tokenizer=qwen_tok, qwen_backend=(gen_backend == 'qwen'),
                              struct_feats=struct_feats, size_texts=size_texts)
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

        print(f'Loading BiomedCLIP ...')
        bclip, _ = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        bclip = bclip.float().eval()

        self.model = AnomalyCLIP(cfg, bclip)

        names_to_update = ["report_gen"]
        for n, p in self.model.named_parameters():
            p.requires_grad_(any(x in n for x in names_to_update))

        enabled = {n for n, p in self.model.named_parameters() if p.requires_grad}
        print(f'Trainable: {len(enabled)} — {enabled}')

        if cfg.MODEL.INIT_WEIGHTS:
            # 热启动: 按名加载 (torch2.6 weights_only 默认值会拒绝含调度器的检查点,
            # 文件为本项目自产, 显式 weights_only=False)
            _ckpt = torch.load(cfg.MODEL.INIT_WEIGHTS, map_location='cpu', weights_only=False)
            _sd = _ckpt['state_dict'] if 'state_dict' in _ckpt else _ckpt
            _model_sd = self.model.state_dict()
            _new = {k: v for k, v in _sd.items()
                    if k in _model_sd and _model_sd[k].shape == v.shape}
            print(f'INIT_WEIGHTS: 加载 {len(_new)}/{len(_sd)} 层 from {cfg.MODEL.INIT_WEIGHTS}')
            _model_sd.update(_new)
            self.model.load_state_dict(_model_sd)

        self.model.to(self.device)
        # 单一优化器: 只训生成器 (视觉投影 + 征象投影 + LoRA)
        gen_lr = float(getattr(cfg.TRAINER.ANOMALY_DETECT, 'GEN_LR', 0.0001))
        gen_params = [p for p in self.model.parameters() if p.requires_grad]
        wd = float(getattr(cfg.OPTIM, 'WEIGHT_DECAY', 0.0005))
        self.optim = torch.optim.Adam(gen_params, lr=gen_lr, weight_decay=wd)
        print(f'Optimizer: Adam (生成器专用) lr={gen_lr}, {len(gen_params)} params')
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
        struct = batch.get("struct", None)
        size_ids = batch.get("size_ids", None)
        size_mask = batch.get("size_mask", None)
        if captions is not None:
            captions = captions.to(self.device)
            prob_real = prob_real.to(self.device)
        if struct is not None:
            struct = struct.to(self.device)
        if size_ids is not None:
            size_ids = size_ids.to(self.device)
            size_mask = size_mask.to(self.device)

        out = model(image, captions, prob_real, struct, size_ids, size_mask)
        loss, s_img, gen_losses = out

        if getattr(self.model, 'gen_backend', '') == 'qwen':
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.report_gen.parameters(), max_norm=1.0)
            self.optim.step()
        else:
            self.model_backward_and_update(loss)

        ret = {"loss": loss.item(), "s_img_mean": s_img.mean().item()}
        if gen_losses is not None:
            for k, v in gen_losses.items():
                ret[f"gen_{k}"] = v.item()

        # 推进学习率调度器 (原版 BiomedCoOp 训练器同款钩子;
        # 缺失导致 lr 永远停在 warmup 值 1e-5)
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return ret

    def parse_batch_train(self, batch):
        img, label = batch["img"].to(self.device), batch["label"].to(self.device)
        return img, label

    def model_inference(self, input):
        return self.model(input)[0]  # s_img (分类由结构化分类器负责, 不在此模型)

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
            print(f'Loading {name} from {path} (epoch={ckpt.get("epoch")})')
            self._models[name].load_state_dict(state, strict=False)
