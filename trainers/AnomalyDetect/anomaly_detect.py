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

import os.path as osp
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

    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()

        # ---- extract features ----
        image_features, patch_tokens = self._extract_patch_tokens(image)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        # ---- Path A (classification) ----
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features_norm @ text_features.t()

        # ---- Path B (anomaly detection) ----
        patch_features = self.patch_proj(patch_tokens)      # [B, 196, 768]
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True).clamp(min=1e-8)

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
            apply = s_img > 0.4
            if apply.any():
                bg_suppressed = image * (mask_224 + self.bg_factor * (1 - mask_224))
                masked_image = torch.where(apply.view(-1,1,1,1), bg_suppressed, image)

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

            return logits, loss_ce, loss_sccm, loss_kdsp, loss_consist, s_img
        else:
            return logits, s_img, anomaly_scores, concept_scores, masked_image


# ============================================================
#  Trainer
# ============================================================

@TRAINER_REGISTRY.register()
class AnomalyDetect_BiomedCLIP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.ANOMALY_DETECT.PREC in ["fp16", "fp32", "amp"]

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

        names_to_update = ["prompt_learner.ctx", "patch_proj", "text_to_patch"]
        for n, p in self.model.named_parameters():
            p.requires_grad_(any(x in n for x in names_to_update))

        enabled = {n for n, p in self.model.named_parameters() if p.requires_grad}
        print(f'Trainable: {len(enabled)} — {enabled}')

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('anomaly_clip', self.model, self.optim, self.sched)
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.scaler = GradScaler() if cfg.TRAINER.ANOMALY_DETECT.PREC == 'amp' else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        prec = self.cfg.TRAINER.ANOMALY_DETECT.PREC

        if prec == 'amp':
            with autocast():
                loss = model(image, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits, loss_ce, loss_sccm, loss_kdsp, loss_consist, s_img = model(image, label)
            loss = loss_ce + loss_sccm + loss_kdsp + loss_consist

            self.model_backward_and_update(loss)

        return {"loss": loss.item(),
                "loss_ce": loss_ce.item(), "loss_consist": loss_consist.item(),
                "s_img_mean": s_img.mean().item(),
                "acc": compute_accuracy(logits, label)[0].item()}

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
