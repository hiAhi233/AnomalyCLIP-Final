"""
report_generator_qwen.py — Qwen2.5-0.5B 医学报告生成器 (多视觉词元版)
================================================================
LLaVA 模式: 196 个 patch 特征按空间布局池化成 K 个视觉词元 (默认 64 = 8x8),
另有 19 维结构化征象经 MLP 投影成 K2 个征象词元, 拼在文本 token 前面。
生成器由此拥有空间信息 + 征象信息 (大小/强化/钙化)。

训练: LoRA 微调 (默认 r=4, 只训投影层 + LoRA 适配器, 冻结主干)
推理: 离线自回归生成 + 停止词截断 + 诊断意见首句收尾
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_size_text(feat: dict) -> str:
    """尺寸以文字注入 (精确数字走离散 token, 无 MLP 损耗)
    feat: {'长径mm':..., '短径mm':...}
    """
    parts = []
    ld = feat.get('长径mm')
    sd = feat.get('短径mm')
    if ld is not None and not (isinstance(ld, float) and ld != ld):
        parts.append(f"长径{ld:.0f}mm")
    if sd is not None and not (isinstance(sd, float) and sd != sd):
        parts.append(f"短径{sd:.0f}mm")
    return '，'.join(parts) if parts else "大小未测"


class QwenReportGenerator(nn.Module):
    def __init__(self, model_path='models/qwen2.5-0.5b', patch_dim=768,
                 vis_tokens=64, struct_dim=19, struct_tokens=4, size_max_len=32,
                 lora_r=4, lora_alpha=8, lora_dropout=0.1,
                 max_length=256, gen_max_tokens=110, temperature=0.4,
                 stop_words=("手术", "病理", "免疫组化", "Assistant", "病史"), device=None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.gen_max_tokens = gen_max_tokens
        self.temperature = temperature
        self.stop_words = stop_words
        self.vis_tokens = vis_tokens
        self.grid = int(vis_tokens ** 0.5)  # 64 → 8x8
        self.struct_dim = struct_dim
        self.struct_tokens = struct_tokens
        self.size_max_len = size_max_len

        # ---- 加载 Qwen (bf16, 冻结主干) ----
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True).to(self.device)

        self.embed_dim = self.qwen.config.hidden_size  # 896 for 0.5B

        # 冻结 Qwen 全部参数
        for p in self.qwen.parameters():
            p.requires_grad = False

        # ---- 视觉投影层 (可训练, bf16 对齐 Qwen) ----
        self.visual_proj = nn.Sequential(
            nn.Linear(patch_dim, self.embed_dim),
        ).to(self.device).bfloat16()

        # ---- 结构化征象投影: 19 维测量值 → K 个征象词元 ----
        # (大小/CT值/强化/钙化等 — 报告句子的内容来源, 治尺寸编造)
        self.struct_proj = nn.Sequential(
            nn.Linear(struct_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim * struct_tokens),
        ).to(self.device).bfloat16()

        # ---- LoRA 适配器 (默认 r=4, 容量受限 → 治背诵) ----
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none", task_type=TaskType.CAUSAL_LM,
        )
        self.qwen = get_peft_model(self.qwen, lora_config)

        self.tokenizer.pad_token = self.tokenizer.eos_token

    # ----------------------------------------------------------
    # 视觉编码: [B, 196, patch_dim] → [B, vis_tokens, embed_dim]
    # 保持 14x14 → 8x8 空间布局 (自适应平均池化), 而非无序压平
    # ----------------------------------------------------------
    def _visual_emb(self, visual_patches):
        B = visual_patches.shape[0]
        h = w = int(visual_patches.shape[1] ** 0.5)  # 14
        proj = self.visual_proj(visual_patches.bfloat16())   # [B, 196, D]
        proj = proj.view(B, h, w, -1).permute(0, 3, 1, 2)    # [B, D, 14, 14]
        pooled = F.adaptive_avg_pool2d(proj, (self.grid, self.grid))  # [B, D, g, g]
        return pooled.flatten(2).permute(0, 2, 1)            # [B, K, D]

    def _struct_emb(self, struct_feat):
        """struct_feat: [B, 19] 标准化后的结构化征象 → [B, K, D]"""
        proj = self.struct_proj(struct_feat.bfloat16())            # [B, K*D]
        return proj.view(-1, self.struct_tokens, self.embed_dim)  # [B, K, D]

    # ----------------------------------------------------------
    # 教师强制训练: [视觉 | 尺寸文本 | 征象词元 | 报告] → 下一个 token
    # ----------------------------------------------------------
    def forward(self, visual_patches, size_ids, size_mask, struct_feat, input_ids, attention_mask):
        """
        visual_patches: [B, 196, 768]  trunk 原始 patch token (全图, 不再用掩膜图)
        size_ids:       [B, S1]       尺寸文本 token (pad 填充)
        size_mask:      [B, S1]
        struct_feat:    [B, 19]        结构化征象 (标准化后)
        input_ids:      [B, L]         报告 token 序列 (非代表层 → 全 pad)
        attention_mask: [B, L]         非代表层 → 全 0
        Returns: 标量交叉熵 (全批无报告时返回零)
        """
        B = visual_patches.shape[0]

        vis_emb = self._visual_emb(visual_patches)                    # [B, V, D]
        size_emb = self.qwen.get_input_embeddings()(size_ids)        # [B, S1, D]
        struct_emb = self._struct_emb(struct_feat)                    # [B, K, D]
        text_emb = self.qwen.get_input_embeddings()(input_ids)       # [B, L, D]

        inputs_embeds = torch.cat([vis_emb, size_emb, struct_emb, text_emb], dim=1)
        vis_mask = torch.ones(B, vis_emb.shape[1], dtype=attention_mask.dtype, device=self.device)
        n_pre = vis_emb.shape[1] + size_emb.shape[1] + struct_emb.shape[1]
        pre_mask = torch.cat([vis_mask, size_mask,
                              torch.ones(B, struct_emb.shape[1], dtype=attention_mask.dtype, device=self.device)], dim=1)
        attn = torch.cat([pre_mask, attention_mask], dim=1)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels = torch.cat([
            torch.full((B, n_pre), -100, dtype=labels.dtype, device=self.device),
            labels], dim=1)

        # 全批都是非代表层 (无报告): 返回零损失, 保持梯度图连通
        if not (labels != -100).any():
            return torch.zeros((), device=self.device, requires_grad=True)

        outputs = self.qwen(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
        return outputs.loss

    # ----------------------------------------------------------
    # 推理: 图像 + 尺寸文本 + 征象 → 报告
    # ----------------------------------------------------------
    @torch.no_grad()
    def generate(self, visual_patches, size_texts, struct_feat, max_new_tokens=None, temperature=None):
        """
        visual_patches: [B, 196, 768]
        size_texts:     list[str] 尺寸文本 (长度 B)
        struct_feat:    [B, 19]
        Returns: list[str]
        """
        self.qwen.eval()
        B = visual_patches.shape[0]
        max_new_tokens = max_new_tokens or self.gen_max_tokens
        temperature = temperature if temperature is not None else self.temperature

        vis_emb = self._visual_emb(visual_patches)                # [B, V, D]
        vis_mask = torch.ones(B, vis_emb.shape[1], dtype=torch.long, device=self.device)

        # 尺寸文本 tokenize
        enc = self.tokenizer(size_texts, padding=True, truncation=True,
                             max_length=self.size_max_len, return_tensors='pt')
        size_ids = enc['input_ids'].to(self.device)
        size_mask = enc['attention_mask'].to(self.device)
        size_emb = self.qwen.get_input_embeddings()(size_ids)

        struct_emb = self._struct_emb(struct_feat)                # [B, K, D]
        struct_mask = torch.ones(B, struct_emb.shape[1], dtype=torch.long, device=self.device)

        # 直启式提示: 给报告起始语让模型续写, 避免 Qwen 进入聊天模式
        prompt = "影像所见："
        prompt_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)
        prompt_emb = self.qwen.get_input_embeddings()(prompt_ids).repeat(B, 1, 1)
        prompt_mask = torch.ones(B, prompt_emb.shape[1], dtype=torch.long, device=self.device)

        inputs_embeds = torch.cat([vis_emb, size_emb, struct_emb, prompt_emb], dim=1)
        attn = torch.cat([vis_mask, size_mask, struct_mask, prompt_mask], dim=1)

        out_ids = self.qwen.generate(
            inputs_embeds=inputs_embeds, attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        texts = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        # 截断策略:
        #   1) 遇病理段标记即停 (训练文本只含影像所见+诊断意见)
        #   2) 保留到"诊断意见"第一句的句号为止 — 所见+意见完整, 其后漂移全丢
        cleaned = []
        for t in texts:
            for sw in self.stop_words:
                i = t.find(sw)
                if i > 0:
                    t = t[:i]
            j = t.find('诊断意见')
            if j >= 0:
                k = t.find('。', j)
                t = t[:k + 1] if k > 0 else t[:j]
            else:
                k = t.rfind('。')
                t = t[:k + 1] if k > 0 else t
            cleaned.append(t.strip())
        return cleaned


# ============================================================
# 自测
# ============================================================

if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    gen = QwenReportGenerator()
    print(f'Qwen embed dim: {gen.embed_dim}')

    # 训练模式自测
    B, L, S1 = 2, 32, 16
    patches = torch.randn(B, 196, 768).to(gen.device)
    size_ids = torch.randint(100, 1000, (B, S1)).to(gen.device)
    size_mask = torch.ones(B, S1, dtype=torch.long).to(gen.device)
    struct = torch.randn(B, 19).to(gen.device)
    ids = torch.randint(100, 1000, (B, L)).to(gen.device)
    mask = torch.ones(B, L, dtype=torch.long).to(gen.device)
    loss = gen(patches, size_ids, size_mask, struct, ids, mask)
    print(f'train loss: {loss.item():.4f}')
    loss.backward()
    print('forward+backward OK')

    # 全空批自测 (非代表层)
    loss0 = gen(patches, size_ids, size_mask, struct, ids, torch.zeros_like(mask))
    print(f'empty-batch loss: {loss0.item():.4f}')

    # 推理自测
    texts = gen.generate(patches[:1], ["长径18mm，短径35mm"], struct[:1], max_new_tokens=50)
    print(f'generated: {texts[0][:200]}')
