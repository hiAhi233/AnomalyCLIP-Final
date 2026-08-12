"""
report_generator_net.py — 中文医学报告生成网络
================================================================
借鉴 Vinyals 2017 (On the Automatic Generation of Medical Imaging Reports)
的层级 LSTM 架构，适配:
  - 图像特征: BiomedCLIP ViT patch 特征 (768d)
  - 语言: 中文 (jieba 分词, vocab.pkl)
  - 征象: 9 个乳腺/胸腺征象多标签 (MLC)

模块:
  MLC          9 征象多标签分类 → top-k 语义特征
  CoAttention  视觉特征 + 征象语义 协同注意力 → 上下文 ctx
  SentenceLSTM 句子主题 topic + 停止判断 p_stop
  WordLSTM     逐词生成中文句子
  ReportGenNet 组合 (forward: teacher forcing / generate: Beam Search)

Usage (由 anomaly_detect.py 调用):
  from trainers.AnomalyDetect.report_generator_net import ReportGenNet
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


# ============================================================
# 征象定义 (与 report_scorer.py 一致)
# ============================================================

FINDING_ORDER = ["tumor_mass", "inflammation", "necrosis", "calcification",
                 "cyst_fluid", "fibrosis_scar", "normal_tissue",
                 "regular_architecture", "isoechoic_texture"]
N_FINDINGS = len(FINDING_ORDER)


# ============================================================
# 中文词表 (pickle 加载需要类在此定义)
# ============================================================

class Vocabulary:
    """中文医学报告词表 (与 build_report_corpus.py 共用)"""

    def __init__(self):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.word2idx)

    def get_word(self, idx):
        return self.id2word.get(idx, '<unk>')


def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab


# ============================================================
# MLC: 征象多标签分类
# ============================================================

class MLC(nn.Module):
    """9 征象多标签分类 → top-k 语义特征"""

    def __init__(self, visual_size=512, semantic_dim=512, k=5):
        super().__init__()
        self.classifier = nn.Linear(visual_size, N_FINDINGS)
        self.embed = nn.Embedding(N_FINDINGS, semantic_dim)
        self.k = k

    def forward(self, avg_features):
        # avg_features: [B, 768]
        logits = self.classifier(avg_features)          # [B, 9]
        tags = torch.sigmoid(logits)                    # 多标签概率
        topk_idx = torch.topk(tags, self.k, dim=-1)[1]  # [B, k]
        semantic_features = self.embed(topk_idx)        # [B, k, 512]
        return tags, semantic_features


# ============================================================
# Co-Attention: 视觉 + 征象语义协同注意力
# ============================================================

class CoAttention(nn.Module):
    def __init__(self, visual_size=512, semantic_dim=512, hidden_size=512,
                 embed_size=512, k=5):
        super().__init__()
        self.W_v = nn.Linear(visual_size, visual_size)
        self.W_v_h = nn.Linear(hidden_size, visual_size)
        self.W_v_att = nn.Linear(visual_size, 1)

        self.W_a = nn.Linear(semantic_dim, semantic_dim)
        self.W_a_h = nn.Linear(hidden_size, semantic_dim)
        self.W_a_att = nn.Linear(semantic_dim, 1)

        self.W_fc = nn.Linear(visual_size + semantic_dim, embed_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.k = k

    def forward(self, visual_features, semantic_features, h_sent):
        """
        visual_features:    [B, 768]  (病灶聚焦全局特征)
        semantic_features:  [B, k, 512]
        h_sent:             [B, 1, hidden] (上句隐藏状态)
        """
        # 视觉注意力
        W_v = self.tanh(self.W_v(visual_features) + self.W_v_h(h_sent.squeeze(1)))
        alpha_v = self.softmax(self.W_v_att(W_v))               # [B, 1]
        v_att = visual_features * alpha_v                       # [B, 768]

        # 语义注意力
        W_a = self.tanh(self.W_a(semantic_features) + self.W_a_h(h_sent))  # [B, k, 512]
        alpha_a = self.softmax(self.W_a_att(W_a))               # [B, k, 1]
        a_att = (semantic_features * alpha_a).sum(dim=1)        # [B, 512]

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))       # [B, 512]
        return ctx, alpha_v, alpha_a


# ============================================================
# SentenceLSTM: 句子主题 + 停止判断
# ============================================================

class SentenceLSTM(nn.Module):
    def __init__(self, embed_size=512, hidden_size=512, num_layers=1,
                 dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout)
        self.W_topic = nn.Linear(hidden_size, embed_size)
        self.W_stop = nn.Linear(hidden_size, 2)

    def forward(self, ctx, prev_hidden, states=None):
        # ctx: [B, 512], prev_hidden: [B, 1, hidden]
        ctx = ctx.unsqueeze(1)                                  # [B, 1, 512]
        hidden, states = self.lstm(ctx, states)                 # [B, 1, hidden]
        topic = self.W_topic(hidden)                            # [B, 1, 512]
        p_stop = self.W_stop(hidden.squeeze(1))                 # [B, 2]
        return topic, p_stop, hidden, states


# ============================================================
# WordLSTM: 逐词生成
# ============================================================

class WordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=512, hidden_size=512,
                 num_layers=1, n_max=50):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.n_max = n_max
        self.vocab_size = vocab_size

    def forward(self, topic_vec, captions):
        """
        topic_vec:  [B, 512] (当前句主题, squeeze 后)
        captions:   [B, n_max] (句内词序列, 含 <start>)
        """
        embeddings = self.embed(captions)                       # [B, n_max, 512]
        # 主题向量作为第一个时间步输入
        topic = topic_vec.unsqueeze(1)                          # [B, 1, 512]
        embeddings = torch.cat([topic, embeddings], dim=1)[:, :-1, :]  # teacher: 前 t 词 → 预测 t+1
        hidden, _ = self.lstm(embeddings)                       # [B, n_max, hidden]
        outputs = self.linear(hidden)                           # [B, n_max, vocab]
        return outputs

    def sample(self, topic_vec, start_token, end_token=None, unk_token=None,
               n_max=None, temperature=0.8):
        """自回归生成一句 (推理用, temperature 采样打破死循环, <unk> 屏蔽)"""
        n_max = n_max or self.n_max
        batch_size = topic_vec.shape[0]
        device = topic_vec.device

        sampled_ids = torch.zeros(batch_size, n_max, dtype=torch.long, device=device)
        sampled_ids[:, 0] = start_token

        hidden = None
        # 初始输入: 主题向量 [B, 1, 512]
        inp = topic_vec.unsqueeze(1)
        for i in range(1, n_max):
            out, hidden = self.lstm(inp, hidden)
            logits = self.linear(out[:, -1, :])                 # [B, vocab]
            # 屏蔽 <unk> 和 <pad> token (模型不会生成这些, 强迫选真实词)
            if unk_token is not None:
                logits[:, unk_token] = -1e9
            logits[:, 0] = -1e9  # <pad> always masked
            # temperature 采样: 打破贪心解码的重复循环
            probs = torch.softmax(logits / temperature, dim=-1)
            pred = torch.multinomial(probs, 1).squeeze(-1)      # [B]
            sampled_ids[:, i] = pred
            inp = self.embed(pred.unsqueeze(1))                 # [B, 1, 512]
            # 遇到 <end> 停止
            if end_token is not None and (pred == end_token).all():
                break
        return sampled_ids


# ============================================================
# ReportGenNet: 组合
# ============================================================

class ReportGenNet(nn.Module):
    def __init__(self, vocab_size, visual_size=512, embed_size=512,
                 hidden_size=512, k=5, s_max=8, n_max=50):
        super().__init__()
        self.mlc = MLC(visual_size, embed_size, k)
        self.co_attention = CoAttention(visual_size, embed_size, hidden_size, embed_size, k)
        self.sentence_model = SentenceLSTM(embed_size, hidden_size)
        self.word_model = WordLSTM(vocab_size, embed_size, hidden_size, n_max=n_max)
        self.s_max = s_max
        self.n_max = n_max

    def forward(self, avg_features, captions, prob_real, tag_target=None):
        """
        teacher forcing 训练

        avg_features: [B, 768] 病灶聚焦全局特征
        captions:     [B, s_max, n_max] 报告分句 token (含 <start><end>)
        prob_real:    [B, s_max] 每句真实存在标记 (1=有句, 0=padding)
        tag_target:   [B, 9] 征象 one-hot (可选, 有则算 MLC loss)

        Returns:
            dict: {loss_tag, loss_stop, loss_word}
        """
        B = avg_features.shape[0]

        # MLC 征象
        tags, semantic_features = self.mlc(avg_features)         # [B, 9], [B, k, 512]
        loss_tag = torch.tensor(0.0, device=avg_features.device)
        if tag_target is not None:
            loss_tag = F.binary_cross_entropy(tags, tag_target)

        # 句子循环
        sentence_states = None
        prev_hidden = torch.zeros(B, 1, self.sentence_model.lstm.hidden_size,
                                  device=avg_features.device)

        loss_stop = torch.tensor(0.0, device=avg_features.device)
        loss_word = torch.tensor(0.0, device=avg_features.device)

        for s in range(min(self.s_max, captions.shape[1])):
            ctx, _, _ = self.co_attention(avg_features, semantic_features, prev_hidden)
            topic, p_stop, hidden, sentence_states = \
                self.sentence_model(ctx, prev_hidden, sentence_states)

            # 停止 loss (只对真实存在的句子算)
            mask = prob_real[:, s].float()
            if mask.sum() > 0:
                loss_stop = loss_stop + (F.cross_entropy(
                    p_stop, prob_real[:, s].long(), reduction='none') * mask).sum()

            # 词 loss (teacher forcing, 前 t 词 → 第 t+1 词)
            sent = captions[:, s, :]                             # [B, n_max]
            word_mask = (sent > 0).float()
            if word_mask.sum() > 0:
                outputs = self.word_model(topic.squeeze(1), sent)  # [B, n_max, vocab]
                # 预测 t 位置 → 与 t+1 位置真实词对比 (错位)
                targets = sent
                shift_out = outputs[:, :-1, :]                   # 预测 t=0..n-2
                shift_tgt = targets[:, 1:]                       # 真实 t=1..n-1
                shift_mask = word_mask[:, 1:]
                if shift_mask.sum() > 0:
                    word_ce = F.cross_entropy(
                        shift_out.reshape(-1, shift_out.shape[-1]),
                        shift_tgt.reshape(-1), reduction='none')
                    word_ce = word_ce.reshape_as(shift_mask) * shift_mask
                    # 按有效 token 数归一化 (均值)
                    loss_word = loss_word + word_ce.sum() / shift_mask.sum()

            prev_hidden = hidden

        # loss 归一化: 按真实句子数
        n_sent = max(prob_real.sum().float().item(), 1)
        loss_stop = loss_stop / n_sent
        loss_word = loss_word / n_sent

        return {"loss_tag": loss_tag,
                "loss_stop": loss_stop,
                "loss_word": loss_word}

    def generate(self, avg_features, vocab, max_sentences=None, max_words=None):
        """
        推理生成 (temperature 采样, <unk> 屏蔽)

        avg_features: [B, 768]
        vocab: Vocabulary 对象 (含 word2idx)

        Returns:
            list[str]: 每样本生成的报告句子列表 (拼接后为报告)
        """
        B = avg_features.shape[0]
        max_sentences = max_sentences or self.s_max
        max_words = max_words or self.n_max

        start_idx = vocab('<start>')
        end_idx = vocab('<end>')
        unk_idx = vocab('<unk>')

        tags, semantic_features = self.mlc(avg_features)
        sentence_states = None
        prev_hidden = torch.zeros(B, 1, self.sentence_model.lstm.hidden_size,
                                  device=avg_features.device)

        all_reports = [[] for _ in range(B)]

        for _ in range(max_sentences):
            ctx, _, _ = self.co_attention(avg_features, semantic_features, prev_hidden)
            topic, p_stop, hidden, sentence_states = \
                self.sentence_model(ctx, prev_hidden, sentence_states)

            # 生成句子 (temperature=0.8 打破循环, 可调低提高确定性)
            sampled = self.word_model.sample(topic.squeeze(1),
                torch.full((B,), start_idx, device=avg_features.device),
                max_words, temperature=0.8)

            # 解码 + 判断停止 (训练时: prob_real=1=句存在, 0=无句子)
            # p_stop class 0 = 停止, class 1 = 继续下一句
            stop_all = True
            for b in range(B):
                stop_pred = torch.argmax(torch.softmax(p_stop[b], dim=-1)).item()
                if stop_pred == 0:   # 模型说"停"
                    continue         # 跳过该样本 (已停止)
                # 模型说"继续" → 有下一句
                stop_all = False
                words = []
                for w in sampled[b].tolist():
                    if w == end_idx or w == vocab('<pad>'):
                        break
                    if w != start_idx:
                        words.append(vocab.get_word(w))
                if words:
                    all_reports[b].append(''.join(words))

            prev_hidden = hidden
            if stop_all:
                break

        # 拼接: 句号断句 + 每两句换行分段
        final = []
        for sents in all_reports:
            lines = []
            for i, s in enumerate(sents):
                lines.append(s + '。')
                if (i + 1) % 2 == 0:
                    lines.append('\n')
            final.append(''.join(lines))
        return final


# ============================================================
# 快速自测
# ============================================================

if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    vocab_size = 1000
    net = ReportGenNet(vocab_size=vocab_size, s_max=8, n_max=20)

    B = 2
    avg_feat = torch.randn(B, 768)
    captions = torch.randint(0, vocab_size, (B, 8, 20)).long()
    captions[:, :, 0] = 1  # <start>
    prob_real = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    tags = torch.rand(B, 9)

    losses = net(avg_feat, captions, prob_real, tags)
    for k, v in losses.items():
        print(f'{k}: {v.item():.4f}')
    print('ReportGenNet forward OK')

    # 推理测试
    class FakeVocab:
        def __init__(self):
            self.w2i = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
            self.i2w = {v: k for k, v in self.w2i.items()}
            for i in range(4, vocab_size):
                self.w2i[f'w{i}'] = i
                self.i2w[i] = f'w{i}'
        def __call__(self, w):
            return self.w2i.get(w, 3)
        def get_word(self, i):
            return self.i2w.get(i, '<unk>')

    reports = net.generate(avg_feat, FakeVocab(), max_sentences=3, max_words=10)
    print(f'生成: {reports}')
