"""
probe_lora_finetune.py — 判定实验: 微调视觉主干后, 特征能否学会"看纵隔占位"
================================================================
背景: 冻结 BiomedCLIP 特征对 11 类纵隔占位无判别力 (5折CV 16.8% ≈ 随机)。
本探针做"部分微调": 解冻 ViT 最后 4 个 block + 线性分类头,
用 707 例代表层训练 30 轮 (~10 分钟), 回答"补课后能不能学会"。

判据:
  test acc > 50%  → 特征微调可行, 全盘推进 (分类/报告/定位)
  test acc ≈ 25%  → 图像单独分不了 11 类, 转二分类/检测定位主线

Usage:
  python probe_lora_finetune.py
"""

import os, sys, json, re, warnings, argparse
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from open_clip.src.open_clip import create_model_from_pretrained

UNFREEZE_BLOCKS = 4        # 微调最后 N 个 ViT block
EPOCHS = 30
BATCH = 32
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3


def load_rep_split(data_dir='data/Mediastinum', binary=False):
    reps = set()
    for dp, _, fns in os.walk(data_dir):
        for f in fns:
            if f.endswith('_rep.txt'):
                with open(os.path.join(dp, f), encoding='utf-8') as fp:
                    reps.add(os.path.normpath(os.path.join(dp, fp.read().strip())))
    split = json.load(open(os.path.join(data_dir, 'split_Mediastinum.json'), encoding='utf-8'))
    out = {part: [] for part in split}
    for part in split:
        for rel, lab, cname in split[part]:
            p = os.path.normpath(os.path.join(data_dir, rel))
            if p in reps:
                if binary:
                    lab = 1 if cname == 'thymoma' else 0   # 胸腺瘤 vs 其他
                out[part].append((p, lab))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--unfreeze-blocks', type=int, default=UNFREEZE_BLOCKS)
    ap.add_argument('--binary', action='store_true', help='二分类: 胸腺瘤 vs 其他')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save', default='output/probe_lora')
    args = ap.parse_args()

    print('Loading BiomedCLIP ...')
    model, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.float().to(args.device)
    visual = model.visual

    # ---- 解冻最后 N 个 block ----
    blocks = visual.trunk.blocks
    n_blocks = len(blocks)
    print(f'ViT blocks: {n_blocks}, 解冻最后 {args.unfreeze_blocks} 个')
    for p in visual.parameters():
        p.requires_grad = False
    for i in range(n_blocks - args.unfreeze_blocks, n_blocks):
        for p in blocks[i].parameters():
            p.requires_grad = True

    n_cls = 2 if args.binary else 11
    head = nn.Linear(512, n_cls).to(args.device)
    opt = torch.optim.AdamW([
        {"params": [p for p in visual.parameters() if p.requires_grad], "lr": BACKBONE_LR},
        {"params": head.parameters(), "lr": HEAD_LR},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ---- 数据 ----
    split = load_rep_split(binary=args.binary)
    print(f'rep-only: train {len(split["train"])} / val {len(split["val"])} / test {len(split["test"])}')
    tfm_train = T.Compose([
        T.RandomResizedCrop(224, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    tfm_eval = T.Compose([
        T.Resize(224), T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    class DS(torch.utils.data.Dataset):
        def __init__(self, items, tfm):
            self.items, self.tfm = items, tfm
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            p, lab = self.items[i]
            img = Image.open(p).convert('RGB')
            return self.tfm(img), lab

    loaders = {
        part: torch.utils.data.DataLoader(DS(split[part], tfm_train if part == 'train' else tfm_eval),
                                          batch_size=BATCH, shuffle=(part == 'train'))
        for part in split
    }

    def evaluate(part):
        visual.eval(); head.eval()
        correct, total, preds, labels = 0, 0, [], []
        with torch.no_grad():
            for x, y in loaders[part]:
                x, y = x.to(args.device), y.to(args.device)
                f = visual(x)
                out = head(f)
                preds.append(out.argmax(1).cpu()); labels.append(y.cpu())
        preds = torch.cat(preds); labels = torch.cat(labels)
        from sklearn.metrics import f1_score
        return (preds == labels).float().mean().item() * 100, \
               f1_score(labels.numpy(), preds.numpy(), average='macro') * 100

    print(f'{"epoch":>6} {"train_acc":>10} {"val_acc":>9} {"test_acc":>9} {"test_mf1":>9}')
    best_test = 0
    for epoch in range(1, args.epochs + 1):
        visual.train(); head.train()
        corr = tot = 0
        for x, y in loaders['train']:
            x, y = x.to(args.device), y.to(args.device)
            out = head(visual(x))
            loss = nn.functional.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            corr += (out.argmax(1) == y).sum().item(); tot += len(y)
        sched.step()
        ta, va = evaluate('train'), evaluate('val')
        ea, em = evaluate('test')
        best_test = max(best_test, ea)
        print(f'{epoch:>6} {ta[0]:>9.1f}% {va[0]:>8.1f}% {ea:>8.1f}% {em:>8.1f}%')

    print(f'\n最佳 test acc: {best_test:.1f}%')
    print('对照: 冻结特征线性探针 test 17.4% / 5折CV 16.8%; 多数类基线 28.7%')

    os.makedirs(args.save, exist_ok=True)
    torch.save({'visual': visual.state_dict(), 'head': head.state_dict()},
               os.path.join(args.save, 'probe_model.pt'))
    print(f'Saved → {args.save}/probe_model.pt')


if __name__ == '__main__':
    main()
