"""前纵隔占位 707 例 CT 数据集 (胸腺区占位_AI3.csv 构建)

数据目录结构:
  data/Mediastinum/{11个类别}/P{影像号}_增强_z{1..10}.png   (代表层 + 均匀层)
  data/Mediastinum/{类别}/P{影像号}_报告.txt / _rep.txt
  data/Mediastinum/split_Mediastinum.json   (患者级分层划分 70/15/15)
  data/Mediastinum/captions.json            (只挂代表层: 一图一文)
"""

import os, pickle, random, re
from collections import defaultdict
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, listdir_nohidden


@DATASET_REGISTRY.register()
class Mediastinum(DatasetBase):
    dataset_dir = "Mediastinum"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir
        self.split_path = os.path.join(self.dataset_dir, "split_Mediastinum.json")

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_and_split_data(self.image_dir)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            preprocessed = os.path.join(self.dataset_dir, f"shot_{num_shots}-seed_{cfg.SEED}.pkl")
            if os.path.exists(preprocessed):
                with open(preprocessed, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                with open(preprocessed, "wb") as f:
                    pickle.dump({"train": train, "val": val}, f, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        # 可选: 只用代表层 (病灶最大层, 一图一标) — 消除"均匀层无病灶却挂标签"的噪声
        if getattr(cfg.DATASET, 'REP_ONLY', False):
            reps = set()
            for label_dir in os.listdir(self.dataset_dir):
                d = os.path.join(self.dataset_dir, label_dir)
                if not os.path.isdir(d):
                    continue
                for f in os.listdir(d):
                    if f.endswith('_rep.txt'):
                        with open(os.path.join(d, f), 'r', encoding='utf-8') as fp:
                            rep_name = fp.read().strip()
                        reps.add(os.path.normpath(os.path.join(self.dataset_dir, label_dir, rep_name)))
            def _keep(items):
                return [x for x in items if os.path.normpath(x.impath) in reps]
            train, val, test = _keep(train), _keep(val), _keep(test)
            print(f'REP_ONLY: train {len(train)} / val {len(val)} / test {len(test)} (仅代表层)')

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.7, p_val=0.15, ignored=[], new_cnames=None):
        """兜底划分 (split json 缺失时): 按患者分层 70/15/15"""
        categories = [d for d in listdir_nohidden(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
        categories = [c for c in categories if c not in ignored]
        categories.sort()
        print(f"Mediastinum: {len(categories)} classes → {categories}")

        patient_slices = defaultdict(list)   # pid -> [(impath, label, category)]
        patient_labels = {}
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            for f in listdir_nohidden(category_dir):
                if not f.lower().endswith('.png'):
                    continue
                m = re.match(r'P(\d+)_', f)
                pid = int(m.group(1)) if m else -abs(hash(f))
                impath = os.path.join(category_dir, f)
                patient_slices[pid].append((impath, label, category))
                patient_labels[pid] = (label, category)

        # 按标签分层 (保证每类都进 val/test)
        by_label = defaultdict(list)
        for pid in patient_slices:
            by_label[patient_labels[pid][1]].append(pid)
        rng = random.Random(42)
        split_pids = {"train": [], "val": [], "test": []}
        for lab, ps in by_label.items():
            rng.shuffle(ps)
            n_t = max(1, int(len(ps) * p_trn))
            n_v = max(1, int(len(ps) * p_val)) if len(ps) > 2 else 0
            split_pids["train"] += ps[:n_t]
            split_pids["val"] += ps[n_t:n_t + n_v]
            split_pids["test"] += ps[n_t + n_v:]

        def _collect(pid_list):
            out = []
            for pid in pid_list:
                label, cls = patient_labels[pid]
                cname = new_cnames[cls] if (new_cnames and cls in new_cnames) else cls
                for impath, _, _ in patient_slices[pid]:
                    out.append(Datum(impath=impath, label=label, classname=cname))
            return out

        return (_collect(split_pids["train"]), _collect(split_pids["val"]),
                _collect(split_pids["test"]))

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath.replace(path_prefix, "").lstrip("/").lstrip("\\")
                out.append((impath, item.label, item.classname))
            return out
        split = {"train": _extract(train), "val": _extract(val), "test": _extract(test)}
        write_json(split, filepath)
        print(f"Saved split → {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        # 注意: split json 由 build 脚本以 UTF-8 写入, dassl 的 read_json
        # 用系统默认编码 (中文 Windows = GBK) 会读成乱码 → 显式 UTF-8
        import json as _json
        with open(filepath, "r", encoding="utf-8") as f:
            split = _json.load(f)

        def _convert(items):
            return [Datum(impath=os.path.join(path_prefix, impath), label=int(label), classname=classname)
                    for impath, label, classname in items]
        return _convert(split["train"]), _convert(split["val"]), _convert(split["test"])

    def generate_fewshot_dataset_(self, num_shots, split):
        if split == "train":
            return self.generate_fewshot_dataset(self.train_x, num_shots=num_shots)
        elif split == "val":
            return self.generate_fewshot_dataset(self.val, num_shots=num_shots)
        return []

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        if subsample == "all":
            return args
        dataset = args[0]
        labels = sorted(set(item.label for item in dataset))
        m = max(1, len(labels) // 2)
        selected = labels[:m] if subsample == "base" else labels[m:]
        relabeler = {y: i for i, y in enumerate(selected)}
        output = []
        for ds in args:
            ds_new = []
            for item in ds:
                if item.label in selected:
                    ds_new.append(Datum(impath=item.impath, label=relabeler[item.label], classname=item.classname))
            output.append(ds_new)
        return output
