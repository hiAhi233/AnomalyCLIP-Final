"""胸腺瘤 CT 数据集（8 例 demo，与 BUSI 格式兼容）"""

import os, pickle, random
from collections import defaultdict
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing, listdir_nohidden


@DATASET_REGISTRY.register()
class Thymoma(DatasetBase):
    dataset_dir = "Thymoma"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir)  # 直接用顶层（含子目录）
        self.split_path = os.path.join(self.dataset_dir, "split_Thymoma.json")

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
        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.6, p_val=0.2, ignored=[], new_cnames=None):
        categories = [d for d in listdir_nohidden(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
        categories = [c for c in categories if c not in ignored]
        categories.sort()
        p_tst = 1 - p_trn - p_val
        print(f"Thymoma: {len(categories)} classes → {categories}")

        def _collate(ims, y, c):
            return [Datum(impath=im, label=y, classname=c) for im in ims]

        # ---- 按患者分组 (P<id>_ 前缀), 患者级划分避免同一患者跨集泄漏 ----
        import re
        patient_slices = defaultdict(list)   # pid -> [(impath, label, classname)]
        patient_labels = {}
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            for f in listdir_nohidden(category_dir):
                if not f.lower().endswith('.png'):
                    continue
                m = re.match(r'P(\d+)_', f)
                pid = int(m.group(1)) if m else -abs(hash(f))  # 无ID兜底: 每图独立
                impath = os.path.join(category_dir, f)
                patient_slices[pid].append((impath, label, category))
                patient_labels[pid] = (label, category)

        pids = list(patient_slices.keys())
        random.shuffle(pids)
        n_total = len(pids)
        n_train = max(1, round(n_total * p_trn))
        n_val = max(1, round(n_total * p_val))
        print(f"Thymoma: {n_total} 患者 → train {n_train} / val {n_val} / test {n_total - n_train - n_val}")

        def _collect(pid_list):
            out = []
            for pid in pid_list:
                label, cls = patient_labels[pid]
                cname = new_cnames[cls] if (new_cnames and cls in new_cnames) else cls
                for impath, _, _ in patient_slices[pid]:
                    out.append(Datum(impath=impath, label=label, classname=cname))
            return out

        train = _collect(pids[:n_train])
        val = _collect(pids[n_train:n_train + n_val])
        test = _collect(pids[n_train + n_val:])

        return train, val, test

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
        def _convert(items):
            return [Datum(impath=os.path.join(path_prefix, impath), label=int(label), classname=classname)
                    for impath, label, classname in items]
        split = read_json(filepath)
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
