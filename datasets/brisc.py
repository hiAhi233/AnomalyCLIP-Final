import os
import pickle
import math
import random

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing, listdir_nohidden


@DATASET_REGISTRY.register()
class BRISC(DatasetBase):

    dataset_dir = "BRISC"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "BRISC")

        self.split_path = os.path.join(self.dataset_dir, "split_BRISC.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # ------------------------------------------------
        # Load or create split
        # ------------------------------------------------
        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_and_split_data(self.image_dir)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        # ------------------------------------------------
        # Few-shot
        # ------------------------------------------------
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir,
                f"shot_{num_shots}-seed_{seed}.pkl"
            )

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))

                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # ------------------------------------------------
        # Subsample classes (base / new / all)
        # ------------------------------------------------
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(
            train, val, test, subsample=subsample
        )

        super().__init__(train_x=train, val=val, test=test)

    # ============================================================
    # Utilities (identical pattern to KneeXray)
    # ============================================================

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                out.append(Datum(impath=impath, label=int(label), classname=classname))
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])
        return train, val, test

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, item.label, item.classname))
            return out

        split = {
            "train": _extract(train),
            "val": _extract(val),
            "test": _extract(test),
        }

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2):
        categories = listdir_nohidden(image_dir)
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, {p_tst:.0%} test")

        def _collate(imgs, label, cname):
            return [Datum(impath=im, label=label, classname=cname) for im in imgs]

        train, val, test = [], [], []
        for label, cname in enumerate(categories):
            cdir = os.path.join(image_dir, cname)
            imgs = [os.path.join(cdir, im) for im in listdir_nohidden(cdir)]
            random.shuffle(imgs)

            n = len(imgs)
            n_tr = round(n * p_trn)
            n_val = round(n * p_val)

            train.extend(_collate(imgs[:n_tr], label, cname))
            val.extend(_collate(imgs[n_tr:n_tr + n_val], label, cname))
            test.extend(_collate(imgs[n_tr + n_val:], label, cname))

        return train, val, test

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        labels = sorted({item.label for item in args[0]})
        m = math.ceil(len(labels) / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES")
        selected = labels[:m] if subsample == "base" else labels[m:]
        relabel = {y: i for i, y in enumerate(selected)}

        outputs = []
        for dataset in args:
            filtered = [
                Datum(impath=i.impath, label=relabel[i.label], classname=i.classname)
                for i in dataset if i.label in selected
            ]
            outputs.append(filtered)

        return outputs
