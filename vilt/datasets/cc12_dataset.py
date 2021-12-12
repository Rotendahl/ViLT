from glob import glob
from .base_dataset import BaseDataset
import os


class CC12Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        data_path = kwargs["config"]["data_root"]
        if split == "test":
            split = "val"

        if split == "train":
            names = [
                f.split(".")[0]
                for f in os.listdir(data_path)
                if "cc12" in f and "train" in f
            ]

        elif split == "val":
            names = [
                f.split(".")[0]
                for f in os.listdir(data_path)
                if "cc12" in f and "val" in f
            ]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        return self.get_suite(index)
