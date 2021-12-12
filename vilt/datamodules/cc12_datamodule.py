from vilt.datasets import CC12Dataset
from .datamodule_base import BaseDataModule


class CC12DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CC12Dataset

    @property
    def dataset_name(self):
        return "cc12"
