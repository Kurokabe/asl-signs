import sys

sys.path.append("/ASL/src")
import json
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data.dataset import SignDataset


class SignDataModule(pl.LightningDataModule):
    def __init__(
        self,
        max_sequence_length: int,
        normalize: bool,
        substract: bool,
        root_dir: str = "/ASL/input/asl-signs/",
        batch_size=4,
        num_workers=1,
        prefetch_factor=2,
    ):
        super().__init__()
        data_to_load = pd.read_csv(os.path.join(root_dir, "train.csv"))

        with open(
            os.path.join(root_dir, "sign_to_prediction_index_map.json"), "r"
        ) as f:
            label_to_index_mapping = json.load(f)

        train, val = train_test_split(data_to_load, stratify=data_to_load["sign"])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.train_dataset = SignDataset(
            max_sequence_length,
            normalize,
            substract,
            train,
            label_to_index_mapping,
            root_dir,
        )
        self.val_dataset = SignDataset(
            max_sequence_length,
            normalize,
            substract,
            val,
            label_to_index_mapping,
            root_dir,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            drop_last=True,
        )
