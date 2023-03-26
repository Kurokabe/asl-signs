import os
import pandas as pd
import numpy as np
from typing import List, Dict
from torch.utils.data import Dataset
from .preprocessing import preprocess_data


class SignDataset(Dataset):
    ROWS_PER_FRAME = 543  # number of landmarks per frame

    def __init__(
        self,
        max_sequence_length: int,
        normalize: bool,
        substract: bool,
        data_to_load: pd.DataFrame,
        label_to_index_mapping: Dict["str", int],
        root_dir: str = "/kaggle/input/asl-signs/",
    ):
        self.normalize = normalize
        self.substract = substract
        self.paths = [os.path.join(root_dir, path) for path in data_to_load["path"]]
        self.labels = pd.get_dummies(
            [label_to_index_mapping[sign] for sign in data_to_load["sign"]]
        ).values
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        labels = labels.astype(np.float32)
        x, y = self.load_relevant_data_subset(self.paths[idx]), labels

        x = preprocess_data(x, self.max_sequence_length, self.normalize, self.substract)
        return x, y

    def load_relevant_data_subset(self, pq_path):
        data_columns = ["x", "y", "z"]
        data = pd.read_parquet(pq_path, columns=data_columns)
        n_frames = int(len(data) / SignDataset.ROWS_PER_FRAME)
        data = data.values.reshape(
            n_frames, SignDataset.ROWS_PER_FRAME, len(data_columns)
        )
        data = data.astype(np.float32)
        return data
