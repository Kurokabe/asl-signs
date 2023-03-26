from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

import os
import pandas as pd
import torch
import numpy as np
from typing import List, Dict
from torch.utils.data import Dataset
import json

LANDMARK_IDX = [0,9,11,13,14,17,117,118,119,199,346,347,348] + list(range(468,543))
MAX_LENGTH = 64

data_to_load = pd.read_csv("/ASL/input/asl-signs/train.csv")
with open("/ASL/input/asl-signs/sign_to_prediction_index_map.json", "r") as f:
    label_to_index_mapping = json.load(f)

class SignDataset(Dataset):
    ROWS_PER_FRAME = 543  # number of landmarks per frame
    def __init__(self, data_to_load: pd.DataFrame, label_to_index_mapping: Dict["str", int], root_dir: str = "/kaggle/input/asl-signs/"):
        self.paths = [os.path.join(root_dir, path) for path in data_to_load["path"]]
        self.labels = pd.get_dummies([label_to_index_mapping[sign] for sign in data_to_load["sign"]]).values
        self.face_landmarks = list(range(0, 468))
        self.left_hand_landmarks = list(range(468, 489))
        self.pose_landmarks = list(range(489, 522))
        self.right_hand_landmarks = list(range(522, 543))
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        labels = self.labels[idx]
        labels = labels.astype(np.float32)
        x, y = self.load_relevant_data_subset(self.paths[idx]), labels

        x = np.nan_to_num(x)
        x = self.normalize(x)
        return x, y
    
    def load_relevant_data_subset(self, pq_path):
        data_columns = ['x', 'y', 'z']
        data = pd.read_parquet(pq_path, columns=data_columns)
        n_frames = int(len(data) / SignDataset.ROWS_PER_FRAME)
        data = data.values.reshape(n_frames, SignDataset.ROWS_PER_FRAME, len(data_columns))
        data = data.astype(np.float32)
        return data
    
    def normalize(self, data):
        # Face
        data[:, self.face_landmarks] = data[:, self.face_landmarks] - data[:, self.face_landmarks].mean(axis=0)
        data[:, self.face_landmarks] = data[:, self.face_landmarks] / (data[:, self.face_landmarks].std(axis=0) + 1e-5)

        # Left hand
        data[:, self.left_hand_landmarks] = data[:, self.left_hand_landmarks] - data[:, self.left_hand_landmarks].mean(axis=0)
        data[:, self.left_hand_landmarks] = data[:, self.left_hand_landmarks] / (data[:, self.left_hand_landmarks].std(axis=0) + 1e-5)

        # Pose
        data[:, self.pose_landmarks] = data[:, self.pose_landmarks] - data[:, self.pose_landmarks].mean(axis=0)
        data[:, self.pose_landmarks] = data[:, self.pose_landmarks] / (data[:, self.pose_landmarks].std(axis=0) + 1e-5)

        # Right hand
        data[:, self.right_hand_landmarks] = data[:, self.right_hand_landmarks] - data[:, self.right_hand_landmarks].mean(axis=0)
        data[:, self.right_hand_landmarks] = data[:, self.right_hand_landmarks] / (data[:, self.right_hand_landmarks].std(axis=0) + 1e-5)
        return data

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.shape[dim]
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate:

    def __init__(self, max_length, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.max_length = max_length
        self.dim = dim

    def pad_collate(self, batch):
        x, y = list(zip(*batch))
        x = [pad_tensor(torch.tensor(tensor[:self.max_length]), self.max_length, self.dim) for tensor in x]
        y = [torch.tensor(labels) for labels in y]
        x = torch.stack(x)
        y = torch.stack(y)
        return x, y
            
#         sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
#         shorted_sequence = batch[-1][0].shape[0]
        
#         x = [torch.tensor(b[0][:shorted_sequence]) for b in sorted_batch]
#         y = [torch.tensor(b[1]) for b in sorted_batch]
        
#         padded_x = pad_sequence(x, batch_first=True)
# #         packed = pack_padded_sequence(padded_x, batch_first=True, lengths=[b[0].shape[0] for b in sorted_batch])

#         return padded_x, torch.stack(y)

    def __call__(self, batch):
        return self.pad_collate(batch)
    
class SignDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str = "/ASL/input/asl-signs/", batch_size=4, num_workers=1, prefetch_factor=2):
        super().__init__()
        train, val = train_test_split(data_to_load, stratify=data_to_load["sign"])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        self.train_dataset = SignDataset(train, label_to_index_mapping, root_dir)
        self.val_dataset = SignDataset(val, label_to_index_mapping, root_dir)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True, 
            collate_fn=PadCollate(max_length=MAX_LENGTH, dim=0), 
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            collate_fn=PadCollate(max_length=MAX_LENGTH, dim=0), 
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            drop_last=True)
