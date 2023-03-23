import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

from typing import List, Callable, Union, Any, TypeVar, Tuple


class TCNClassifier(pl.LightningModule):
    def __init__(
        self,
        input_shape,
        max_sequence_length: int,
        num_class=250,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_class = num_class
        self.max_sequence_length = max_sequence_length
        self.accuracy = MulticlassAccuracy(num_classes=num_class)

        self.tcn = nn.Sequential(
            nn.Conv1d(input_shape, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, dilation=4, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, dilation=8, padding=8),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, kernel_size=3, dilation=16, padding=16),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 2048, kernel_size=3, dilation=32, padding=32),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        # x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len)
        x = self.tcn(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the TCN
        out = self.fc(x)  # Pass the output through a linear layer
        return out

    def training_step(self, batch, batch_idx):
        input, target = batch
        out = self(input)

        loss = F.cross_entropy(out, target)
        y_pred = F.softmax(out, dim=1)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        accuracy = self.accuracy(y_pred.argmax(dim=1), target.argmax(dim=1))
        self.log(
            "train/accuracy",
            accuracy,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        out = self(input)

        loss = F.cross_entropy(out, target)
        y_pred = F.softmax(out, dim=1)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        accuracy = self.accuracy(y_pred.argmax(dim=1), target.argmax(dim=1))
        self.log(
            "val/accuracy",
            accuracy,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)

        return [optimizer]
