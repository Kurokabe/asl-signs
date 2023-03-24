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
        trial,
        num_class=250,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_class = num_class
        self.max_sequence_length = max_sequence_length
        self.accuracy = MulticlassAccuracy(num_classes=num_class)
        self.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        self.tcn = []
        n_layers = trial.suggest_int("n_layers", 2, 8)
        for i in range(n_layers):
            output_shape = trial.suggest_categorical(
                f"channel_{i}", [64, 128, 256, 512, 1024, 2048]
            )
            kernel_size = trial.suggest_categorical(f"kernel_size_{i}", [3, 5, 7])
            dilation = trial.suggest_categorical(f"dilation_{i}", [1, 2, 4, 8, 16, 32])
            padding = (kernel_size - 1) * dilation
            self.tcn.append(
                nn.Conv1d(
                    input_shape,
                    output_shape,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.tcn.append(nn.BatchNorm1d(output_shape))
            self.tcn.append(nn.ReLU())
            input_shape = output_shape
        self.tcn.append(nn.AdaptiveAvgPool1d(1))
        self.tcn = nn.Sequential(*self.tcn)
        self.fc = nn.Linear(input_shape, num_class)

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
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        return [optimizer]
