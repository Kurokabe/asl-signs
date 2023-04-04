import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from tsai.all import MiniRocketPlus

from typing import List, Callable, Union, Any, TypeVar, Tuple, OrderedDict


class MiniRocketHead(nn.Module):
    def __init__(self, c_in, c_out, seq_len=1, fc_dropout=0.0, lstm_cells=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=c_in,  # 45, see the data definition
            hidden_size=lstm_cells,  # Can vary
            dropout=fc_dropout,
        )

        self.linear = nn.Linear(lstm_cells, c_out)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x)


class SignModel(pl.LightningModule):
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
        self.classifier = self.get_classifier()
        self.accuracy = MulticlassAccuracy(num_classes=num_class)

    def get_classifier(self):
        return MiniRocketPlus(
            c_in=self.input_shape,
            c_out=self.num_class,
            seq_len=self.max_sequence_length,
            fc_dropout=0.2,
            num_features=100,
            max_dilations_per_kernel=16,
            # custom_head=MiniRocketHead(
            #     49980, self.num_class, self.max_sequence_length, 0.2, 128
            # ),
        ).to(self.device)

    def forward(self, x):
        # print(x.shape)
        # for layer in self.classifier:
        #     x = layer(x)
        #     print(x.shape)
        # return x
        out = self.classifier(x)
        return out

    def training_step(self, batch, batch_idx):
        input, target = batch
        out = self(input)

        loss = F.cross_entropy(out, target)

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        y_pred = F.softmax(out, dim=1)
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
