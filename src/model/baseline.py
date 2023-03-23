import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

from typing import List, Callable, Union, Any, TypeVar, Tuple


class SignModel(pl.LightningModule):
    def __init__(
        self,
        input_shape,
        hidden_units: List[int],
        max_sequence_length: int,
        num_class=250,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.num_class = num_class
        self.max_sequence_length = max_sequence_length
        self.classifier = self.get_classifier()
        self.accuracy = MulticlassAccuracy(num_classes=num_class)

    def get_classifier(self):
        classifier = []
        input_dim = self.input_shape
        final_dim = self.max_sequence_length
        for units in self.hidden_units:
            output_dim = units
            classifier.append(
                nn.Conv1d(
                    input_dim, output_dim, kernel_size=3, stride=1, padding="same"
                )
            )
            classifier.append(nn.BatchNorm1d(output_dim))
            classifier.append(nn.MaxPool1d(kernel_size=2, stride=2))
            # classifier.append(nn.Linear(final_dim, final_dim // 2))
            classifier.append(nn.LeakyReLU())
            input_dim = output_dim
            final_dim = final_dim // 2

        classifier.append(nn.Flatten())
        classifier.append(nn.Linear(output_dim * final_dim, self.num_class))
        classifier.append(nn.Softmax(dim=1))
        return nn.Sequential(*classifier)
        return classifier

    def forward(self, x):
        # print(x.shape)
        # for layer in self.classifier:
        #     x = layer(x)
        #     print(x.shape)
        # return x
        y_pred = self.classifier(x)
        return y_pred

    def training_step(self, batch, batch_idx):
        input, target = batch
        y_pred = self(input)

        loss = F.cross_entropy(y_pred, target)
        accuracy = self.accuracy(y_pred, target)
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
        y_pred = self(input)

        loss = F.cross_entropy(y_pred, target)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        accuracy = self.accuracy(y_pred, target)
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
