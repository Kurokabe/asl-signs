import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')

from torch import nn
from abc import abstractmethod

LANDMARK_IDX = [0,9,11,13,14,17,117,118,119,199,346,347,348] + list(range(468,543))
MAX_LENGTH = 64


class SignModel(pl.LightningModule):
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List = [128, 256],
                 num_class=250,
                 num_landmark=543):
        super().__init__()
        
        classifier = []
        for units in hidden_dims:
            output_dim = units
            classifier.append(nn.Linear(input_dim, output_dim))
            classifier.append(nn.BatchNorm1d(output_dim))
            classifier.append(nn.LeakyReLU())
            input_dim = output_dim
        classifier.append(nn.Linear(output_dim, num_class))
        classifier.append(nn.Softmax(dim=1))
        self.classifier = nn.Sequential(*classifier)
        self.accuracy = MulticlassAccuracy(num_classes=num_class)
        
    
        
    def forward(self, x):
        y_pred = self.classifier(x)
        return y_pred
    
    def training_step(self, batch, batch_idx):
        input, target = batch
        y_pred = self(input)
        

        loss = F.cross_entropy(y_pred, target)
        
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        
        
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

        optimizer = Adam(self.parameters(), lr=2e-5)

        return [optimizer]