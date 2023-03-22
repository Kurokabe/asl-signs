import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from typing import Tuple, List


LANDMARK_IDX = [0,9,11,13,14,17,117,118,119,199,346,347,348] + list(range(468,543))
MAX_LENGTH = 64

class SignModel(pl.LightningModule):
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = [128, 256],
                 classifier_units: List = [128, 256],
                 num_class=250,
                 num_landmark=543):
        super().__init__()
        seq_len, num_landmark, num_dim = input_shape
        self.classifier_units = classifier_units
        self.num_class = num_class
        self.num_landmark = num_landmark
        self.batch_norm = nn.BatchNorm2d(num_dim)
        self.vqvae = VQVAE(num_dim, embedding_dim, num_embeddings, hidden_dims)
        
        latent_dim = (seq_len, num_landmark)
        for _ in hidden_dims:
            latent_dim = [dim // 2 for dim in latent_dim]
        self.latent_dim = (embedding_dim, *latent_dim)
        print("Latent dim =", self.latent_dim)
        
        self.classifier = self.get_classifier()
        self.accuracy = MulticlassAccuracy(num_classes=num_class)
        
    
    def get_classifier(self):
        classifier = []
        input_dim = self.latent_dim[0]
        for units in self.classifier_units:
            output_dim = units
            classifier.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding="same"))
            classifier.append(nn.LeakyReLU())
            input_dim = output_dim
        
        
        classifier.append(nn.Flatten())
        classifier.append(nn.Linear(output_dim * self.latent_dim[1] * self.latent_dim[2], self.num_class))
        classifier.append(nn.Softmax(dim=1))
        return  nn.Sequential(*classifier)

    def encode(self, x):
        x = self.batch_norm(x)
        x = self.vqvae.encode(x)[0]
        quantized_inputs, vq_loss = self.vqvae.vq_layer(x)
        return quantized_inputs, vq_loss
    
    def decode(self, x):
        x = self.vqvae.decode(x)
        return x
        
    def forward(self, x):
        x = x[:,:,LANDMARK_IDX,:]
        x = torch.nan_to_num(x)
        
        b, l, f, c = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        z, loss = self.encode(x)
        x_hat = self.decode(z)
        
        pred = self.classifier(z)
        return [x, x_hat, pred, loss]
    
    def training_step(self, batch, batch_idx):
        input, target = batch
        x, x_hat, y_pred, vq_loss = self(input)
        

        classification_loss = F.cross_entropy(y_pred, target)
        recons_loss = F.mse_loss(x_hat, x) * 100
        vq_loss *= 100
        
        loss = classification_loss + recons_loss + vq_loss
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        
        self.log(
            "train/classification_loss",
            classification_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/recons_loss",
            recons_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/vq_loss",
            vq_loss,
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
        x, x_hat, y_pred, vq_loss = self(input)
        

        classification_loss = F.cross_entropy(y_pred, target)
        recons_loss = F.mse_loss(x_hat, x) * 100
        vq_loss *= 100
        
        loss = classification_loss + recons_loss + vq_loss
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/classification_loss",
            classification_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/recons_loss",
            recons_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/vq_loss",
            vq_loss,
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