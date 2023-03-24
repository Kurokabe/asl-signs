from typing import Any, Callable, List, Tuple, TypeVar, Union

import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.optim import Adam
from torchmetrics.classification import MulticlassAccuracy
from data.data import label_to_index_mapping
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# from torch import tensor as Tensor

Tensor = TypeVar("torch.tensor")

from abc import abstractmethod

from torch import nn

LANDMARK_IDX = [0, 9, 11, 13, 14, 17, 117, 118, 119, 199, 346, 347, 348] + list(
    range(468, 543)
)
MAX_LENGTH = 64


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(
            0, 2, 3, 1
        ).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = (
            torch.sum(flat_latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        )  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(
            encoding_one_hot, self.embedding.weight
        )  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return (
            quantized_latents.permute(0, 3, 1, 2).contiguous(),
            vq_loss,
        )  # [B x D x H x W]


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class VQVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        num_embeddings: int,
        hidden_dims: List = None,
        beta: float = 0.25,
        img_size: int = 64,
        **kwargs
    ) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU(),
            )
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1
                ),
                nn.LeakyReLU(),
            )
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1], out_channels=3, kernel_size=4, stride=2, padding=1
                ),
                nn.Tanh(),
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "VQ_Loss": vq_loss}

    def sample(
        self, num_samples: int, current_device: Union[int, str], **kwargs
    ) -> Tensor:
        raise Warning("VQVAE sampler is not implemented.")

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class SignModel(pl.LightningModule):
    def __init__(
        self,
        input_shape: int,
        num_dim: int,
        embedding_dim: int,
        num_embeddings: int,
        max_sequence_length: int,
        hidden_dims: List = [128, 256],
        classifier_units: List = [128, 256],
        num_class=250,
    ):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.num_landmark = input_shape // num_dim
        self.num_dim = num_dim
        self.classifier_units = classifier_units
        self.num_class = num_class
        self.vqvae = VQVAE(num_dim, embedding_dim, num_embeddings, hidden_dims)

        latent_dim = (self.max_sequence_length, self.num_landmark)
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
            classifier.append(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, padding="same")
            )
            classifier.append(nn.BatchNorm2d(output_dim))
            classifier.append(nn.LeakyReLU())
            classifier.append(nn.Dropout(0.4))
            input_dim = output_dim

        classifier.append(nn.Flatten())
        classifier.append(
            nn.Linear(
                output_dim * self.latent_dim[1] * self.latent_dim[2], self.num_class
            )
        )
        return nn.Sequential(*classifier)

    def encode(self, x):
        x = self.vqvae.encode(x)[0]
        quantized_inputs, vq_loss = self.vqvae.vq_layer(x)
        return quantized_inputs, vq_loss

    def decode(self, x):
        x = self.vqvae.decode(x)
        return x

    def forward(self, x):
        x = x.reshape(
            x.shape[0], self.max_sequence_length, self.num_landmark, self.num_dim
        )
        x = x.permute(0, 3, 1, 2)
        z, loss = self.encode(x)
        x_hat = self.decode(z)

        logits = self.classifier(z)
        return [x, x_hat, logits, loss]

    def on_train_epoch_start(self) -> None:
        self.train_y_pred = []
        self.train_y_true = []

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        input, target = batch
        x, x_hat, logits, vq_loss = self(input)

        if optimizer_idx == 0:
            recons_loss = F.mse_loss(x_hat, x)

            vqvae_loss = recons_loss + vq_loss
            self.log(
                "train/vqvae_loss",
                vqvae_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

            self.log(
                "train/recons_loss",
                recons_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/vq_loss",
                vq_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            return vqvae_loss

        elif optimizer_idx == 1:
            classification_loss = F.cross_entropy(logits, target)

            self.log(
                "train/classification_loss",
                classification_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            y_pred = F.softmax(logits, dim=1)

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
            self.train_y_true.append(target)
            self.train_y_pred.append(y_pred)

            return classification_loss

    def train_epoch_end(self, outputs):
        target = torch.cat(self.train_y_true)
        y_pred = torch.cat(self.train_y_pred)
        self.log_confusion_matrix(y_pred, target, "train/confusion_matrix")

    def on_validation_epoch_start(self) -> None:
        self.val_y_true = []
        self.val_y_pred = []

    def validation_step(self, batch, batch_idx):
        input, target = batch
        x, x_hat, logits, vq_loss = self(input)

        recons_loss = F.mse_loss(x_hat, x)

        vqvae_loss = recons_loss + vq_loss
        self.log(
            "val/vqvae_loss",
            vqvae_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "val/recons_loss",
            recons_loss,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/vq_loss",
            vq_loss,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        classification_loss = F.cross_entropy(logits, target)

        self.log(
            "val/classification_loss",
            classification_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        y_pred = F.softmax(logits, dim=1)

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

        self.val_y_true.append(target)
        self.val_y_pred.append(y_pred)

        return classification_loss

    def validation_epoch_end(self, outputs):
        target = torch.cat(self.val_y_true)
        y_pred = torch.cat(self.val_y_pred)
        self.log_confusion_matrix(y_pred, target, "val/confusion_matrix")

    def log_confusion_matrix(self, y_pred, target, name):
        y_pred = y_pred.argmax(dim=1).detach().cpu().numpy()
        y_true = target.argmax(dim=1).detach().cpu().numpy()
        labels = label_to_index_mapping.keys()
        cm = confusion_matrix(
            y_true, y_pred, normalize="true", labels=list(range(len(labels)))
        )
        cm_df = pd.DataFrame(
            cm,
            index=labels,
            columns=labels,
        )
        plt.figure(figsize=(100, 100))
        ax = sns.heatmap(
            cm_df,
            annot=True,
        )
        ax.set(xlabel="Predicted label", ylabel="True label")
        fig = ax.get_figure()

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".jpg") as tmp:
            fig.savefig(tmp.name)
            img = imread(tmp.name)
            img = np.transpose(img, (2, 0, 1))
            self.logger.experiment.add_image(name, img, self.current_epoch)
        plt.close()

    def configure_optimizers(self):
        vqvae_optimizer = Adam(self.parameters(), lr=5e-4)
        classifier_optimizer = Adam(self.parameters(), lr=5e-4, weight_decay=1e-5)

        return [vqvae_optimizer, classifier_optimizer], []
