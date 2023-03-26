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


class BetaVAE(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        max_sequence_length: int,
        num_features: int,
        hidden_dims: List = None,
        beta: int = 4,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "B",
        **kwargs
    ) -> None:
        super(BetaVAE, self).__init__()

        self.max_sequence_length = max_sequence_length
        self.num_features = num_features

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        self.hidden_dims = hidden_dims

        self.seq_length = max_sequence_length
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim
            self.seq_length = self.seq_length // 2
            self.num_features = self.num_features // 2

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(h_dim * self.seq_length * self.num_features, latent_dim)
        self.fc_var = nn.Linear(h_dim * self.seq_length * self.num_features, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(
            latent_dim, h_dim * self.seq_length * self.num_features
        )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)

        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)

        result = result.view(z.shape[0], -1, self.seq_length, self.num_features)

        result = self.decoder(result)

        result = self.final_layer(result)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)

        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        if self.loss_type == "H":  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == "B":  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0]
            )
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError("Undefined loss type.")

        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

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
        latent_dim: int,
        max_sequence_length: int,
        before_classifier_dim: int = 1024,
        hidden_dims: List = [64, 128, 256],
        classifier_units: List = [128, 256],
        num_class=250,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.max_sequence_length = max_sequence_length
        self.num_landmark = input_shape // num_dim
        self.num_dim = num_dim
        self.latent_dim = latent_dim
        self.classifier_units = classifier_units
        self.num_class = num_class
        self.beta_vae = BetaVAE(
            num_dim,
            latent_dim,
            max_sequence_length=max_sequence_length,
            num_features=self.num_landmark,
            hidden_dims=hidden_dims,
        )

        # latent_dim = (self.max_sequence_length, self.num_landmark)
        # for _ in hidden_dims:
        #     latent_dim = [dim // 2 for dim in latent_dim]
        # self.latent_dim = (embedding_dim, *latent_dim)

        self.before_classifier_dim = before_classifier_dim
        self.classifier_mu = nn.Linear(self.latent_dim, self.before_classifier_dim)
        self.classifier_var = nn.Linear(self.latent_dim, self.before_classifier_dim)
        self.classifier = self.get_classifier()
        self.accuracy = MulticlassAccuracy(num_classes=num_class)

    def get_classifier(self):
        classifier = []
        input_dim = self.before_classifier_dim * 2
        for units in self.classifier_units:
            output_dim = units
            classifier.append(nn.Linear(input_dim, output_dim))
            classifier.append(nn.BatchNorm1d(output_dim))
            classifier.append(nn.LeakyReLU())
            classifier.append(nn.Dropout(0.4))
            input_dim = output_dim

        classifier.append(nn.Flatten())
        classifier.append(nn.Linear(output_dim, self.num_class))
        return nn.Sequential(*classifier)

    def forward(self, x):
        x = x.reshape(
            x.shape[0], self.max_sequence_length, self.num_landmark, self.num_dim
        )
        x = x.permute(0, 3, 1, 2)

        x_hat, input, mu, log_var = self.beta_vae(x)
        loss = self.beta_vae.loss_function(x_hat, input, mu, log_var, M_N=0.005)
        mu = self.classifier_mu(mu)
        log_var = self.classifier_var(log_var)
        mu_var = torch.cat([mu, log_var], dim=1)

        logits = self.classifier(mu_var)
        return [x, x_hat, logits, loss]

    def on_train_epoch_start(self) -> None:
        self.train_y_pred = []
        self.train_y_true = []

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        input, target = batch
        x, x_hat, logits, loss = self(input)

        if optimizer_idx == 0:
            self.log(
                "train/rec_loss",
                loss["Reconstruction_Loss"],
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

            self.log(
                "train/kld_loss",
                loss["KLD"],
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/loss",
                loss["loss"],
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            return loss["loss"]

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
        # self.log_confusion_matrix(y_pred, target, "train/confusion_matrix")

    def on_validation_epoch_start(self) -> None:
        self.val_y_true = []
        self.val_y_pred = []

    def validation_step(self, batch, batch_idx):
        input, target = batch
        x, x_hat, logits, loss = self(input)
        self.log(
            "val/rec_loss",
            loss["Reconstruction_Loss"],
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "val/kld_loss",
            loss["KLD"],
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/loss",
            loss["loss"],
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
        # self.log_confusion_matrix(y_pred, target, "val/confusion_matrix")

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
