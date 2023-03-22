import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from data import SignDataModule
from vqvae import SignModel

# from torchinfo import summary


def cli_main():
    torch.set_float32_matmul_precision("medium")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val/loss")
    cli = LightningCLI(
        SignModel,
        SignDataModule,
        trainer_defaults={"callbacks": [checkpoint_callback]},
    )

    # summary(cli.model, input_size=(2, 3, 128, 256))


if __name__ == "__main__":
    cli_main()