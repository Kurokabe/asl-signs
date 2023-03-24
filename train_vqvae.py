import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from src.data.datamodule import SignDataModule
from src.model.vqvae import SignModel
from pytorch_lightning.loggers import TensorBoardLogger

# from torchinfo import summary


def cli_main():
    torch.set_float32_matmul_precision("medium")
    logger = TensorBoardLogger("lightning_logs", name="vqvae")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val/vqvae_loss")
    cli = LightningCLI(
        SignModel,
        SignDataModule,
        trainer_defaults={"callbacks": [checkpoint_callback], "logger": logger},
    )

    # summary(cli.model, input_size=(2, 3, 128, 256))


if __name__ == "__main__":
    cli_main()
