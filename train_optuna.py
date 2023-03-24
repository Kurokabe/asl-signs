import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
from src.data.datamodule import SignDataModule
from src.model.tcn_optuna import TCNClassifier
from pytorch_lightning.loggers import TensorBoardLogger


def objective(trial):
    print("#" * 80)
    print("Trial: ", trial.number)
    print("#" * 80)
    logger = TensorBoardLogger("lightning_logs", name="optuna", version=trial.number)

    trainer = Trainer(
        logger=logger,
        max_epochs=50,
        devices=[trial.number % 8],
        accelerator="gpu"
        # callbacks=[PyTorchLightningPruningCallback(trial, monitor="val/accuracy")],
    )

    model = TCNClassifier(input_shape=264, max_sequence_length=48, trial=trial)
    datamodule = SignDataModule(
        max_sequence_length=48,
        normalize=trial.suggest_categorical("normalize", [True, False]),
        substract=trial.suggest_categorical("substract", [True, False]),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        num_workers=8,
    )

    trainer.fit(model, datamodule=datamodule)
    return trainer.logged_metrics["val/accuracy_epoch"]
    # return logger.metrics[-1]["val/accuracy"]


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, n_jobs=8)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
