import torch
import hydra
from model import RoadModel
import pytorch_lightning as L
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from src import RoadDataModule, LogPredictionsCallback, val_checkpoint, regular_checkpoint


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadModel(cfg, device)
    datamodule = RoadDataModule(cfg)

    wandb_logger = WandbLogger(project="road-segmentation", name="baseline", log_model='all')

    trainer = L.Trainer(max_epochs=cfg.train.max_epochs,
                        accelerator="gpu",
                        devices=1,
                        logger=wandb_logger,
                        callbacks=[
                            LogPredictionsCallback(),
                            val_checkpoint,
                            regular_checkpoint
                        ])
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
