import torch
import hydra
import pytorch_lightning as L
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from src import RoadDataModule, RoadModel, LogPredictionsCallback, val_checkpoint, regular_checkpoint


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadModel(cfg, device)
    datamodule = RoadDataModule(cfg)

    wandb_logger = WandbLogger(project="road-segmentation", name=cfg.run_name, config=cfg)

    trainer = L.Trainer(max_epochs=cfg.train.max_epochs,
                        accelerator="gpu",
                        devices=1,
                        logger=wandb_logger,
                        callbacks=[
                            LogPredictionsCallback(),
                            val_checkpoint,
                            regular_checkpoint
                        ])

    if cfg.action == "train":
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    elif cfg.action == "test":
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    else:
        raise ValueError(f"Unknown action: {cfg.action}")


if __name__ == "__main__":
    main()
