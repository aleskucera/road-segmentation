import torch
import hydra
import pytorch_lightning as L
from model import RoadModel
from src import RoadDataModule
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadModel(cfg, device)
    datamodule = RoadDataModule(cfg)

    wandb_logger = WandbLogger(project="road-segmentation", name="baseline", log_model='all')

    trainer = L.Trainer(max_epochs=5, accelerator="gpu", devices=1, logger=wandb_logger)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
