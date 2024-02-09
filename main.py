import torch
import hydra
import lightning as L
from model import RoadModel
from dataset import RoadDataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    train_ds = RoadDataset(cfg.ds.path, "train")
    val_ds = RoadDataset(cfg.ds.path, "test")
    test_ds = RoadDataset(cfg.ds.path, "test")

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=11)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=11)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadModel(cfg, device)

    wandb_logger = WandbLogger(project="road-segmentation", name="baseline", log_model='all')

    trainer = L.Trainer(max_epochs=5, accelerator="gpu", devices=1, logger=wandb_logger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.validate(model, dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
