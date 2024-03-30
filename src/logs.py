import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class LogPredictionsCallback(Callback):
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        batch = next(iter(trainer.val_dataloaders))
        images, labels = batch
        image, label = images[0].to(pl_module.device), labels[0].to(pl_module.device)
        logits = pl_module(image.unsqueeze(0)).squeeze(0)

        class_labels = {0: "background", 1: "road"}

        ground_truth_mask = label.float().cpu().detach().numpy()
        prediction_mask = logits.argmax(0).float().cpu().detach().numpy()
        error_mask = torch.abs(label - logits.argmax(0)).cpu().detach().numpy()

        # Log ground truth mask
        ground_truth_image = wandb.Image(
            image.permute(1, 2, 0).cpu().detach().numpy(),
            masks={"ground_truth": {"mask_data": ground_truth_mask, "class_labels": class_labels}},
        )
        trainer.logger.experiment.log({"ground_truth": ground_truth_image})

        # Log prediction mask
        prediction_image = wandb.Image(
            image.permute(1, 2, 0).cpu().detach().numpy(),
            masks={"prediction": {"mask_data": prediction_mask, "class_labels": class_labels}},
        )
        trainer.logger.experiment.log({"prediction": prediction_image})

        # Log error mask
        error_image = wandb.Image(
            image.permute(1, 2, 0).cpu().detach().numpy(),
            masks={"error": {"mask_data": error_mask, "class_labels": class_labels}},
        )
        trainer.logger.experiment.log({"error": error_image})
