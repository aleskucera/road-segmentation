from typing import Any

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class LogPredictionsCallback(Callback):
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        batch = next(iter(trainer.val_dataloaders))
        images, labels = batch
        image, label = images[0].to(pl_module.device), labels[0].to(pl_module.device)

        self._log_prediction(trainer, pl_module, image, label)

    def on_test_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        print(f"Batch {batch_idx}")
        images, labels = batch
        image, label = images[0].to(pl_module.device), labels[0].to(pl_module.device)

        self._log_prediction(trainer, pl_module, image, label)

    @staticmethod
    def _log_prediction(trainer: "pl.Trainer", pl_module: "pl.LightningModule", image: torch.Tensor,
                        label: torch.Tensor = None):
        logits = pl_module(image.unsqueeze(0)).squeeze(0)

        # Apply inverse normalization
        mean = torch.tensor(pl_module.config.ds.mean).view(1, 3, 1, 1).to(pl_module.device)
        std = torch.tensor(pl_module.config.ds.std).view(1, 3, 1, 1).to(pl_module.device)
        image = image * std + mean

        class_labels = {0: "void", 1: "feasible", 2: "infeasible", 3: "other"}

        # Log prediction mask
        prediction_mask = logits.argmax(0).float().cpu().detach().numpy()
        prediction_image = wandb.Image(
            image.permute(1, 2, 0).cpu().detach().numpy(),
            masks={"prediction": {"mask_data": prediction_mask, "class_labels": class_labels}},
        )
        trainer.logger.experiment.log({"prediction": prediction_image})

        if label is not None:
            # Log ground truth mask
            ground_truth_mask = label.float().cpu().detach().numpy()
            ground_truth_image = wandb.Image(
                image.permute(1, 2, 0).cpu().detach().numpy(),
                masks={"ground_truth": {"mask_data": ground_truth_mask, "class_labels": class_labels}},
            )
            trainer.logger.experiment.log({"ground_truth": ground_truth_image})

            # Log error mask
            error_mask = torch.abs(label - logits.argmax(0)).cpu().detach().numpy()
            error_image = wandb.Image(
                image.permute(1, 2, 0).cpu().detach().numpy(),
                masks={"error": {"mask_data": error_mask, "class_labels": class_labels}},
            )
            trainer.logger.experiment.log({"error": error_image})
