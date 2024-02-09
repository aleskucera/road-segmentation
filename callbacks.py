from typing import Any

import wandb
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT


class LogPredictionsCallback(Callback):
    def on_validation_batch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule",
                                outputs: STEP_OUTPUT, batch: Any, batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        if batch_idx == 0:
            image, label = batch
            logits = pl_module(image)
            prediction = logits.argmax(1).float().cpu().detach().numpy()

            predictions = wandb.Image(
                prediction,
                masks={
                    "predictions": {
                        "mask_data": label.float().cpu().detach().numpy(),
                        "class_labels": ["background", "road"],
                    }
                },
            )
            trainer.logger.experiment.log({"predictions": predictions})
