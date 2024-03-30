import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torchmetrics.classification import Accuracy, JaccardIndex


class RoadModel(pl.LightningModule):
    def __init__(self, config: DictConfig, device: torch.device):
        super().__init__()

        self.config = config

        self.model = hydra.utils.instantiate(config.model)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.std = torch.tensor(config.ds.std).view(1, 3, 1, 1).to(device)
        self.mean = torch.tensor(config.ds.mean).view(1, 3, 1, 1).to(device)

        self.accuracy = Accuracy(task="multiclass", num_classes=config.model.num_classes)
        self.jaccard = JaccardIndex(task="multiclass", num_classes=config.model.num_classes)

        self.test_step_outputs = []
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, image):
        logits = self.model(image)['out']
        return logits

    def shared_step(self, batch, stage):
        image, label = batch
        logits = self.forward(image)
        loss = self.criterion(logits, label)
        jaccard = self.jaccard(logits, label)
        accuracy = self.accuracy(logits, label)

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_jaccard", jaccard, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "accuracy": accuracy, "jaccard": jaccard}

    def training_step(self, batch, batch_idx):
        self.training_step_outputs.append(self.shared_step(batch, "train"))
        return self.training_step_outputs[-1]

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs.append(self.shared_step(batch, "val"))
        return self.validation_step_outputs[-1]

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(self.shared_step(batch, "test"))
        return self.test_step_outputs[-1]

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.config.optimizer, self.model.parameters())
