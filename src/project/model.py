from typing import Generator

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.classification import Accuracy


class LitResNet(pl.LightningModule):
    def __init__(
        self,
        label_smoothing: float = 0.0,
        lr: float = 1e-2,
        channels_last: bool = False,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = False,
        trick_name: str = 'None',
        trick_value: str = 'None',
    ):
        super().__init__()
        self.save_hyperparameters()

        (
            self.lr,
            self.label_smoothing,
            self.channels_last,
            self.batch_size,
            self.num_workers,
            self.pin_memory,
            self.trick_name,
            self.trick_value,
        ) = (lr, label_smoothing, channels_last, batch_size, num_workers, pin_memory, trick_name, trick_value)
        self.model = models.resnet18(pretrained=False, num_classes=10)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.train_acc = Accuracy(task='multiclass', num_classes=10)
        self.val_acc = Accuracy(task='multiclass', num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = self.train_acc(preds, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = self.val_acc(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def on_fit_start(self):
        self.logger.log_hyperparams(self.hparams)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }

    def summary(self):
        return (
            f'Model Config:\n'
            f'M - Learning Rate: {self.lr}\n'
            f'M - Label Smoothing: {self.label_smoothing}\n'
            f'M - Channels Last: {self.channels_last}'
        )


class LitResNetBuilder:

    def __iter__(self) -> Generator[LitResNet, None, None]:
        for smoothing in [0.0, 0.05, 0.1, 0.2]:
            yield LitResNet(label_smoothing=smoothing, trick_name='LabelSmoothing', trick_value=str(smoothing))

        for cl in [False, True]:
            if cl:
                lit = LitResNet(channels_last=cl, trick_name='ChannelsLast', trick_value=str(cl))
                lit.model = lit.model.to(memory_format=torch.channels_last)
            else:
                lit = LitResNet(channels_last=cl, trick_name='ChannelsLast', trick_value=str(cl))

            yield lit

        for mode in ['off', 'default', 'reduce-overhead', 'max-autotune']:
            lit = LitResNet(trick_name='TorchCompile', trick_value=str(mode))
            if mode != 'off':
                lit.model = torch.compile(lit.model, mode=mode)
            yield lit
