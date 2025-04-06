from typing import Generator

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models


class LitResNet(pl.LightningModule):
    def __init__(self, label_smoothing: float = 0.0, lr: float = 1e-2, channels_last: bool = False):
        super().__init__()
        self.save_hyperparameters()

        self.model = models.resnet18(pretrained=False, num_classes=10)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, x):
        x = x.to(memory_format=torch.contiguous_format) if self.hparams.channels_last else x
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def summary(self):
        return (
            f'Model Config:\n'
            f'M  - Learning Rate: {self.lr}\n'
            f'M  - Label Smoothing: {self.label_smoothing}\n'
            f'M  - Channels Last: {self.channels_last}'
        )


class LitResNetBuilder:
    trick: str = ''
    value: str = ''

    def __iter__(self) -> Generator[LitResNet, None, None]:
        for smoothing in [0.0, 0.05, 0.1, 0.2]:
            self.trick, self.value = 'LabelSmoothing', str(smoothing)
            yield LitResNet(label_smoothing=smoothing)

        for cl in [False, True]:
            self.trick, self.value = 'ChannelsLast', str(cl)
            lit = LitResNet(channels_last=cl)
            if cl:
                lit.model = lit.model.to(memory_format=torch.channels_last)
            yield lit

        for mode in ['off', 'default', 'reduce-overhead', 'max-autotune']:
            self.trick, self.value = 'TorchCompile', str(mode)
            lit = LitResNet()
            if mode != 'off':
                lit.model = torch.compile(lit.model, mode=mode)
            yield lit
