import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = False,
        trick_name: str = 'None',
        trick_value: str = 'None',
    ):

        super().__init__()
        super().save_hyperparameters()

        self.valset = None
        self.trainset = None
        self.batch_size, self.num_workers, self.pin_memory, self.trick_name, self.trick_value = (
            batch_size,
            num_workers,
            pin_memory,
            trick_name,
            trick_value,
        )

    def setup(self, stage=None):
        self.trainset = datasets.CIFAR10(
            root='./running/data',
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
        )
        self.valset = datasets.CIFAR10(
            root='./running/data',
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def summary(self):
        return (
            f'DataModule Config:\n'
            f'D - Batch Size: {self.batch_size}\n'
            f'D - Num Workers: {self.num_workers}\n'
            f'D - Pin Memory: {self.pin_memory}'
        )


class DataModuleBuilder:
    def __iter__(self):
        for batch_size in [32, 64, 128, 256]:
            yield CIFAR10DataModule(batch_size=batch_size, trick_name='Batch Size', trick_value=str(batch_size))

        for num_worker in [2, 4, 8, 12]:
            yield CIFAR10DataModule(num_workers=num_worker, trick_name='Num Workers', trick_value=str(num_worker))

        for pin in [False, True]:
            yield CIFAR10DataModule(pin_memory=pin, trick_name='Pin Memory', trick_value=str(pin))
