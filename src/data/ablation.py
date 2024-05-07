from typing import Any, Dict, Optional, Tuple, List
from rich import print
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms
import csv
import os
from torchvision.datasets import ImageFolder
import random
import numpy as np

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, decline_rate=None):
        super().__init__(root, transform=transform)
        self.decline_rate = decline_rate

    def __getitem__(self, index):
        # Loop to ensure we return a valid sample
        while True:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.decline_rate is None or np.random.rand() <= self.decline_rate[target]:
                return sample, target
            # Pick a new index if the sample was declined
            index = np.random.randint(0, len(self.samples))

class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        width: int = 256,
        height: int = 256,
        decline_rate = [1.0, 1.0, 0.2, 1.0, 1.0, 1.0],
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((width, height)),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
    
    @property
    def num_classes(self) -> int:
        """Get the number of classes.
        """
        return 6
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup the data.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = CustomImageFolder(f"{self.hparams.data_dir}/train", transform=self.transforms, decline_rate=self.hparams.decline_rate)
            self.data_val = ImageFolder(f"{self.hparams.data_dir}/val", transform=self.transforms)
            self.data_test = ImageFolder(f"{self.hparams.data_dir}/test", transform=self.transforms)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

if __name__ == "__main__":
    datamodule = DataModule()
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    # print(train_dataloader)
    # print(val_dataloader)
    # print(test_dataloader)
    # print(datamodule.num_classes)
    # print(datamodule.batch_size_per_device)
    # print(datamodule.hparams.data_dir)
    # print(datamodule.hparams.batch_size)
    # print(datamodule.hparams.num_workers)
    # print(datamodule.hparams.pin_memory)
    # print(datamodule.transforms)
    # print(datamodule.data_train)
    # print(datamodule.data_val)
    # print(datamodule.data_test)

    for (x, y) in train_dataloader:
        print(x.shape, y.shape)
        # break