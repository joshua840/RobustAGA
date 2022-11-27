import os
import pytorch_lightning as pl
from torchvision import transforms
from .dataset import Flower17Dataset
from .parser import str2bool
from .dataset import AtexCIFAR10
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import numpy as np
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/cifar10",
        dataset="cifar10",
        batch_size_train: int = 128,
        batch_size_test: int = 100,
        normalization=True,
        num_workers=4,
        decoy_patch_size=3,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        if normalization:
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            self.transform_train = transforms.Compose(
                [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),]
            )
            self.transform_test = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        # downloadt
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if self.hparams.dataset == "cifar10":
            self.cifar_train = CIFAR10(root=self.hparams.data_dir, train=True, transform=self.transform_train)
            self.cifar_test = CIFAR10(root=self.hparams.data_dir, train=False, transform=self.transform_test)

        else:
            raise Exception("name error")

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.hparams.batch_size_train,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Data arguments")
        group.add_argument("--num_workers", default=4, type=int, help="number of workers")
        group.add_argument("--batch_size_train", default=128, type=int, help="batchsize of data loaders")
        group.add_argument("--batch_size_test", default=100, type=int, help="batchsize of data loaders")
        group.add_argument("--data_dir", default="/data/cifar10", type=str, help="directory of cifar10 dataset")
        group.add_argument("--transform", default=True, type=str2bool, help="whether apply transform or not")
        return parser


class ImageNet100DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/Data/ImageNet100",
        dataset="imagenet100",
        batch_size_train: int = 128,
        batch_size_test: int = 100,
        normalization=True,
        num_workers=4,
        test_shuffle=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.test_shuffle = test_shuffle

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
        )

        val_transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
        )

        self.train_dataset = ImageFolder(train_dir, transform=train_transform)
        self.test_dataset = ImageFolder(val_dir, transform=val_transform)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size_train,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size_test,
            shuffle=self.test_shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Data arguments")
        group.add_argument("--num_workers", default=8, type=int, help="number of workers")
        group.add_argument("--batch_size_train", default=128, type=int, help="batchsize of data loaders")
        group.add_argument("--batch_size_test", default=100, type=int, help="batchsize of data loaders")
        group.add_argument("--data_dir", default="/Data/ImageNet100", type=str, help="directory of imagenet dataset")
        return parser


class FlowersDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "~/data/Flowers17",
        batch_size_train: int = 128,
        batch_size_test: int = 100,
        num_workers=4,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform_train = transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.transform_test = transforms.Compose(
            [transforms.Resize(160), transforms.CenterCrop(128), transforms.ToTensor(), normalize,]
        )

    def prepare_data(self):
        # downloadt
        Flower17Dataset(self.hparams.data_dir, train=True, download=True)

    def setup(self, stage=None):
        self.flowers17_train = Flower17Dataset(root=self.hparams.data_dir, train=True, transform=self.transform_train)
        self.flowers17_test = Flower17Dataset(root=self.hparams.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(
            self.flowers17_train,
            batch_size=self.hparams.batch_size_train,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.flowers17_test,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Data arguments")
        group.add_argument("--num_workers", default=4, type=int, help="number of workers")
        group.add_argument("--batch_size_train", default=128, type=int, help="batchsize of data loaders")
        group.add_argument("--batch_size_test", default=100, type=int, help="batchsize of data loaders")
        group.add_argument("--data_dir", default="/data/Flowers17", type=str, help="directory of Flowers17 dataset")
        return parser


class AtexCIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/cifar10",
        batch_size_train: int = 128,
        batch_size_test: int = 100,
        num_workers=4,
        sg_path="",
        fx_path="",
        transform=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        if transform == True:
            self.transform_train = self._transform_train

            self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            self.transform_train = transforms.Compose([transforms.ToTensor(), normalize,])
            self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    @staticmethod
    def _transform_train(image, sg):
        # Custom random crop with padding
        image = TF.pad(image, 4)
        sg = TF.pad(sg, 4)

        i = torch.randint(0, 8, size=(1,)).item()
        j = torch.randint(0, 8, size=(1,)).item()

        image = TF.crop(image, i, j, 32, 32)
        sg = TF.crop(sg, i, j, 32, 32)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            sg = TF.hflip(sg)

        # To tensor
        image, sg = TF.to_tensor(image), torch.as_tensor(np.array(sg)[:, :])

        # Normalization
        image = TF.normalize(image, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True)

        # Return
        return image, sg

    def prepare_data(self):
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.cifar_train = AtexCIFAR10(
            root=self.hparams.data_dir,
            sg_path=self.hparams.sg_path,
            fx_path=self.hparams.fx_path,
            train=True,
            transform=self.transform_train,
        )
        self.cifar_test = CIFAR10(root=self.hparams.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.hparams.batch_size_train,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Data arguments")
        group.add_argument("--num_workers", default=4, type=int, help="number of workers")
        group.add_argument("--batch_size_train", default=64, type=int, help="batchsize of data loaders")
        group.add_argument("--batch_size_test", default=100, type=int, help="batchsize of data loaders")
        group.add_argument("--data_dir", default="/data/cifar10", type=str, help="directory of cifar10 dataset")
        group.add_argument(
            "--sg_path",
            # default="./output/Cifar10Pretrained/cifar10_lenet_smoothgrad.pt",
            default="./output/Cifar10Pretrained/ROB15-14/cifar10_resnet_smoothgrad.pt",
            type=str,
            help="smooth grad path",
        )
        group.add_argument(
            "--fx_path",
            default="./output/Cifar10Pretrained/ROB15-14/cifar10_resnet_logit.pt",
            type=str,
            help="logit path",
        )
        group.add_argument("--transform", default=False, type=str2bool, help="whether apply transform or not")
        return parser

