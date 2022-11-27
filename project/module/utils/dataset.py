from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple
import torch

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from tqdm import tqdm
import requests
import tarfile
from torchvision.datasets import CIFAR10


class Flower17Dataset(VisionDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(Flower17Dataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download(root)

        self.data = []
        self.targets = []

        for cls in range(10):
            if self.train:
                for no in range(1, 73):
                    self.data.append(os.path.join(root, "jpg", "image_" + str(80 * cls + no).zfill(4) + ".jpg"))
                    self.targets.append(cls)
            else:
                for no in range(73, 81):
                    self.data.append(os.path.join(root, "jpg", "image_" + str(80 * cls + no).zfill(4) + ".jpg"))
                    self.targets.append(cls)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_file, target = self.data[index], self.targets[index]
        img = Image.open(img_file).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def download(self, root):
        if os.path.isdir(root):
            return

        os.makedirs(root, exist_ok=True)
        url_img = "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"
        response = requests.get(url_img, stream=True)

        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

        file_path = os.path.join(root, "17flowers.tgz")
        with open(file_path, "wb") as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)

        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

        with tarfile.open(file_path) as f:
            f.extractall(root)


class AtexCIFAR10(CIFAR10):
    def __init__(
        self,
        root: str,
        sg_path: str,
        fx_path: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.sg = torch.load(sg_path, map_location="cpu")
        self.fx = torch.load(fx_path, map_location="cpu")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target, sg, fx = self.data[index], self.targets[index], self.sg[index], self.fx[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img, sg = self.transform(img, sg)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, sg, fx

    def __len__(self) -> int:
        return len(self.data)
