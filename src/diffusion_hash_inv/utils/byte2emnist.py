"""
Make EMNIST image from parsed Hash calculation log.
"""

from __future__ import annotations

from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from torch.utils.data import ConcatDataset, Subset
import torch

from diffusion_hash_inv.config import ImgConfig
from diffusion_hash_inv.logger import Logs
from diffusion_hash_inv.utils.file_io import FileIO
from diffusion_hash_inv.main.context import RuntimeConfig

class Byte2EMNIST:
    """
    A class to convert byte values(0x00 ~ 0xFF) to EMNIST images.
    """

    def __init__(self, runtime_config: RuntimeConfig, io_controller: FileIO) -> None:
        self.runtime_config = runtime_config
        self.img_cfg: ImgConfig = ImgConfig()
        self.io_controller = io_controller

        # Load EMNIST dataset
        transform = transforms.Compose([
            transforms.Resize(self.img_cfg.img_size),
            transforms.ToTensor(),
        ])
        emnist_train = datasets.EMNIST(
            root=self.io_controller.emnist_dir,
            split='byclass', train=True, download=True, transform=transform)
        emnist_test = datasets.EMNIST(
            root=self.io_controller.emnist_dir,
            split='byclass', train=False, download=True, transform=transform)
        self.emnist_dataset = ConcatDataset([emnist_train, emnist_test])

    def load_emnist(self):
        """
        Load the EMNIST dataset.
        """
