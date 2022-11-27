from typing import Tuple, Union
from torch import Tensor
from torch.types import _size
import torch


class Flatten(torch.nn.Flatten):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__(start_dim=start_dim, end_dim=end_dim)

    def forward(self, input):
        self.input = input
        return input.flatten(self.start_dim, self.end_dim)

    def lrp(self, R, lrp_mode="epsilon"):
        return R.reshape(self.input.shape)
