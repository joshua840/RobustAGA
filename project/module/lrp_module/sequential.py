import torch
import torch.nn.functional as F
from .activation import ReLU


class Sequential(torch.nn.Sequential):
    def lrp(self, R, lrp_mode="epsilon"):
        for module in reversed(self):
            R = module.lrp(R, lrp_mode)
        return R
