import torch
from torch.nn import functional as F
from torch.nn import init


class BatchNorm2d(torch.nn.BatchNorm2d):
    def lrp(self, R, lrp_mode=""):
        return R

