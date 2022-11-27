"""LeNet in PyTorch."""
import torch.nn as nn
import torch.nn.functional as F

from .batchnorm import BatchNorm2d
from .convolution import Conv2d
from .flatten import Flatten
from .linear import Linear
from .pool import AdaptiveAvgPool2d, MaxPool2d
from .activation import ReLU, Softplus
from .sequential import Sequential


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv11 = Conv2d(3, 32, 3, padding=1, first_layer=True)
        self.conv12 = Conv2d(32, 32, 3, padding=1)
        self.conv21 = Conv2d(32, 64, 3, padding=1)
        self.conv22 = Conv2d(64, 64, 3, padding=1)
        self.fc1 = Linear(4096, 256)
        self.fc2 = Linear(256, 10)
        self.act1 = ReLU(inplace=True)
        self.act2 = ReLU(inplace=True)
        self.act3 = ReLU(inplace=True)
        self.act4 = ReLU(inplace=True)
        self.act5 = ReLU(inplace=True)
        self.maxpool1 = MaxPool2d(2)
        self.maxpool2 = MaxPool2d(2)
        self.flatten = Flatten()

    def forward(self, x):
        out = self.act1(self.conv11(x))
        out = self.act2(self.conv12(out))
        out = self.maxpool1(out)
        out = self.act3(self.conv21(out))
        out = self.act4(self.conv22(out))
        out = self.maxpool2(out)
        out = self.flatten(out)
        out = self.act5(self.fc1(out))
        out = self.fc2(out)
        return out

    def lrp(self, R, lrp_mode="epsilon"):
        R = self.fc2.lrp(R, lrp_mode=lrp_mode)
        R = self.fc1.lrp(self.act5.lrp(R, lrp_mode=lrp_mode), lrp_mode=lrp_mode)
        R = self.flatten.lrp(R, lrp_mode=lrp_mode)
        R = self.maxpool2.lrp(R, lrp_mode=lrp_mode)
        R = self.conv22.lrp(self.act4.lrp(R, lrp_mode=lrp_mode), lrp_mode=lrp_mode)
        R = self.conv21.lrp(self.act3.lrp(R, lrp_mode=lrp_mode), lrp_mode=lrp_mode)
        R = self.maxpool1.lrp(R, lrp_mode=lrp_mode)
        R = self.conv12.lrp(self.act2.lrp(R, lrp_mode=lrp_mode), lrp_mode=lrp_mode)
        R = self.conv11.lrp(self.act1.lrp(R, lrp_mode=lrp_mode), lrp_mode=lrp_mode)
        return R

