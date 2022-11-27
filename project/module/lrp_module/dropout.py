import torch
import torch.nn.functional as F


class Dropout2d(torch.nn.Dropout2d):
    def __init__(self, p=0.5, inplace=False):
        super().__init__(p=p, inplace=inplace)

    def forward(self, input):
        return F.dropout2d(input, self.p, self.training, self.inplace)

    def lrp(self, R, lrp_mode="epsilon"):
        return R
