"""LeNet in PyTorch."""
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv11 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv12 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv22 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 10)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv11(x))
        out = self.act(self.conv12(out))
        out = F.max_pool2d(out, 2)
        out = self.act(self.conv21(out))
        out = self.act(self.conv22(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.act(self.fc1(out))
        out = self.fc2(out)
        return out
