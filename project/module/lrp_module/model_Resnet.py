"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn.functional as F
from .batchnorm import BatchNorm2d
from .convolution import Conv2d
from .flatten import Flatten
from .linear import Linear
from .pool import AdaptiveAvgPool2d, MaxPool2d
from .activation import ReLU
from .sequential import Sequential
from .utils import stabilize


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.expansion * planes),
            )
        self.act1 = ReLU()
        self.act2 = ReLU()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        self.output = out
        self.identity = self.shortcut(x)

        out = self.output + self.identity
        out = self.act2(out)
        return out

    def lrp(self, R, lrp_mode="epsilon"):
        R = self.act2.lrp(R, lrp_mode)

        if lrp_mode in ["epsilon"]:
            Rout, Rid = self._epsilon_lrp(R, eps=1e-8)
        elif lrp_mode in ["epsilon_gamma_box", "epsilon_plus", "epsilon_plus_box"]:
            Rout, Rid = self._alpha_beta_lrp(R, alpha=1, beta=0)
        elif lrp_mode in ["epsilon_alpha2_beta1"]:
            Rout, Rid = self._alpha_beta_lrp(R, alpha=2, beta=1)
        elif lrp_mode in ["deconvnet", "guided_backprop"]:
            Rout, Rid = self._gradient(R)
        else:
            raise NameError(f"{lrp_mode} is not a valid lrp name")

        module_list = [self.bn2, self.conv2, self.act1, self.bn1, self.conv1]

        Rid = self.shortcut.lrp(Rid, lrp_mode)

        for module in module_list:
            Rout = module.lrp(Rout, lrp_mode)

        return Rout + Rid

    def _epsilon_lrp(self, R, eps):
        zs = stabilize(self.output + self.identity, eps)
        Rout = self.output / zs * R
        Rid = self.identity / zs * R

        return Rout, Rid

    def _alpha_beta_lrp(self, R, alpha, beta):
        out_p = F.relu(self.output)
        out_n = self.output - out_p

        id_p = F.relu(self.identity)
        id_n = self.identity - id_p

        Rout_p = (out_p / (out_p + id_p + 1e-8)) * R
        Rid_p = (id_p / (out_p + id_p + 1e-8)) * R

        Rout_n = (out_n / (out_p + id_n + 1e-8)) * R
        Rid_n = (id_n / (out_n + id_n + 1e-8)) * R

        Rout = alpha * Rout_p + (beta * Rout_n)
        Rid = alpha * Rid_p + (beta * Rid_n)

        return Rout, Rid

    def _gradient(self, R):
        return R, R


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(self.expansion * planes)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.expansion * planes),
            )
        self.activation = ReLU()

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = Linear(512 * block.expansion, num_classes)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.act = ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def lrp(self, R, lrp_mode="epsilon"):
        module_list = [
            self.linear,
            self.flatten,
            self.avgpool,
            self.layer4,
            self.layer3,
            self.layer2,
            self.layer1,
            self.act,
            self.bn1,
            self.conv1,
        ]

        for module in module_list:
            R = module.lrp(R, lrp_mode)

        return R


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()
