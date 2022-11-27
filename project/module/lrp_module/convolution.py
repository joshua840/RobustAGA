# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from .utils import stabilize


class Conv2d(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        first_layer=False,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
        )

        self.first_layer = first_layer

    def forward(self, input):
        self.input = input
        self.output = self._conv_forward(input=input, weight=self.weight, bias=self.bias)
        self.output_shape = self.output.shape
        return self.output

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def lrp(self, R, lrp_mode):
        if lrp_mode == "epsilon":
            return self._epsilon_lrp(R, eps=1e-8)
        elif lrp_mode == "epsilon_gamma_box":
            if self.first_layer:
                return self._zbox_lrp(R)
            return self._gamma_lrp(R, gamma=0.25, eps=1e-8)
        elif lrp_mode == "epsilon_plus_box":
            if self.first_layer:
                return self._zbox_lrp(R)
            return self._alpha_beta_lrp(R, alpha=1, beta=0, eps=1e-8)
        elif lrp_mode == "epsilon_plus":
            return self._alpha_beta_lrp(R, alpha=1, beta=0, eps=1e-8)
        elif lrp_mode == "epsilon_alpha2_beta1":
            return self._alpha_beta_lrp(R, alpha=2, beta=1, eps=1e-8)
        elif lrp_mode in ["deconvnet", "guided_backprop"]:
            return self._gradient(R)
        raise NameError(f"{lrp_mode} is not a valid lrp name")

    def _epsilon_lrp(self, R, eps):
        zs = stabilize(self.output, eps)
        return (
            self.input * torch.autograd.grad(self.output, self.input, R / zs, retain_graph=True, create_graph=True)[0]
        )

    def _gamma_lrp(self, R, gamma=0.25, eps=1e-8):
        weight = self.weight + gamma * self.weight.clamp(min=0)
        bias = self.bias + gamma * self.bias.clamp(min=0) if self.bias is not None else None
        output = self._conv_forward(input=self.input, weight=weight, bias=bias)
        zs = stabilize(output)
        return self.input * torch.autograd.grad(zs, self.input, R / zs, retain_graph=True, create_graph=True)[0]

    def _alpha_beta_lrp(self, R, alpha, beta, eps=1e-8):
        input_p = self.input.clamp(min=0)
        input_n = self.input.clamp(max=0)
        weight_p = self.weight.clamp(min=0)
        weight_n = self.weight.clamp(max=0)
        if self.bias is not None:
            bias_p = self.bias.clamp(min=0)
            bias_n = self.bias.clamp(max=0)
        else:
            bias_p, bias_n = None, None

        def f(x1, x2, w1, w2, b1, b2):
            z1 = self._conv_forward(x1, w1, b1)
            z2 = self._conv_forward(x2, w2, b2)
            zs = stabilize(z1 + z2, eps)

            tmp1 = x1 * torch.autograd.grad(z1, x1, R / zs, retain_graph=True, create_graph=True)[0]
            tmp2 = x2 * torch.autograd.grad(z2, x2, R / zs, retain_graph=True, create_graph=True)[0]

            return tmp1 + tmp2

        R_alpha = f(input_p, input_n, weight_p, weight_n, bias_p, None)

        if beta != 0:
            R_beta = f(input_p, input_n, weight_n, weight_p, bias_n, None)
            Rx = alpha * R_alpha - beta * R_beta
        else:
            Rx = R_alpha

        return Rx

    def _zbox_lrp(self, R):
        if R.shape[2] == 224:  # imagenet
            mean = torch.tensor((0.485, 0.456, 0.406), device=self.input.device)
            std = torch.tensor((0.229, 0.224, 0.225), device=self.input.device)
        elif R.shape[2] == 32:  # CIFAR10
            mean = torch.tensor((0.4914, 0.4822, 0.4465), device=self.input.device)
            std = torch.tensor((0.2023, 0.1994, 0.2010), device=self.input.device)

        x_i = self.input
        x_l = ((0 - mean) / std).reshape(1, 3, 1, 1).expand(x_i.shape).requires_grad_()
        x_h = ((1 - mean) / std).reshape(1, 3, 1, 1).expand(x_i.shape).requires_grad_()

        w = self.weight
        b = self.bias
        z_i = self._conv_forward(x_i, w, b)
        z_l = self._conv_forward(x_l, w.clamp(min=0), b.clamp(min=0))
        z_h = self._conv_forward(x_h, w.clamp(max=0), b.clamp(max=0))

        zs = stabilize(z_i - z_l - z_h)

        R_i = x_i * torch.autograd.grad(z_i, x_i, R / zs, retain_graph=True, create_graph=True)[0]
        R_l = x_l * torch.autograd.grad(z_l, x_l, R / zs, retain_graph=True, create_graph=True)[0]
        R_h = x_h * torch.autograd.grad(z_h, x_h, R / zs, retain_graph=True, create_graph=True)[0]

        return R_i - R_l - R_h

    def _gradient(self, R):
        return torch.autograd.grad(self.output, self.input, R, retain_graph=True, create_graph=True)[0]

