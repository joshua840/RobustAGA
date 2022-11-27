import torch
from torch.nn import functional as F
from .utils import stabilize


class MaxPool2d(torch.nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def forward(self, input):
        self.input = input
        self.output, _ = F.max_pool2d(
            input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, True
        )
        return self.output

    def lrp(self, R, lrp_mode):
        return self._gradient(R)

    def _gradient(self, R):
        return torch.autograd.grad(self.output, self.input, R, retain_graph=True, create_graph=True)[0]


class AvgPool2d(torch.nn.AvgPool2d):
    def __init__(
        self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )

        self.beta = 0

    def forward(self, input):
        self.input = input
        self.output = F.avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )
        return self.output

    def lrp(self, R, lrp_mode):
        if lrp_mode in [
            "epsilon",
            "epsilon_gamma_box",
            "epsilon_plus",
            "epsilon_alpha2_beta1",
            "epsilon_plus_box",
        ]:
            return self._epsilon_lrp(R, 1e-8)
        elif lrp_mode in ["deconvnet", "guided_backprop"]:
            return self._gradient(R)
        raise NameError(f"{lrp_mode} is not a valid lrp name")

    def _epsilon_lrp(self, R, eps):
        zs = stabilize(self.output, eps)
        return (
            self.input * torch.autograd.grad(self.output, self.input, R / zs, retain_graph=True, create_graph=True)[0]
        )

    def _gradient(self, R):
        return torch.autograd.grad(self.output, self.input, R, retain_graph=True, create_graph=True)[0]

    def _alpha_beta_lrp(self, R, alpha, beta):
        x_p = F.relu(self.input)
        # x_n = self.input - x_p

        def f(x):
            zs = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
            zs = stabilize(zs)
            return x * torch.autograd.grad(zs, x, R / zs, retain_graph=True, create_graph=True)[0]

        return f(x_p)
        # return alpha * f(x_p) - beta * f(x_n)


class AdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d):
    def __init__(self, output_size):
        super().__init__(output_size=output_size)
        self.beta = 0

    def forward(self, input):
        self.input = input
        self.output = F.adaptive_avg_pool2d(input, self.output_size)
        return self.output

    def lrp(self, R, lrp_mode):
        if lrp_mode in [
            "epsilon",
            "epsilon_gamma_box",
        ]:
            return self._epsilon_lrp(R, 1e-8)
        elif lrp_mode in ["epsilon_plus", "epsilon_plus_box"]:
            return self._alpha_beta_lrp(R, 1, 0)
        elif lrp_mode in ["epsilon_alpha2_beta1"]:
            return self._alpha_beta_lrp(R, 2, 1)
        elif lrp_mode in ["deconvnet", "guided_backprop"]:
            return self._gradient(R)
        raise NameError(f"{lrp_mode} is not a valid lrp name")

    def _epsilon_lrp(self, R, eps):
        zs = stabilize(self.output, eps)
        return (
            self.input * torch.autograd.grad(self.output, self.input, R / zs, retain_graph=True, create_graph=True)[0]
        )

    def _gradient(self, R):
        return torch.autograd.grad(self.output, self.input, R, retain_graph=True, create_graph=True)[0]

    def _alpha_beta_lrp(self, R, alpha, beta):
        x_p = F.relu(self.input)
        # x_n = self.input - x_p

        def f(x):
            zs = F.adaptive_avg_pool2d(x, self.output_size)
            return x * torch.autograd.grad(zs, x, R / stabilize(zs), retain_graph=True, create_graph=True)[0]

        return f(x_p)
        # return alpha * f(x_p) - beta * f(x_n)
