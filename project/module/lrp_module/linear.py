# we referenced the zennit package.
# https://github.com/chr5tphr/zennit
import torch
from torch.nn import functional as F
from .utils import stabilize


class Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, input):
        self.input = input
        self.output = F.linear(self.input, self.weight, self.bias)
        return self.output

    def lrp(self, R, lrp_mode):
        if lrp_mode in [
            "epsilon",
            "epsilon_gamma_box",
            "epsilon_plus",
            "epsilon_alpha2_beta1",
        ]:
            return self._epsilon_lrp(R, 1e-8)
        elif lrp_mode in ["epsilon_plus_box"]:
            return self._alpha_beta_lrp(R, alpha=1, beta=0, eps=1e-8)
        elif lrp_mode in ["deconvnet", "guided_backprop"]:
            return self._gradient(R)
        raise NameError(f"{lrp_mode} is not a valid lrp name")

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
            z1 = F.linear(x1, w1, b1)
            z2 = F.linear(x2, w2, b2)
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

    def _epsilon_lrp(self, R, eps=1e-8):
        zs = stabilize(self.output, eps)
        return (
            self.input * torch.autograd.grad(self.output, self.input, R / zs, retain_graph=True, create_graph=True)[0]
        )

    def _gradient(self, R):
        return torch.autograd.grad(self.output, self.input, R, retain_graph=True, create_graph=True)[0]
