import torch
from torch.nn import functional as F


class ReLU(torch.nn.ReLU):
    def __init__(self, inplace=False):
        super().__init__(inplace=inplace)

    def forward(self, input):
        self.input = input
        self.output = F.relu(input, inplace=self.inplace)
        return self.output

    def lrp(self, R, lrp_mode):
        if lrp_mode in [
            "epsilon",
            "epsilon_gamma_box",
            "epsilon_plus",
            "epsilon_alpha2_beta1",
            "epsilon_plus_box",
        ]:
            return R
        elif lrp_mode == "deconvnet":
            return self._deconvnet(R)
        elif lrp_mode == "guided_backprop":
            return self._guided_backprop(R)
        raise NameError(f"{lrp_mode} is not a valid lrp name")

    def _deconvnet(self, R):
        return R.clamp(min=0)

    def _guided_backprop(self, R):
        grad_input = torch.autograd.grad(self.output, self.input, R, retain_graph=True, create_graph=True)[0]
        return grad_input * (R > 0)


class Softplus(torch.nn.Softplus):
    def __init__(self, beta=1, threshold=20):
        super().__init__(beta, threshold)

    def forward(self, input):
        self.input = input
        self.output = F.softplus(input, self.beta, self.threshold)
        return self.output

    def lrp(self, R, lrp_mode):
        if lrp_mode in [
            "epsilon",
            "epsilon_gamma_box",
            "epsilon_plus",
            "epsilon_alpha2_beta1",
            "epsilon_plus_box",
        ]:
            return R
        elif lrp_mode == "deconvnet":
            return self._deconvnet(R)
        elif lrp_mode == "guided_backprop":
            return self._guided_backprop(R)

        raise NameError(f"{lrp_mode} is not a valid lrp name")

    def _deconvnet(self, R):
        return R.clamp(min=0)

    def _guided_backprop(self, R):
        grad_input = torch.autograd.grad(self.output, self.input, R, retain_graph=True, create_graph=True)[0]
        return grad_input * (R > 0)
