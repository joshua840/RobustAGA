import torch
import torch.nn.functional as F

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from module import LitClassifier
from .utils.parser import str2bool


class LitL2PlusCosdClassifier(LitClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        if self.hparams.dataset == "cifar10":
            self.std = torch.tensor([0.2023, 0.1994, 0.2010])[None, :, None, None].cuda()
        else:
            self.std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].cuda()

    def default_step(self, x, y, stage):
        criterion_fn = lambda output, y: output[range(len(y)), y].sum()
        with torch.enable_grad():
            x.requires_grad = True

            y_hat = self(x)
            yc_hat = criterion_fn(y_hat, y)
            grad = torch.autograd.grad(outputs=yc_hat, inputs=x, create_graph=True, retain_graph=True)[0]

            acc = self.metric.get_accuracy(y_hat, y)
            ce_loss = F.cross_entropy(y_hat, y, reduction="mean")

            eps_into_norm = self.hparams.eps / (255 * self.std)

            if self.hparams.perturb == "gaussian":
                z = eps_into_norm * torch.randn_like(x)
            elif self.hparams.perturb == "uniform":
                z = eps_into_norm * (2 * torch.rand_like(x) - 1)

            if self.hparams.perturb in ["grad", "grad_sign"]:
                z = grad.detach() if self.hparams.perturb == "grad" else torch.sign(grad).detach()
                z = eps_into_norm * z / (z.flatten(1).norm(dim=1)[:, None, None, None] + 1e-8)

            if self.hparams.detach_source_grad:
                grad = grad.detach()

            x_r = (x + z).clone().detach().requires_grad_()
            y_hat_r = self(x_r)

            yc_hat_r = criterion_fn(y_hat_r, y)

            grad_r = torch.autograd.grad(outputs=yc_hat_r, inputs=x_r, create_graph=True, retain_graph=True)[0]

            cossim = self.metric.calc_cossim(grad, grad_r).mean()
            cosd = (1 - cossim) / 2

            l2 = (grad - grad_r).flatten(start_dim=1).norm(dim=1).square().mean()

            reg_loss = self.hparams.lamb_l2 * l2 + self.hparams.lamb_cos * cosd
            loss = ce_loss + reg_loss

        self.log_dict(
            {f"{stage}_loss": loss, f"{stage}_ce_loss": ce_loss, f"{stage}_acc": acc, f"{stage}_reg_loss": reg_loss},
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LitClassifier.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Cossim classifier")
        group.add_argument("--eps", type=float, default=4)
        group.add_argument("--lamb_l2", type=float, default=1.0)
        group.add_argument("--lamb_cos", type=float, default=1.0)
        group.add_argument("--perturb", type=str, default="uniform", help="gaussian, uniform, grad")
        group.add_argument("--detach_source_grad", type=str2bool, default="True")

        return parser

