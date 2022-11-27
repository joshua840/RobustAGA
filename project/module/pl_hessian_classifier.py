import torch
import torch.nn.functional as F

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from module import LitClassifier


class LitHessianClassifier(LitClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def default_step(self, x, y, stage):
        with torch.enable_grad():
            x.requires_grad = True

            y_hat = self(x)
            acc = self.metric.get_accuracy(y_hat, y)

            # cross entropy loss
            ce_loss = F.cross_entropy(y_hat, y, reduction="mean")

            # hessian regularization
            criterion_fn = lambda output, y: output[range(len(y)), y].sum()
            v = torch.randn_like(x)
            g = criterion_fn(y_hat, y)

            grad1 = torch.autograd.grad(outputs=g, inputs=x, create_graph=True, retain_graph=True)[0]
            dot_vg_vec = torch.einsum("nchw,nchw->n", v, grad1)
            grad2 = torch.autograd.grad(outputs=dot_vg_vec.sum(), inputs=x, create_graph=True, retain_graph=True)[0]
            fn_sq = torch.einsum("nchw,nchw->n", grad2, grad2)
            reg_loss = fn_sq.mean()

            loss = ce_loss + self.hparams.lamb * reg_loss

        self.log_dict(
            {f"{stage}_loss": loss, f"{stage}_ce_loss": ce_loss, f"{stage}_reg_loss": reg_loss, f"{stage}_acc": acc,},
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LitClassifier.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Hessian classifier")
        group.add_argument("--lamb", type=float, default=1.0)
        return parser
