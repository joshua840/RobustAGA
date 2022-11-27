from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from module import LitClassifier
import torch.nn.functional as F


class LitClassifierUpperBoundTester(LitClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1).cuda()

    def test_step(self, batch, batch_idx):
        with torch.enable_grad():
            x_s, y_s = batch
            x_s = x_s.requires_grad_()
            yhat_s = self(x_s)

            yhat_s_c = yhat_s[range(len(y_s)), y_s]
            g_s = torch.autograd.grad(yhat_s_c, x_s, torch.ones_like(yhat_s_c))[0].detach()
            del yhat_s_c

            yhat_s = yhat_s.detach()
            acc = self.metric.get_accuracy(yhat_s, y_s)
            self.log("test_acc", acc)

            H_norm_square = 0

            for i in range(self.hparams.rand_num_iter):
                # Do a random perturbation on input
                r = (1 / self.std) * (self.hparams.rand_eps / 255) * torch.randn_like(x_s)
                x_r = (x_s + r).clone().detach().requires_grad_()
                x_r.data = torch.clamp(x_r, x_s.min(), x_s.max())

                yhat_r = self(x_r)

                yhat_r_c = yhat_r[range(len(y_s)), y_s]
                g_r = torch.autograd.grad(yhat_r_c, x_r, torch.ones_like(yhat_r_c))[0].detach()
                del yhat_r_c

                Hvp_norm_square = self.metric.vectorized_hvp_norm_square(
                    self, x_s, r, y_s, criterion="logit"
                )  # |Hv|_2^2
                Hvp_norm = Hvp_norm_square.sqrt()
                H_norm_square += Hvp_norm_square.detach()  # expectation of Hvp_norm square is equal to H_norm square

                # Pan upper bounds
                l2_distance = F.mse_loss(g_s, g_r, reduction="sum") / g_s.shape[0]

                # Our upper bounds
                cossim = self.metric.calc_cossim(g_s, g_r)
                cosd = (1 - cossim) / 2

                Dx_norm = g_r.flatten(start_dim=1).norm(dim=-1).detach()

                self.log_dict(
                    {
                        f"scale_{self.hparams.rand_eps}_l2": l2_distance,
                        f"scale_{self.hparams.rand_eps}_cosd": cosd,
                        f"scale_{self.hparams.rand_eps}_|Dfr|": Dx_norm,
                        f"scale_{self.hparams.rand_eps}_|Hfx_r|": Hvp_norm,
                        f"scale_{self.hparams.rand_eps}_|Hfx_r|_|Dfr|": Hvp_norm / (Dx_norm + 1e-8),
                    },
                    prog_bar=True,
                    sync_dist=True,
                )

                name = f"grad_scale_{self.hparams.rand_eps}(h_r,h_s)"
                self.log_hm_metrics(g_r, g_s, name, normalization="max")

                name = f"grad_sumnorm_scale_{self.hparams.rand_eps}(h_r,h_s)"
                self.log_hm_metrics(g_r, g_s, name, normalization="sum")

                acc = self.metric.get_accuracy(yhat_r, y_s)
                self.log_dict(
                    {f"scale_{self.hparams.rand_eps}_acc": acc,}, prog_bar=True, sync_dist=True,
                )

            H_norm_square = H_norm_square / self.hparams.rand_eps
            H_norm = H_norm_square.sqrt().mean()
            self.log("|Hfx|", H_norm)
        return

    def log_hm_metrics(self, h1, h2, name, normalization):
        h1 = h1.abs().sum(dim=1)
        h2 = h2.abs().sum(dim=1)

        if normalization == "sum":
            h1 = h1 / (h1.sum(dim=(1, 2), keepdim=True) + 1e-8)
            h2 = h2 / (h2.sum(dim=(1, 2), keepdim=True) + 1e-8)
        elif normalization == "max":
            h_max1 = h1.max(dim=2, keepdims=True)[0].max(dim=1, keepdims=True)[0]
            h1 = h1 / (h_max1 + 1e-8)
            h_max2 = h2.max(dim=2, keepdims=True)[0].max(dim=1, keepdims=True)[0]
            h2 = h2 / (h_max2 + 1e-8)

        mse = F.mse_loss(h1, h2, reduction="sum") / h1.shape[0]
        pcc = self.metric.calc_pcc(h1, h2)
        ssim = self.metric.calc_ssim(h1, h2)
        cossim = self.metric.calc_cossim(h1, h2)

        self.log_dict(
            {f"{name}_mse": mse, f"{name}_pcc": pcc, f"{name}_ssim": ssim, f"{name}_cossim": cossim},
            prog_bar=True,
            sync_dist=True,
        )
        return

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LitClassifier.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Random perturbation test arguments")
        group.add_argument("--rand_num_iter", type=int, default=10)
        group.add_argument("--rand_eps", type=float, default=4)
        return parser
