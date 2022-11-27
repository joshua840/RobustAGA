from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from . import LitClassifier
from module.lrp_module.load_model import load_model
from module.utils.interpreter import Interpreter
import torch.nn.functional as F


class LitClassifierRandPerturbSimilarityTester(LitClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1).cuda()
        if self.hparams.hm_method in [
            "epsilon_gamma_box",
            "epsilon_plus",
            "epsilon_alpha2_beta1",
            "deconvnet",
            "guided_backprop",
            "epsilon_plus_box",
            "epsilon",
            "epsilon_plus_flat",
            "epsilon_alpha2_beta1_flat",
        ]:
            del self.model, self.interpreter
            self.model = load_model(self.hparams.model, self.hparams.activation_fn, self.hparams.softplus_beta)
        self.interpreter = Interpreter(self.model)

    def test_step(self, batch, batch_idx):
        with torch.enable_grad():
            x_s, y_s = batch
            x_s = x_s.requires_grad_()
            yhat_s = self(x_s)
            h_s = self.interpreter.get_heatmap(
                x_s, y_s, yhat_s, self.hparams.hm_method, self.hparams.hm_norm, self.hparams.hm_thres, False,
            ).detach()

            yhat_s = yhat_s.detach()
            eps_into_norm = self.hparams.rand_eps / (255 * self.std)

            for i in range(self.hparams.rand_num_iter):
                # Do a random perturbation on input
                z = eps_into_norm * (2 * torch.rand_like(x_s) - 1)
                x_r = (x_s + z).clone().detach().requires_grad_()
                x_r.data = torch.clamp(x_r, x_s.min(), x_s.max())

                yhat_r = self(x_r)
                h_r = self.interpreter.get_heatmap(
                    x_r, y_s, yhat_r, self.hparams.hm_method, self.hparams.hm_norm, self.hparams.hm_thres, False,
                )

                prefix = f"{self.hparams.hm_method}_{self.hparams.hm_norm}_{self.hparams.hm_thres}"
                name = f"{prefix}_scale_{self.hparams.rand_eps}(h_r,h_s)"
                self.log_hm_metrics(h_r, h_s, name)

                h_r = h_r / (h_r.abs().sum(dim=(1, 2), keepdim=True) + 1e-8)
                h_s = h_s / (h_s.abs().sum(dim=(1, 2), keepdim=True) + 1e-8)

                name = f"{prefix}_sumnorm_scale_{self.hparams.rand_eps}(h_r,h_s)"
                self.log_hm_metrics(h_r, h_s, name)

        return

    def log_hm_metrics(self, h1, h2, name):
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
        group.add_argument("--hm_method", type=str, default="grad", help="interpretation method")
        group.add_argument("--hm_norm", type=str, default="standard")
        group.add_argument("--hm_thres", type=str, default="abs")
        group.add_argument("--rand_eps", type=float, default=4)

        return parser
