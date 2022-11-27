from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from . import LitClassifier
from module.lrp_module.load_model import load_model
from module.utils.interpreter import Interpreter
import torch.nn.functional as F

from module.utils.convert_activation import (
    convert_relu_to_softplus,
    convert_softplus_to_relu,
)


class LitClassifierXAIAdvTester(LitClassifier):
    def __init__(
        self, **kwargs,
    ):
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

        self.prepare_target()

    def prepare_target(self):
        k = 5
        l = 32
        if "image" in self.hparams.dataset:
            k = k * 7
            l = l * 7

        mask = torch.ones((1, l, l), dtype=torch.float32)

        mask[:, k:-k, k:-k] = torch.zeros((l - 2 * k, l - 2 * k), dtype=torch.float32)

        self.h_t = mask.cuda()

    def forward(self, x):
        x = self.model(x)
        return x

    def test_step(self, batch, batch_idx):
        x_s, y_s = batch
        with torch.enable_grad():
            x_s = x_s.requires_grad_()

            yhat_s = self(x_s)
            h_s = self.interpreter.get_heatmap(
                x_s, y_s, yhat_s, self.hparams.hm_method, self.hparams.hm_norm, self.hparams.hm_thres, False,
            ).detach()
            yhat_s = yhat_s.detach()

            # get adv imgs
            x_adv = self.get_adv_img(x_s, y_s, yhat_s).detach().requires_grad_()
            yhat_adv = self(x_adv)
            h_adv = self.interpreter.get_heatmap(
                x_adv, y_s, yhat_adv, self.hparams.hm_method, self.hparams.hm_norm, self.hparams.hm_thres, False,
            ).detach()
            yhat_adv.detach()

            h_t_expand = self.h_t.expand(h_adv.shape)

            # metrics
            loss_f = F.mse_loss(yhat_s, yhat_adv, reduction="sum") / h_adv.shape[0]
            acc_adv = self.metric.get_accuracy(yhat_adv, y_s)

            prefix = (
                f"frameadv_method_{self.hparams.hm_method}_eps_{self.hparams.adv_eps}_iter_{self.hparams.adv_num_iter}"
            )

            # log results
            self.log_dict(
                {f"{prefix}_loss_f": loss_f, f"{prefix}_acc_adv": acc_adv}, prog_bar=True, sync_dist=True,
            )

            self.log_hm_metrics(h_adv, h_s, f"{prefix}_(h_a,h_s)")
            self.log_hm_metrics(h_adv, h_t_expand, f"{prefix}_(h_a,h_t)")

    def get_adv_img(self, x, y, yhat):
        if self.hparams.activation_fn == "relu":
            convert_relu_to_softplus(self.model, beta=20.0)

        eps = (self.hparams.adv_eps / 255.0) * 5
        with torch.enable_grad():
            x_adv = x.clone().detach().requires_grad_()
            adv_optimizer = torch.optim.Adam([x_adv], lr=4.0 * eps / self.hparams.adv_num_iter)

            # Do adversarial attack on XAI
            for i in range(self.hparams.adv_num_iter):
                adv_optimizer.zero_grad()

                yhat_adv = self(x_adv)
                h_adv = self.interpreter.get_heatmap(
                    x_adv,
                    y,
                    yhat_adv,
                    self.hparams.hm_method,
                    self.hparams.hm_norm,
                    self.hparams.hm_thres,
                    True,
                    self.hparams,
                )
                h_t_expand = self.h_t.expand(h_adv.shape)

                # calculate loss
                loss_expl = F.mse_loss(h_adv, h_t_expand, reduction="sum") / h_adv.shape[0]
                loss_output = F.mse_loss(yhat_adv, yhat.detach())
                total_loss = self.hparams.adv_gamma * loss_expl + (1 - self.hparams.adv_gamma) * loss_output

                # update adversarial example
                total_loss.backward()
                adv_optimizer.step()

                delta = torch.clamp(x_adv - x, -eps, eps)
                x_adv.data = torch.clamp(x.data + delta.data, x.min(), x.max())

        if self.hparams.activation_fn == "relu":
            convert_softplus_to_relu(self.model)

        return x_adv.detach()

    def log_hm_metrics(self, h1, h2, name):
        loss = F.mse_loss(h1, h2, reduction="sum") / h1.shape[0]
        pcc = self.metric.calc_pcc(h1, h2)
        ssim = self.metric.calc_ssim(h1, h2)
        cossim = self.metric.calc_cossim(h1, h2)

        h1 = h1 / (h1.abs().sum(dim=(1, 2), keepdim=True) + 1e-8)
        h2 = h2 / (h2.abs().sum(dim=(1, 2), keepdim=True) + 1e-8)
        sumnorm_ssim = self.metric.calc_ssim(h1, h2)
        # log results
        self.log_dict(
            {
                f"{name}_mse": loss,
                f"{name}_pcc": pcc,
                f"{name}_ssim": ssim,
                f"{name}_cossim": cossim,
                f"{name}_sumnorm_ssim": sumnorm_ssim,
            },
            prog_bar=True,
            sync_dist=True,
        )
        return

    @staticmethod
    def perturbation(h_s, x_s, ratio, mode="insertion"):
        order = h_s.flatten(1).argsort(descending=True)
        n_perturb = int(ratio * order.shape[1])
        n_order = order[:, n_perturb]
        threshold = h_s.flatten(1)[range(len(h_s)), n_order]
        if mode == "insertion":
            mask = (h_s > threshold.reshape(len(h_s), 1, 1)).unsqueeze(1)
        elif mode == "deletion":
            mask = (h_s < threshold.reshape(len(h_s), 1, 1)).unsqueeze(1)
        else:
            raise NameError("Wrong mode name. It should be one of the [insertion, deletion]")

        return (x_s * mask).detach()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LitClassifier.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Adversarial attack test")
        group.add_argument("--adv_num_iter", type=int, default=100)
        group.add_argument("--adv_eps", type=int, default=4)
        group.add_argument(
            "--adv_gamma", type=float, default=0.5, help="(1-r) loss_feature + r loss_hmt",
        )
        group.add_argument("--hm_method", type=str, default="grad", help="interpretation method")
        group.add_argument("--hm_norm", type=str, default="standard")
        group.add_argument("--hm_thres", type=str, default="abs")
        return parser
