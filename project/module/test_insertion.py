from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from . import LitClassifier
from module.lrp_module.load_model import load_model
from module.utils.interpreter import Interpreter


class LitClassifierAOPCTester(LitClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        self.softmax = torch.nn.Softmax(dim=1)

    def test_step(self, batch, batch_idx):
        with torch.enable_grad():
            x_s, y_s = batch
            x_s = x_s.requires_grad_()

            yhat_s = self(x_s)
            h_s = self.interpreter.get_heatmap(
                x_s, y_s, yhat_s, self.hparams.hm_method, self.hparams.hm_norm, self.hparams.hm_thres, False,
            ).detach()

        for i in range(self.hparams.aopc_iter):
            ratio = float(i) / self.hparams.aopc_iter
            x_p = self.perturbation(h_s, x_s, ratio=ratio)

            logit = self(x_p)
            prob = self.softmax(logit)

            aopc_prob = prob[range(len(y_s)), y_s].detach().mean()

            self.log(
                f"aopc_insertion_method_{self.hparams.hm_method}_{ratio}", aopc_prob, prog_bar=True,
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
        group = parser.add_argument_group("AOPC test arguments")
        group.add_argument("--aopc_iter", type=int, default=20)
        group.add_argument("--hm_method", type=str, default="grad", help="interpretation method")
        group.add_argument("--hm_norm", type=str, default="standard")
        group.add_argument("--hm_thres", type=str, default="abs")
        return parser

