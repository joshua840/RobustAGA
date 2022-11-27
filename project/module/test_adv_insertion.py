from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from module import LitClassifierXAIAdvTester


class LitClassifierAdvAOPCTester(LitClassifierXAIAdvTester):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.softmax = torch.nn.Softmax(dim=1)
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

    def test_step(self, batch, batch_idx):
        x_s, y_s = batch
        with torch.enable_grad():
            x_s = x_s.requires_grad_()

            yhat_s = self(x_s).detach()

            # get adv imgs
            x_adv = self.get_adv_img(x_s, y_s, yhat_s).detach().requires_grad_()
            yhat_adv = self(x_adv)
            h_adv = self.interpreter.get_heatmap(
                x_adv, y_s, yhat_adv, self.hparams.hm_method, self.hparams.hm_norm, self.hparams.hm_thres, False,
            ).detach()
            yhat_adv.detach()

        for i in range(self.hparams.aopc_iter):
            ratio = float(i) / self.hparams.aopc_iter
            x_p = self.perturbation(h_adv, x_s, ratio=ratio)

            logit = self(x_p)
            prob = self.softmax(logit)

            aopc_prob = prob[range(len(y_s)), y_s].detach().mean()

            self.log(
                f"aopc_adv_insertion_method_{self.hparams.hm_method}_eps{self.hparams.adv_eps}_{ratio}",
                aopc_prob,
                prog_bar=True,
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LitClassifierXAIAdvTester.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("AOPC test arguments")
        group.add_argument("--aopc_iter", type=int, default=20)
        return parser

