import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from module.models.load_model import load_model
from module.utils.metrics import Metrics
from module.utils.interpreter import Interpreter


class LitClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = load_model(self.hparams.model, self.hparams.activation_fn, self.hparams.softplus_beta)

        self.metric = Metrics()
        self.interpreter = Interpreter(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

    def default_step(self, x, y, stage):
        y_hat = self(x)
        acc = self.metric.get_accuracy(y_hat, y)

        loss = F.cross_entropy(y_hat, y, reduction="mean")
        self.log_dict(
            {f"{stage}_loss": loss, f"{stage}_acc": acc,}, prog_bar=True, sync_dist=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch

        loss = self.default_step(x, y, stage="train")

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        loss = self.default_step(x, y, stage="valid")

    def test_step(self, batch, batch_idx):
        x, y = batch

        loss = self.default_step(x, y, stage="test")

    def configure_optimizers(self):
        if self.hparams.optimizer == "sgd":
            optim = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.hparams.weight_decay,
            )
            # nesterov 추가 추후 수정
        elif self.hparams.optimizer == "adam":
            optim = torch.optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "adamw":
            optim = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
            )


        sche = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=self.hparams.milestones, gamma=0.1)
        scheduler = {
            "scheduler": sche,
            "name": "lr_history",
        }

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Default classifier")
        group.add_argument("--model", type=str, default="none", help="which model to be used")
        group.add_argument("--activation_fn", type=str, default="softplus", help="activation function of model")
        group.add_argument("--softplus_beta", type=float, default=3.0, help="beta of softplus")
        group.add_argument("--optimizer", type=str, default="adamw")
        group.add_argument("--weight_decay", type=float, default=4e-5)
        group.add_argument("--learning_rate", type=float, default=1e-3)
        group.add_argument("--milestones", nargs="+", default=[150, 225], type=int, help="lr scheduler")
        return parser
