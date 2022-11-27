import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

from module import (
    LitClassifier,
    LitHessianClassifier,
    LitL2PlusCosdClassifier,
)
import neptune.new as neptune
from module.utils.data_module import CIFAR10DataModule, ImageNet100DataModule, FlowersDataModule



def cli_main():
    # ------------ args -------------
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=1234, type=int, help="random seeds")
    parser.add_argument("--regularizer", default="none", type=str, help="A regularizer to be used")
    parser.add_argument("--loggername", default="default", type=str, help="a name of logger to be used")
    parser.add_argument("--project", default="default", type=str, help="a name of project to be used")
    parser.add_argument("--dataset", default="cifar10", type=str, help="dataset to be loaded")

    temp_args, _ = parser.parse_known_args()
    if temp_args.regularizer == "none":
        Classifier = LitClassifier
    elif temp_args.regularizer == "hessian":
        Classifier = LitHessianClassifier
    elif temp_args.regularizer == "l2_cosd":
        Classifier = LitL2PlusCosdClassifier
    else:
        raise Exception("regularizer name error")

    if temp_args.dataset == "cifar10":
        Dataset = CIFAR10DataModule
    elif temp_args.dataset == "imagenet100":
        Dataset = ImageNet100DataModule
    else:
        raise Exception("dataset name error")

    parser = Classifier.add_model_specific_args(parser)
    parser = Dataset.add_data_specific_args(parser)

    _, _ = parser.parse_known_args()  # This command blocks the help message of Trainer class.
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # ------------ data -------------
    data_module = Dataset(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        num_workers=args.num_workers,
    )
    # ------------ logger -------------
    if args.loggername == "tensorboard":
        logger = True  # tensor board is a default logger of Trainer class
        dirpath = args.default_root_dir
    elif args.loggername == "neptune":
        API_KEY = os.environ.get("NEPTUNE_API_TOKEN")
        ID = os.environ.get("NEPTUNE_ID")
        run = neptune.init(
            api_token=API_KEY, project=f"{ID}/{args.default_root_dir.split('/')[-1]}", capture_stdout=False
        )
        logger = NeptuneLogger(run=run, log_model_checkpoints=False)
        dirpath = os.path.join(args.default_root_dir, logger.version)
    elif args.loggername == "wandb":
        logger = WandbLogger(project=args.project)
        dirpath = args.default_root_dir
    else:
        raise Exception("Wrong logger name.")

    # ------------ callbacks -------------
    if args.loggername == "wandb":
        checkpoint_callback = ModelCheckpoint(monitor="valid_acc", filename="max", save_last=True, mode="max",)
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_acc",
            filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
            save_last=True,
            mode="max",
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ------------ trainer -------------
    trainer = pl.Trainer.from_argparse_args(
        args,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        track_grad_norm=2,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    # ------------ model -------------
    model = Classifier(**vars(args))

    # ------------ run -------------
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, dataloaders=data_module)


if __name__ == "__main__":
    cli_main()
