from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger
import pytorch_lightning as pl
import os
import neptune.new as neptune
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from module import (
    LitClassifierAOPCTester,
    LitClassifierAdvAOPCTester,
    LitClassifierRandPerturbSimilarityTester,
    LitClassifierXAIAdvTester,
    LitClassifierUpperBoundTester,
)

from module.utils.data_module import CIFAR10DataModule, ImageNet100DataModule, FlowersDataModule
from module.utils.neptune_utils import load_ckpt, set_prev_args, safe_model_loader, wandb_load_ckpt
from argparse import ArgumentParser


def cli_main():
    # ------------
    # args
    # ------------

    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=1234, type=int, help="random seeds")
    parser.add_argument("--exp_id", default="", type=str)
    parser.add_argument("--loggername", default="default", type=str, help="a name of logger to be used")
    parser.add_argument("--project", default="default", type=str, help="a name of project to be used")
    parser.add_argument("--dataset", default="cifar10", type=str, help="dataset to be loaded")
    parser.add_argument("--test_method", type=str, help="test method")

    temp_args, _ = parser.parse_known_args()
    if temp_args.dataset == "cifar10":
        Dataset = CIFAR10DataModule
    elif temp_args.dataset == "imagenet100":
        Dataset = ImageNet100DataModule
    elif temp_args.dataset == "flowers17":
        Dataset = FlowersDataModule
    else:
        raise Exception("dataset name error")

    if temp_args.test_method == "aopc":
        Classifier = LitClassifierAOPCTester
    elif temp_args.test_method == "adv":
        Classifier = LitClassifierXAIAdvTester
    elif temp_args.test_method == "adv_aopc":
        Classifier = LitClassifierAdvAOPCTester
    elif temp_args.test_method == "rps":
        Classifier = LitClassifierRandPerturbSimilarityTester
    elif temp_args.test_method == "upper_bound":
        Classifier = LitClassifierUpperBoundTester
    else:
        raise Exception("test_method name error")

    parser = Classifier.add_model_specific_args(parser)
    parser = Dataset.add_data_specific_args(parser)

    _, _ = parser.parse_known_args()  # This command blocks the help message of Trainer class.
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    if args.loggername == "wandb":
        ckpt = wandb_load_ckpt(args.exp_id, args.default_root_dir)
    else:
        ckpt = load_ckpt(args.exp_id, args.default_root_dir)

    args = set_prev_args(ckpt, args)

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
    elif args.loggername == "wandb":
        logger = WandbLogger(id=args.exp_id, project=args.project)
        dirpath = args.default_root_dir
    else:
        raise Exception("Wrong logger name.")

    # ------------ trainer -------------
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, accelerator="gpu", deterministic=True)
    # trainer = pl.Trainer.from_argparse_args(args, accelerator="gpu", logger=logger)

    model = Classifier(**vars(args))
    safe_model_loader(model, ckpt)

    # test
    trainer.test(model, dataloaders=data_module)


if __name__ == "__main__":
    cli_main()
