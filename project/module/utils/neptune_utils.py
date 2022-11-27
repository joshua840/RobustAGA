import os
import torch


def load_ckpt(exp_id, root_dir):
    path = os.path.join(root_dir, exp_id, "last.ckpt")
    return torch.load(path, map_location="cpu")


def wandb_load_ckpt(exp_id, root_dir):
    path = os.path.join(root_dir, exp_id, "checkpoints/last.ckpt")
    return torch.load(path, map_location="cpu")


def set_prev_args(ckpt, args):
    for k, v in ckpt["hyper_parameters"].items():
        if k in ["data_dir", "exp_id", "default_root_dir", "batch_size_test"]:
            continue
        setattr(args, k, v)
    return args


def safe_model_loader(model, ckpt):
    try:
        model.load_state_dict(ckpt["state_dict"])
    except:
        try:
            ckpt["state_dict"]["model.fc2.weight"] = ckpt["state_dict"]["model.fc.weight"]
            ckpt["state_dict"]["model.fc2.bias"] = ckpt["state_dict"]["model.fc.bias"]
            del ckpt["state_dict"]["model.fc.weight"], ckpt["state_dict"]["model.fc.bias"]
            model.load_state_dict(ckpt["state_dict"])

        except:
            for key in [i for i in ckpt["state_dict"].keys() if "model_f" in i]:
                del ckpt["state_dict"][key]
            model.load_state_dict(ckpt["state_dict"])

    return

