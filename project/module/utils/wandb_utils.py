import os
import torch

def load_ckpt(exp):
    path = os.path.join( exp, "last.ckpt")
    return torch.load(path, map_location="cpu")


def set_prev_args(ckpt, args):
    for k, v in ckpt["hyper_parameters"].items():
        if k == "data_path":
            continue
        if k == "exp_id":
            continue
        if k == "default_root_dir":
            continue
        setattr(args, k, v)
    return args
