import torch.nn as nn
from ..lrp_module.activation import Softplus


def convert_relu_to_softplus(model, beta):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Softplus(beta=beta))
        else:
            convert_relu_to_softplus(child, beta)


def convert_relu_to_softplus_lrp(model, beta):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Softplus(beta=beta))
        else:
            convert_relu_to_softplus_lrp(child, beta)


def convert_softplus_to_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Softplus):
            setattr(model, child_name, nn.ReLU())
        else:
            convert_softplus_to_relu(child)
