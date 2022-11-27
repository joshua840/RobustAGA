import torchvision
from .resnet import ResNet18
from .lenet import LeNet
from ..utils.convert_activation import convert_relu_to_softplus


def load_model(model, activation_fn, softplus_beta):
    if model == "lenet":
        net = LeNet()
    elif model == "resnet18":
        net = ResNet18()
    elif model == "resnet18_imagenet100":
        net = torchvision.models.resnet18(num_classes=100)
    else:
        raise NameError(f"{model} is a wrong model")

    if activation_fn == "softplus":
        convert_relu_to_softplus(net, softplus_beta)

    return net
