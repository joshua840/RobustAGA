from .model_Resnet import ResNet18
from .model_Resnet_imagenet import resnet18 as resnet18_imagenet
from .model_Lenet import LeNet
from ..utils.convert_activation import convert_relu_to_softplus_lrp
import torchvision


def load_model(model, activation_fn, softplus_beta):
    if model == "lenet":
        net = LeNet()
    elif model == "resnet18":
        net = ResNet18()
    elif model == "resnet18_imagenet100":
        net = resnet18_imagenet(num_classes=100)
    else:
        raise NameError(f"{model} is a wrong model")

    if activation_fn == "softplus":
        convert_relu_to_softplus_lrp(net, softplus_beta)

    return net
