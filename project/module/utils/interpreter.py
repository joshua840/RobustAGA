import torch


class Interpreter:
    def __init__(self, model):
        self.model = model

    def get_heatmap(self, x, y, y_hat, method, normalization="absmax", threshold=False, trainable=False, hparams=None):
        assert method in [
            "grad",
            "xgrad",
            "intgrad",
            "smoothgrad",
            "epsilon_gamma_box",
            "epsilon_plus",
            "epsilon_alpha2_beta1",
            "epsilon_plus_box",
            "deconvnet",
            "guided_backprop",
            "gradcam",
        ], "not in our interpretation method list"

        assert method not in ["gradcam"], "gradcam is under maintenance"

        if x.requires_grad == False:
            x.requires_grad = True

        if y_hat is None and method not in ["intgrad", "smoothgrad", "gradcam"]:
            y_hat = self.model(x)

        if method == "grad":
            y_hat_c = y_hat[range(len(y)), y]
            h = self.simple_grad(x, y_hat_c, trainable=trainable)

        elif method == "xgrad":
            y_hat_c = y_hat[range(len(y)), y]
            h = self.simple_grad(x, y_hat_c, trainable=trainable) * x

        elif method == "intgrad":
            del y_hat
            h = self.int_grad(x, y, num_iter=50, trainable=trainable)

        elif method == "smoothgrad":
            del y_hat
            h = self.smooth_grad(x, y, num_iter=hparams.int_sg_iter, alpha=hparams.int_sg_alpha, trainable=trainable,)

        elif method == "grad_cam":
            if y_hat is None:
                self._register_forward_hook_grad_cam()
                y_hat = self.model(x)

            y_hat_c = y_hat[range(len(y)), y]
            h = self.grad_cam(x, y, y_hat_c, trainable=trainable)
            self._remove_forward_hook_grad_cam()

        else:
            eye = torch.eye(y_hat.shape[1], device=y_hat.device)
            h = self.model.lrp(R=eye[y], lrp_mode=method)

        if threshold == "thres":
            h = torch.nn.functional.threshold(h, threshold=0, value=0)
        elif threshold == "abs":
            h = h.abs()

        # reduction
        if len(h.shape) == 4:
            h = h.sum(dim=1)

        # normalization
        if normalization == "standard":
            h_max = h.max(dim=2, keepdims=True)[0].max(dim=1, keepdims=True)[0]
            h = h / (h_max + 1e-8)
        elif normalization == "absmax":
            h_max = h.abs().max(dim=2, keepdims=True)[0].max(dim=1, keepdims=True)[0]
            h = h / (h_max + 1e-8)
        elif normalization == "sum":
            h = h / (h.sum(dim=(1, 2), keepdims=True) + 1e-8)

        return h

    def simple_grad(self, x, y_hat_c, trainable=False):
        h = torch.autograd.grad(
            y_hat_c, x, grad_outputs=torch.ones_like(y_hat_c), create_graph=trainable, retain_graph=trainable,
        )[0]
        return h

    def int_grad(self, x, y, num_iter=50, trainable=False):
        x_b = torch.zeros_like(x)

        h = 0
        for i in range(num_iter + 1):
            alpha = float(i) / num_iter
            x_in = (1 - alpha) * x_b + alpha * x
            y_hat_c = self.model(x_in)[range(len(y)), y]

            if trainable:
                h += self.simple_grad(x_in, y_hat_c, trainable=trainable)
            else:
                h += self.simple_grad(x_in, y_hat_c, trainable=trainable).detach()
        return h / num_iter

    def smooth_grad(self, x, y, num_iter=50, alpha=0.1, trainable=False):
        sigma = (x.max() - x.min()) * alpha

        h = 0
        for i in range(num_iter):
            x_noise = x + sigma * torch.randn_like(x)
            y_hat_c = self.model(x_noise)[range(len(y)), y]

            if trainable:
                h += self.simple_grad(x_noise, y_hat_c, trainable=trainable)
            else:
                h += self.simple_grad(x_noise, y_hat_c, trainable=trainable).detach()
        return h / num_iter

    def grad_cam(self, x, y, y_hat_c, create_graph=False):
        with torch.enable_grad():
            activation_map = self._target_layer_output[0]
            target_layer_grad = torch.autograd.grad(
                y_hat_c, activation_map, torch.ones_like(y_hat_c), create_graph=create_graph
            )[0]
            weight = torch.nn.functional.adaptive_avg_pool2d(target_layer_grad, 1)

            h = torch.mul(activation_map, weight).sum(dim=1, keepdim=True)
            h = torch.nn.functional.threshold(h, threshold=0, value=0)
        return h

    def _register_forward_hook_grad_cam(self, model_name):
        self._target_layer_output = []

        def my_forward_hook(model, input, output):
            self._target_layer_output.append(output)
            # automatically remove the handler after the first call of hook function.
            self._hook_handler.remove()

        target_layer_dict = {
            "vgg16": ["features", 28],
            "vgg19": ["features", 34],
            "resnet18": ["layer4", 1],
            "resnet50": ["layer4", 1],
        }

        layer, sub_layer = target_layer_dict[model_name]
        self._hook_handler = self.model.__dict__["_modules"][layer][sub_layer].register_forward_hook(my_forward_hook)

    def _remove_forward_hook_grad_cam(self):
        del self._target_layer_output
