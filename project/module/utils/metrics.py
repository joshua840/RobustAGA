import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from math import exp


class Metrics:
    @staticmethod
    def get_binary_accuracy(logit, y, threshold=0):
        y_hat = (logit >= threshold).long().squeeze()
        y = y.squeeze()
        return (y_hat == y).long().sum().item() * 1.0 / len(y_hat)

    @staticmethod
    def get_accuracy(y_hat, y):
        return (y_hat.argmax(dim=1) == y).sum().item() * 1.0 / len(y)


    @staticmethod
    def vectorized_frob_norm(H, reduction="none"):
        norm = torch.norm(H, dim=(-1, -2))
        if reduction == "sum":
            return norm.sum()
        elif reduction == "mean":
            return norm.mean()
        return norm

    @staticmethod
    def vectorized_top_eigenvalue(H, reduction="none"):
        eigvals = torch.linalg.eigvalsh(H)[:, -1]
        if reduction == "sum":
            return eigvals.sum()
        elif reduction == "mean":
            return eigvals.mean()
        return eigvals

    @staticmethod
    def weight_norm(model, reduction="mean"):
        cnt = 0
        sum_norm = 0
        for key, value in model.named_parameters():
            if "weight" in key:
                sum_norm += torch.norm(value.data).item()
                cnt += 1

        return sum_norm / cnt

    @staticmethod
    def estimated_frob_norm_hessian(model, x, y, criterion="ce_loss", n_itr=100, mean=True):
        if criterion == "ce_loss":
            criterion_fn = torch.nn.CrossEntropyLoss()
        elif criterion == "probit":

            def criterion_fn(output, y):
                return torch.nn.functional.softmax(output)[range(len(y)), y].sum()

        elif criterion == "logit":

            def criterion_fn(output, y):
                return output[range(len(y)), y].sum()

        model.zero_grad()
        with torch.enable_grad():
            expec = 0
            x.requires_grad = True
            for itr in range(n_itr):
                x.grad = None
                v = torch.randn_like(x)
                g = criterion_fn(model(x), y)

                grad1 = torch.autograd.grad(outputs=g, inputs=x, create_graph=True, retain_graph=True)[0]
                dot_vg_vec = torch.einsum("nchw,nchw->n", v, grad1)
                grad2 = torch.autograd.grad(outputs=dot_vg_vec.sum(), inputs=x, create_graph=False)[0]
                expec += torch.einsum("nchw,nchw->n", grad2, grad2) / n_itr
        if mean == True:
            return expec.sqrt().mean()
        else:
            return expec.sqrt()

    @staticmethod
    def estimated_frob_norm_hessian_and_grad(model, x, y, criterion="ce_loss", n_itr=100):
        if criterion == "ce_loss":
            criterion_fn = torch.nn.CrossEntropyLoss()
        elif criterion == "probit":

            def criterion_fn(output, y):
                return torch.nn.functional.softmax(output)[range(len(y)), y].sum()

        elif criterion == "logit":

            def criterion_fn(output, y):
                return output[range(len(y)), y].sum()

        with torch.enable_grad():
            expec = 0
            x.requires_grad = True
            for itr in range(n_itr):
                x.grad = None
                v = torch.randn_like(x)
                g = criterion_fn(model(x), y)

                grad1 = torch.autograd.grad(outputs=g, inputs=x, create_graph=True, retain_graph=True)[0]
                dot_vg_vec = torch.einsum("nchw,nchw->n", v, grad1)
                grad2 = torch.autograd.grad(outputs=dot_vg_vec.sum(), inputs=x, create_graph=False)[0]
                expec += torch.einsum("nchw,nchw->n", grad2, grad2) / n_itr

        H = expec.sqrt()
        g = grad1.reshape(x.shape[0], -1).norm(dim=1)
        H_div_g = H / g
        return H.mean(), g.mean(), H_div_g.mean()

    @staticmethod
    def vectorized_hvp_norm_square(model, x, v, y, criterion="ce_loss"):
        """
        A norm of the Hessian vector product for each instances.
        v: random perturbation
        """
        if criterion == "ce_loss":
            criterion_fn = torch.nn.CrossEntropyLoss()
        elif criterion == "probit":

            def criterion_fn(output, y):
                return torch.nn.functional.softmax(output)[range(len(y)), y].sum()

        elif criterion == "logit":

            def criterion_fn(output, y):
                return output[range(len(y)), y].sum()

        with torch.enable_grad():
            x.requires_grad = True
            g = criterion_fn(model(x), y)

            grad1 = torch.autograd.grad(outputs=g, inputs=x, create_graph=True, retain_graph=True)[0]
            dot_vg_vec = torch.einsum("nchw,nchw->n", v, grad1)
            grad2 = torch.autograd.grad(outputs=dot_vg_vec.sum(), inputs=x, create_graph=False)[0]
            hvp_norm = torch.einsum("nchw,nchw->n", grad2, grad2)

        return hvp_norm

    @staticmethod
    def calc_pcc(x, y):
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        mu_x = x.mean(dim=1, keepdim=True)
        mu_y = y.mean(dim=1, keepdim=True)
        std_x = x.std(dim=1)
        std_y = y.std(dim=1)

        pcc = ((x - mu_x) * (y - mu_y)).mean(dim=1) / (std_x * std_y)
        return pcc.mean()


    @staticmethod
    def calc_ssim(tensor1, tensor2):
        """
        tensor1: nchw tensor
        tensor2: nchw tensor 
        """
        if len(tensor1.shape) == 3:
            tensor1 = tensor1.unsqueeze(1)
        if len(tensor2.shape) == 3:
            tensor2 = tensor2.unsqueeze(1)

        return ssim(tensor1, tensor2)

    @staticmethod
    def calc_cossim(tensor1, tensor2):

        tensor1 = tensor1.flatten(start_dim=1)
        tensor2 = tensor2.flatten(start_dim=1)

        return F.cosine_similarity(tensor1, tensor2)


def torch_to_numpy(tensor):
    if len(tensor.shape) == 4:
        numpy_array = tensor.permute(0, 2, 3, 1).contiguous().squeeze().detach().cpu().numpy()
    else:
        numpy_array = tensor.contiguous().squeeze().detach().cpu().numpy()

    return numpy_array


ranking_loss = nn.SoftMarginLoss()
cos = nn.CosineSimilarity(dim=1, eps=1e-15)


def hard_example_mining(dist_mat_p, dist_mat_n, labels, return_inds=False):
    N = dist_mat_p.size(0)
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())

    dist_ap, relative_p_inds = torch.max(dist_mat_p[is_pos].contiguous().view(N, -1), 1, keepdim=True)

    dist_an, relative_n_inds = torch.min(dist_mat_n[is_pos].contiguous().view(N, -1), 1, keepdim=True)

    dist_ap = dist_ap.squeeze(1)

    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


def cosine_dist(x, g1, g2):
    dot_p = x @ g1.t()
    norm1 = torch.norm(x, 2, 1) + 1e-8

    norm2 = torch.norm(g1, 2, 1) + 1e-8

    dot_p = torch.div(dot_p, norm1.unsqueeze(1))

    dot_p = torch.div(dot_p, norm2)

    dot_n = x @ g2.t()
    norm3 = torch.norm(g2, 2, 1) + 1e-8
    dot_n = torch.div(dot_n, norm1.unsqueeze(1))
    dot_n = torch.div(dot_n, norm3)

    return 1.0 - dot_p, 1.0 - dot_n


def exemplar_loss_fn(x, g1, g2, y, a):

    dist_mat_p, dist_mat_n = cosine_dist(x, g1, g2)
    dist_ap, dist_an = hard_example_mining(dist_mat_p, dist_mat_n, a, return_inds=False)
    y = dist_an.new().resize_as_(dist_an).fill_(1)
    loss = ranking_loss(dist_an - dist_ap, y)
    return loss


def get_gradient_with_score(model, y, num_classes, images):
    images.requires_grad = True
    logits = model(images)
    top_scores = logits.gather(1, index=y.unsqueeze(1))
    non_target_indices = np.array([[k for k in range(num_classes) if k != y[j]] for j in range(y.size(0))])

    bottom_scores = logits.gather(1, index=torch.tensor(non_target_indices).cuda())
    bottom_scores = bottom_scores.max(dim=1)[0]

    g1 = torch.autograd.grad(top_scores.mean(), images, retain_graph=True, create_graph=True)[0]
    g2 = torch.autograd.grad(bottom_scores.mean(), images, retain_graph=True, create_graph=True)[0]
    return g1, g2


def topk_alignment_saliency_attack(
    model, images, y, min_value, max_value, eps, num_classes, k_top, num_steps, step_size
):
    def exemplar_loss_fn(x, g1, g2, y):
        dist_ap = 1.0 - cos(x, g1)
        dist_an = 1.0 - cos(x, g2)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        loss = ranking_loss(dist_an - dist_ap, y)
        return loss

    batch_size = images.size(0)
    adv = images.clone()
    with torch.no_grad():
        adv = adv + 2 * eps * (torch.rand_like(images) - 0.5)
        adv = torch.clamp(adv, min_value, max_value)
    adv.requires_grad = True
    g1, _ = get_gradient_with_score(model, y, num_classes, adv)
    g1_abs = torch.abs(g1.reshape(batch_size, -1))
    _, top_idx_g1 = torch.topk(g1_abs, k_top, 1)

    for i in range(num_steps):
        adv.requires_grad = True
        g1_adp, g2_adp = get_gradient_with_score(model, y, num_classes, adv)

        top_g1 = g1_adp.reshape(batch_size, -1).gather(1, index=top_idx_g1)
        top_g2 = g2_adp.reshape(batch_size, -1).gather(1, index=top_idx_g1)
        top_adv = adv.reshape(batch_size, -1).gather(1, index=top_idx_g1)

        exemplar_loss = exemplar_loss_fn(top_adv, top_g1, top_g2, y)
        topK_direction = torch.autograd.grad(exemplar_loss, adv)[0]

        with torch.no_grad():
            adv = adv + step_size * torch.sign(topK_direction)
            adv = torch.minimum(adv, images + eps)
            adv = torch.maximum(adv, images - eps)
            adv = torch.clamp(adv, min_value, max_value)

    return adv


def hm_plot(h):
    h = h / h.max()
    h = (h + 1) / 2
    plt.imshow(h.cpu(), vmin=0, vmax=1, cmap="seismic")
    plt.axis("off")


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * (img1.max() - img1.min())) ** 2
    C2 = (0.03 * (img1.max() - img1.min())) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=7, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
