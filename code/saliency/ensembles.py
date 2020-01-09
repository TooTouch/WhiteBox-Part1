import numpy as np
import torch
from torch.autograd import Variable
from utils import seed_everything, rescale_image

# normal distribution
def normal_dist(img, m, std):
    return torch.zeros_like(img).normal_(m, std**2).numpy()


def generate_smooth_grad(pre_imgs, targets, n, sigma, model, **kwargs):
    seed_everything()
    if 'layer' not in kwargs.keys():
        kwargs['layer'] = None
    
    # make smooth_grad array
    smooth_grad = np.zeros(pre_imgs.shape[:1] + pre_imgs.shape[2:] + pre_imgs.shape[1:2]) # (batch_size, H, W, C)

    # mean, sigma
    mins = pre_imgs.detach().numpy().min(axis=(1,2,3))
    maxs = pre_imgs.detach().numpy().max(axis=(1,2,3))
    mean = [0] * pre_imgs.size(0)
    sigma = (sigma / (maxs - mins)).squeeze()

    for _ in range(n):
        noise = np.array(list(map(normal_dist, pre_imgs, mean, sigma)))

        noisy_imgs = pre_imgs + torch.Tensor(noise)
        outputs, probs, preds = model.generate_image(noisy_imgs, targets, **kwargs)
        smooth_grad = smooth_grad + outputs

    smooth_grad = smooth_grad / n
    smooth_grad = rescale_image(smooth_grad.transpose(0,3,1,2))

    return smooth_grad, probs, preds

def generate_smooth_square_grad(pre_imgs, targets, n, sigma, model, **kwargs):
    seed_everything()
    if 'layer' not in kwargs.keys():
        kwargs['layer'] = None

    # make smooth_square_grad array
    smooth_square_grad = np.zeros(pre_imgs.shape[:1] + pre_imgs.shape[2:] + pre_imgs.shape[1:2]) # (batch_size, H, W, C)

    mean = 0
    # mean, sigma
    mins = pre_imgs.detach().numpy().min(axis=(1,2,3))
    maxs = pre_imgs.detach().numpy().max(axis=(1,2,3))
    mean = [0] * pre_imgs.size(0)
    sigma = (sigma / (maxs - mins)).squeeze()

    for _ in range(n):
        noise = np.array(list(map(normal_dist, pre_imgs, mean, sigma)))

        noisy_imgs = pre_imgs + torch.Tensor(noise)
        outputs, probs, preds = model.generate_image(noisy_imgs, targets, **kwargs)
        smooth_square_grad = smooth_square_grad + outputs**2

    smooth_square_grad = smooth_square_grad / n
    smooth_square_grad = rescale_image(smooth_square_grad.transpose(0,3,1,2))

    return smooth_square_grad, probs, preds


def generate_smooth_var_grad(pre_imgs, targets, n, sigma, model, **kwargs):
    seed_everything()
    if 'layer' not in kwargs.keys():
        kwargs['layer'] = None

    # make smooth_square_grad array
    smooth_var_grad = np.zeros(pre_imgs.shape[:1] + pre_imgs.shape[2:] + pre_imgs.shape[1:2]) # (batch_size, H, W, C)

    smooth_grad, _, _ = generate_smooth_grad(pre_imgs, targets, n, sigma, model, **kwargs)

    mean = 0
    # mean, sigma
    mins = pre_imgs.detach().numpy().min(axis=(1,2,3))
    maxs = pre_imgs.detach().numpy().max(axis=(1,2,3))
    mean = [0] * pre_imgs.size(0)
    sigma = (sigma / (maxs - mins)).squeeze()

    for _ in range(n):
        noise = np.array(list(map(normal_dist, pre_imgs, mean, sigma)))

        noisy_imgs = pre_imgs + torch.Tensor(noise)
        outputs, probs, preds = model.generate_image(noisy_imgs, targets, **kwargs)
        smooth_var_grad = smooth_var_grad + (outputs**2 - smooth_grad**2)

    smooth_var_grad = smooth_var_grad / n
    smooth_var_grad = rescale_image(smooth_var_grad.transpose(0,3,1,2))

    return smooth_var_grad, probs, preds