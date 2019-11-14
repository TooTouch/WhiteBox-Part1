import numpy as np
import torch
from torch.autograd import Variable
from utils import seed_everything

def generate_smooth_grad(pre_img, output_size, n, sigma, layer, model, target_class=None):
    seed_everything()
    
    smooth_grad = np.zeros(output_size)

    mean = 0
    sigma = sigma / (torch.max(pre_img) - torch.min(pre_img)).item()

    for i in range(n):
        noise = Variable(torch.zeros(pre_img.size()).normal_(mean, sigma**2))

        noisy_img = pre_img + noise
        output_img, prob, pred = model.generate_image(noisy_img, layer, target_class)
        smooth_grad = smooth_grad + output_img

    smooth_grad = smooth_grad / n

    return smooth_grad, prob, pred

def generate_smooth_square_grad(pre_img, output_size, n, sigma, layer, model, target_class=None):
    seed_everything()
    
    smooth_square_grad = np.zeros(output_size)

    mean = 0
    sigma = sigma / (torch.max(pre_img) - torch.min(pre_img)).item()

    for i in range(n):
        noise = Variable(torch.zeros(pre_img.size()).normal_(mean, sigma**2))

        noisy_img = pre_img + noise
        output_img, prob, pred = model.generate_image(noisy_img, layer, target_class)
        smooth_square_grad = smooth_square_grad + output_img**2

    smooth_square_grad = smooth_square_grad / n


def generate_smooth_var_grad(pre_img, output_size, n, sigma, layer, model, target_class=None):
    seed_everything()

    smooth_var_grad = np.zeros(output_size)

    smooth_grad, _, _ = generate_smooth_grad(pre_img, output_size, n, sigma, layer, model, target_class)

    mean = 0
    sigma = sigma / (torch.max(pre_img) - torch.min(pre_img)).item()

    for i in range(n):
        noise = Variable(torch.zeros(pre_img.size()).normal_(mean, sigma**2))

        noisy_img = pre_img + noise
        output_img, prob, pred = model.generate_image(noisy_img, layer, target_class)
        smooth_var_grad = smooth_var_grad + (output_img**2 - smooth_grad**2)

    smooth_var_grad = smooth_var_grad / n

    return smooth_var_grad, prob, pred