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


        