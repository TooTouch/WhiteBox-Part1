import torch 
from utils import rescale_image
from collections import OrderedDict
from functools import partial
import numpy as np

class VanillaBackprop:
    def __init__(self, model):
        self.model = model 

        # evaluation mode
        self.model.eval()
        # hook 
        self.hook_layers()

    def hook_layers(self):
        def hook(module, grad_in, grad_out, key):
            self.model.gradients[key] = grad_in[0]

        for pos, module in enumerate(self.model._modules.get('features')):
            module.register_backward_hook(partial(hook, key=pos))

    def generate_image(self, pre_img, layer, target_class=None):
        probs = self.model(pre_img)
        prob = probs.max().item()
        pred = probs.argmax().item()

        self.model.zero_grad()

        if target_class is None:
            target_class = np.argmax(probs.data.numpy())

        one_hot_output = torch.FloatTensor(1, probs.size()[-1])
        one_hot_output[0][target_class] = 1

        probs.backward(gradient=one_hot_output)

        output = self.model.gradients[layer].numpy()[0] # (nb_filter, W, H)
        output = rescale_image(output)

        return output, prob, pred
