import torch
from torch.nn import ReLU
from functools import partial
from utils import rescale_image

class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []

        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook(module, grad_in, grad_out, key):
            self.model.gradients[key] = grad_in[0]

        for pos, module in enumerate(self.model._modules.get('features')):
            if not isinstance(module, ReLU):
                module.register_backward_hook(partial(hook, key=pos))

    def update_relus(self):
        def relu_backward_hook_function(module, grad_in, grad_out):
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]

            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            self.forward_relu_outputs.append(ten_out)

        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_image(self, pre_img, layer, target_class=None):
        probs = self.model(pre_img)
        prob = probs.max().item()
        pred = probs.argmax().item()

        self.model.zero_grad()

        if target_class is None:
            target_class = np.argmax(probs.data.numpy())
        one_hot_output = torch.FloatTensor(1, probs.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        probs.backward(gradient=one_hot_output)

        output = self.model.gradients[layer].data.numpy()[0]
        output = rescale_image(output)

        return output, prob, pred