import torch 
import torch.nn as nn
import numpy as np

from collections import OrderedDict
from functools import partial

from utils import rescale_image
from PIL import Image


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

    def generate_image(self, pre_img, layer=0, target_class=None):
        probs = self.model(pre_img)
        prob = probs.max().item()
        pred = probs.argmax().item()

        self.model.zero_grad()

        if target_class is None:
            target_class = pred

        one_hot_output = torch.FloatTensor(1, probs.size()[-1])
        one_hot_output[0][target_class] = 1

        probs.backward(gradient=one_hot_output)

        output = self.model.gradients[layer].numpy()[0] # (nb_filter, W, H)
        output = rescale_image(output)

        return (output, prob, pred)


class IntegratedGradients:
    def __init__(self, model):
        self.model = model
        self.gradients = None

        self.model.eval()

        self.hook_layers()

    def hook_layers(self):
        def hook(module, grad_in, grad_out, key):
            self.model.gradients[key] = grad_in[0]

        for pos, module in enumerate(self.model._modules.get('features')):
            module.register_backward_hook(partial(hook, key=pos))

    def generate_images_on_linear_path(self, input_image, steps):
        step_list = np.arange(steps+1)/steps
        xbar_list = [input_image*step for step in step_list]
        return xbar_list 

    def generate_gradients(self, pre_img, layer=0, target_class=None):
        probs = self.model(pre_img)
        prob = probs.max().item()
        pred = probs.argmax().item()

        self.model.zero_grad()

        if target_class == None:
            target_class = pred
        one_hot_output = torch.FloatTensor(1, probs.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        probs.backward(gradient=one_hot_output)
        gradients_arr = self.model.gradients[layer].data.numpy()[0]

        return gradients_arr, prob, pred

    def generate_image(self, pre_img, layer=0, target_class=None, steps=100):
        xbar_list = self.generate_images_on_linear_path(pre_img, steps)
        output = np.zeros(pre_img.size()[1:])

        for xbar_image in xbar_list:
            single_integrated_grad, prob, pred = self.generate_gradients(xbar_image, layer, target_class)
            output = output + single_integrated_grad / steps
            
        output = rescale_image(output)
        
        return output, prob, pred


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
            if not isinstance(module, nn.ReLU):
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
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_image(self, pre_img, layer=0, target_class=None):
        probs = self.model(pre_img)
        prob = probs.max().item()
        pred = probs.argmax().item()

        self.model.zero_grad()

        if target_class is None:
            target_class = pred
        one_hot_output = torch.FloatTensor(1, probs.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        probs.backward(gradient=one_hot_output)

        output = self.model.gradients[layer].data.numpy()[0]
        output = rescale_image(output)

        return (output, prob, pred)


class CamExtractor:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            if isinstance(module, nn.MaxPool2d):
                x, location = module(x)
            else:
                x = module(x)
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x
        return conv_output, x

    def forward_pass(self, x):
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)

        return conv_output, x

class GradCam:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_image(self, pre_img, layer, target_class=None):
        extractor = CamExtractor(self.model, layer)
        conv_output, probs = extractor.forward_pass(pre_img)

        prob = probs.max()
        pred = probs.argmax()

        if target_class is None:
            target_class = pred
        one_hot_output = torch.FloatTensor(1, probs.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()

        # gradients
        probs.backward(gradient=one_hot_output, retain_graph=True)
        guided_gradients = extractor.gradients.data.numpy()[0]
        
        # A = w * conv_output
        target = conv_output.data.numpy()[0]
        weights = np.mean(guided_gradients, axis=(1,2))

        output = np.zeros(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            output += w * target[i, :, :]
        
        # minmax scaling * 255
        output = np.maximum(output, 0)
        output = (output - np.min(output)) / (np.max(output) - np.min(output))
        cam = np.uint8(output * 255)
    
        # resize to input image size
        output = np.uint8(Image.fromarray(output).resize((pre_img.shape[2], pre_img.shape[3]), Image.ANTIALIAS))/255

        return (output, prob, pred)


class DeconvNet:
    def __init__(self, model, deconv_model):
        self.model = model
        self.deconv_model = deconv_model

        # evalution mode
        self.model.eval()
        # hook
        self.hook_layers()

    def hook_layers(self):
        def hook(module, input, output, key):
            if isinstance(module, torch.nn.MaxPool2d):
                self.model.feature_maps[key] = output[0]
                self.model.pool_locs[key] = output[1]
            else:
                self.model.feature_maps[key] = output

        for idx, layer in enumerate(self.model._modules.get('features')):
            layer.register_forward_hook(partial(hook, key=idx))

    def generate_image(self, pre_img, layer, max_activation=False, target_class=None):

        # prediction
        probs = self.model(pre_img).detach()
        prob = probs.max().item()
        pred = probs.argmax().item()
        
        # feature size
        num_feat = self.model.feature_maps[layer].shape[1]
        new_feat_map = self.model.feature_maps[layer].clone()

        if max_activation:
            # max feature
            act_lst = []
            for i in range(0, num_feat):
                choose_map = new_feat_map[0, i, :, :]
                activation = torch.max(choose_map)
                act_lst.append(activation.item())

            act_lst = np.array(act_lst)
            mark = np.argmax(act_lst)

            choose_map = new_feat_map[0, mark, :, :]
            max_activation = torch.max(choose_map)

            if mark == 0:
                new_feat_map[:, 1:, :, :] = 0
            else:
                new_feat_map[:, :mark, :, :] = 0
                if mark != num_feat - 1:
                    new_feat_map[:, mark + 1:, :, :] = 0

            choose_map = torch.where(choose_map == max_activation,
                                    choose_map,
                                    torch.zeros(choose_map.shape))

            new_feat_map[0, mark, :, :] = choose_map

            # output deconvnet
            deconv_output = self.deconv_model(new_feat_map, layer, self.model.pool_locs)

        else:
            # output deconvnet
            deconv_output = self.deconv_model(self.model.feature_maps[layer], layer, self.model.pool_locs)

        # denormalization
        output = deconv_output.data.numpy()[0].transpose(1, 2, 0) # (H,W,C)
        output = (output - output.min()) / (output.max() - output.min()) * 255
        output = output.astype(np.uint8)
        
        return (output, prob, pred)