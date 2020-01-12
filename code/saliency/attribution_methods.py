import torch 
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import h5py
from tqdm import tqdm

from collections import OrderedDict
from functools import partial

from utils import rescale_image, resize_image, save_saliency_map
from models import SimpleCNNDeconv


class ConvOutput(object):
    '''
    Convolution Output
    '''
    def __init__(self, model, **kwargs):
        # seqeuntial name
        self.seq_name = 'features' if 'seq_name' not in kwargs.keys() else kwargs['seq_name']
        # model
        self.model = model
        # evaluation = model
        self.model.eval()
        # hook
        self.conv_outputs = OrderedDict()
        self.hook_layers()

    def hook_layers(self):
        def hook_forward(module, input, output, key):
            if isinstance(module, torch.nn.MaxPool2d):
                self.conv_outputs[key] = output[0]
            else:
                self.conv_outputs[key] = output

        for idx, layer in enumerate(self.model._modules.get(self.seq_name)):
            layer.register_forward_hook(partial(hook_forward, key=idx))

    def generate_image(self, pre_imgs, targets, **kwargs):
        # default
        layer = 11 if 'layer' not in kwargs.keys() else kwargs['layer']
        color = False if 'color' not in kwargs.keys() else kwargs['color']

        # convert target type to LongTensor
        targets = torch.LongTensor(targets)

        # prediction
        pre_imgs = Variable(pre_imgs, requires_grad=True)
        outputs = self.model(pre_imgs)
        probs, preds = outputs.detach().max(1)

        # n th convolution block output
        conv_out = self.conv_outputs[layer].mean(axis=1).detach().numpy()

        # minmax scaling * 255
        conv_out = rescale_image(conv_out, channel=False)
        
        colors = [color] * conv_out.shape[0]
        gradcams = np.array(list(map(resize_image, conv_out, pre_imgs, colors)))

        return (gradcams, probs.numpy(), preds.numpy())

    def save(self, dataloader, save_dir, **kwargs):
        save_saliency_map(self.generate_image, dataloader, save_dir, 'ConvOutput', **kwargs)
        


class VanillaBackprop(object):
    '''
    Vanilla Backpropagation
    '''
    def __init__(self, model, **kwargs):
        self.model = model 
        # evaluation mode
        self.model.eval()

    def generate_image(self, pre_imgs, targets, **kwargs):
        # convert target type to LongTensor
        targets = torch.LongTensor(targets)

        # prediction
        pre_imgs = Variable(pre_imgs, requires_grad=True)
        outputs = self.model(pre_imgs)

        # calculate gradients
        self.model.zero_grad()

        one_hot_output = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1).detach()
        outputs.backward(gradient=one_hot_output)
        probs, preds = outputs.detach().max(1)

        # rescale saliency map
        sal_maps = rescale_image(pre_imgs.grad.numpy())

        return (sal_maps, probs.numpy(), preds.numpy())
    
    def save(self, dataloader, save_dir, **kwargs):
        save_saliency_map(self.generate_image, dataloader, save_dir, 'Vanilla Backprop', **kwargs)


class IntegratedGradients(object):
    '''
    Integrated Gradients
    '''
    def __init__(self, model, **kwargs):
        self.model = model
        # evaluation mode
        self.model.eval()

    def generate_images_on_linear_path(self, input_images, steps):
        # divide image by step list
        step_list = np.arange(steps+1)/steps
        xbar_list = [input_images*step for step in step_list]
        return xbar_list 

    def generate_gradients(self, pre_imgs, targets, **kwargs):
        # prediction
        pre_imgs = Variable(pre_imgs, requires_grad=True)
        outputs = self.model(pre_imgs)
        
        # calculate gradients
        self.model.zero_grad()

        one_hot_output = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1).detach()
        outputs.backward(gradient=one_hot_output)
        probs, preds = outputs.detach().max(1)

        imgs = pre_imgs.grad.numpy()

        return (imgs, probs.numpy(), preds.numpy())

    def generate_image(self, pre_imgs, targets, **kwargs):
        # default
        steps = 10 if 'steps' not in kwargs.keys() else kwargs['steps']

        # convert target type to LongTensor
        targets = torch.LongTensor(targets)

        # divide image
        xbar_list = self.generate_images_on_linear_path(pre_imgs, steps)
        sal_maps = np.zeros(pre_imgs.size())

        # make saliency map from divided images
        for xbar_image in xbar_list:
            single_integrated_grad, probs, preds = self.generate_gradients(xbar_image, targets)
            sal_maps = sal_maps + (single_integrated_grad/steps)
        
        # rescale saliency map
        sal_maps = rescale_image(sal_maps)
        
        return (sal_maps, probs, preds)

    def save_saliency_map(self, dataloader, save_dir, **kwargs):
        save_saliency_map(self.generate_image, dataloader, save_dir, 'Integrated Gradients', **kwargs)


class GuidedBackprop(object):
    '''
    Guided Backpropagation
    '''
    def __init__(self, model, **kwargs):
        # seqeuntial name
        self.seq_name = 'features' if 'seq_name' not in kwargs.keys() else kwargs['seq_name']
        # model
        self.model = model    
        # evaluation mode
        self.model.eval()
        # update relu
        self.forward_relu_outputs = []
        self.update_relus()

    def update_relus(self):
        def relu_backward_hook_function(module, grad_in, grad_out):
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]

            return (modified_grad_out,)

        def relu_forward_hook_function(module, input, output):
            self.forward_relu_outputs.append(output)

        for layer in self.model._modules.get(self.seq_name):
            if isinstance(layer, nn.ReLU):
                layer.register_backward_hook(relu_backward_hook_function)
                layer.register_forward_hook(relu_forward_hook_function)

    def generate_image(self, pre_imgs, targets, **kwargs):
        # convert target type to LongTensor
        targets = torch.LongTensor(targets)

        # prediction
        pre_imgs = Variable(pre_imgs, requires_grad=True)
        outputs = self.model(pre_imgs)

        # calculate gradients
        self.model.zero_grad()
        
        one_hot_output = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1).detach()
        outputs.backward(gradient=one_hot_output)
        probs, preds = outputs.detach().max(1)

        sal_maps = rescale_image(pre_imgs.grad.numpy())

        return (sal_maps, probs.numpy(), preds.numpy())

    def save(self, dataloader, save_dir, **kwargs):
        save_saliency_map(self.generate_image, dataloader, save_dir, 'Guided Backprop', **kwargs)


class GradCAM(object):
    '''
    Gradient Class Activation Map (Grad-CAM)
    '''
    def __init__(self, model, **kwargs):
        # seqeuntial name
        self.seq_name = 'features' if 'seq_name' not in kwargs.keys() else kwargs['seq_name']
        # model
        self.model = model
        # evaluation mode
        self.model.eval()
        # hook
        self.hook_layers()
        # initial dicts
        self.conv_outputs = OrderedDict()
        self.gradients = OrderedDict()
    
    def hook_layers(self):
        def hook_forward(module, input, output, key):
            if isinstance(module, torch.nn.MaxPool2d):
                self.conv_outputs[key] = output[0]
            else:
                self.conv_outputs[key] = output

        def hook_backward(module, input, output, key):
            self.gradients[key] = output[0]

        for idx, layer in enumerate(self.model._modules.get(self.seq_name)):
            layer.register_forward_hook(partial(hook_forward, key=idx))
            layer.register_backward_hook(partial(hook_backward, key=idx))
            
    def generate_image(self, pre_imgs, targets, **kwargs):
        # default
        layer = 8 if 'layer' not in kwargs.keys() else kwargs['layer']
        color = False if 'color' not in kwargs.keys() else kwargs['color']

        # convert target type to LongTensor
        targets = torch.LongTensor(targets)

        # prediction
        pre_imgs = Variable(pre_imgs, requires_grad=True)
        outputs = self.model(pre_imgs)

        # calculate gradients
        self.model.zero_grad()

        one_hot_output = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1).detach()
        outputs.backward(gradient=one_hot_output)
        probs, preds = outputs.detach().max(1)

        gradients = self.gradients[layer].numpy()
        
        # A = w * conv_output
        convs = self.conv_outputs[layer].detach().numpy()
        weights = np.mean(gradients, axis=(2,3))
        weights = weights.reshape(weights.shape + (1,1,))

        gradcams = weights * convs
        gradcams = gradcams.sum(axis=1)

        # relu
        gradcams = np.maximum(gradcams, 0)
        
        # minmax scaling * 255
        gradcams = rescale_image(gradcams, channel=False)

        # resize images
        colors = [color] * gradcams.shape[0]
        gradcams = np.array(list(map(resize_image, gradcams, pre_imgs, colors)))

        return (gradcams, probs.numpy(), preds.numpy())

    def save(self, dataloader, save_dir, **kwargs):
        save_saliency_map(self.generate_image, dataloader, save_dir, 'GradCAM', **kwargs)


class CAM(object):
    '''
    Class Activation Map (CAM)
    '''
    def __init__(self, model, **kwargs):
        # seqeuntial name
        self.seq_name = 'features' if 'seq_name' not in kwargs.keys() else kwargs['seq_name']
        # model
        self.model = model
        # evaluation mode
        self.model.eval()
        # hook
        self.hook_layers()
        # initial dicts
        self.conv_outputs = OrderedDict()
    
    def hook_layers(self):
        def hook_forward(module, input, output, key):
            if isinstance(module, torch.nn.MaxPool2d):
                self.conv_outputs[key] = output[0]
            else:
                self.conv_outputs[key] = output

        for idx, layer in enumerate(self.model._modules.get(self.seq_name)):
            layer.register_forward_hook(partial(hook_forward, key=idx))
            
    def generate_image(self, pre_imgs, targets, **kwargs):
        # last layer idx
        layer = 11 if 'layer' not in kwargs.keys() else kwargs['layer']
        color = False if 'color' not in kwargs.keys() else kwargs['color']

        # convert target type to LongTensor
        targets = torch.LongTensor(targets)

        # prediction
        pre_imgs = Variable(pre_imgs, requires_grad=True)
        outputs = self.model(pre_imgs)
        probs, preds = outputs.detach().max(1)

        # last layer output
        last_layer_output = self.conv_outputs[layer].detach().numpy() # (B, C, H, W)

        # w_k 
        w_k = self.model.cam_mlp.mlp[0].weight.detach().numpy() # (nb_class, C)
        b_w_k = np.zeros((targets.shape[0], w_k.shape[1]))
        for i in range(targets.shape[0]):
            b_w_k[i] = w_k[targets[i]]
        b_w_k = b_w_k.reshape(b_w_k.shape + (1,1,)) # (B, C, 1, 1)

        # b_w_k x last layer output
        cams = (b_w_k * last_layer_output).sum(1)
        
        # minmax scaling * 255
        cams = rescale_image(cams, channel=False)
    
        # resize to input image size
        colors = [color] * cams.shape[0]
        cams = np.array(list(map(resize_image, cams, pre_imgs, colors)))

        return (cams, probs.numpy(), preds.numpy())

    def save(self, dataloader, save_dir, **kwargs):
        save_saliency_map(self.generate_image, dataloader, save_dir, 'CAM', **kwargs)
 

class DeconvNet(object):
    '''
    Deconvolutional Network
    '''
    def __init__(self, model, deconv_model, **kwargs):
        # seqeuntial name
        self.seq_name = 'features' if 'seq_name' not in kwargs.keys() else kwargs['seq_name']
        # model
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

        for idx, layer in enumerate(self.model._modules.get(self.seq_name)):
            layer.register_forward_hook(partial(hook, key=idx))

    def generate_image(self, pre_imgs, targets, **kwargs):
        # default
        layer = 0 if 'layer' not in kwargs.keys() else kwargs['layer']

        # convert target type to LongTensor
        targets = torch.LongTensor(targets)

        # prediction
        outputs = self.model(pre_imgs).detach()
        probs, preds = outputs.max(1)

        # output deconvnet
        deconv_outputs = self.deconv_model(self.model.feature_maps[layer], layer, self.model.pool_locs)

        # denormalization
        deconv_outputs = deconv_outputs.data.numpy()
        deconv_outputs = rescale_image(deconv_outputs)
        
        return (deconv_outputs, probs.numpy(), preds.numpy())
    
    def save(self, dataloader, save_dir, **kwargs):
        save_saliency_map(self.generate_image, dataloader, save_dir, 'DeconvNet', **kwargs)


class GuidedGradCAM(object):
    '''
    Guided backpropagation x Grad CAM
    '''
    def __init__(self, model, **kwargs):
        self.GC_model = GradCAM(model, **kwargs)
        self.GB_model = GuidedBackprop(model, **kwargs)

    def generate_image(self, pre_imgs, targets, **kwargs):
        # default
        layer = 8 if 'layer' not in kwargs.keys() else kwargs['layer']        
        color = False if 'color' not in kwargs.keys() else kwargs['color']

        # make saliency map by GradCAM & Guided Backprop 
        output_GC, probs, preds = self.GC_model.generate_image(pre_imgs, targets, layer=layer, color=color)
        output_GB, _, _ = self.GB_model.generate_image(pre_imgs, targets)

        # GradCAM x Guided Backprop
        sal_maps = np.multiply(output_GC, output_GB)
        sal_maps = rescale_image(sal_maps.transpose(0,3,1,2))

        return sal_maps, probs, preds

    def save(self, dataloader, save_dir, **kwargs):
        save_saliency_map(self.generate_image, dataloader, save_dir, 'Guided GradCAM', **kwargs)

    
class InputBackprop(object):
    '''
    Input x Vanilla Backpropagation
    '''
    def __init__(self, model, **kwargs):
        self.VBP_model = VanillaBackprop(model, **kwargs)

    def generate_image(self, pre_imgs, targets, **kwargs):
        # make saliency map by VBP
        output_VBP, probs, preds = self.VBP_model.generate_image(pre_imgs, targets, **kwargs)

        # rescale input image
        input_img = pre_imgs.detach().numpy()
        input_img = rescale_image(input_img)

        # input image x VBP
        sal_maps = np.multiply(output_VBP, input_img)
        sal_maps = rescale_image(sal_maps.transpose(0,3,1,2))

        return sal_maps, probs, preds

    def save(self, dataloader, save_dir, **kwargs):
        save_saliency_map(self.generate_image, dataloader, save_dir, 'Input x Backprop', **kwargs)
