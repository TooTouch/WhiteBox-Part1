import torch 
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import h5py
from tqdm import tqdm

import cv2
from PIL import Image

from collections import OrderedDict
from functools import partial

from utils import rescale_image
from model import SimpleCNNDeconv



class VanillaBackprop(object):
    def __init__(self, model):
        self.model = model 
        # evaluation mode
        self.model.eval()

    def generate_image(self, pre_imgs, targets, **kwargs):
        pre_imgs = Variable(pre_imgs, requires_grad=True)
        outputs = self.model(pre_imgs)

        self.model.zero_grad()

        one_hot_output = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1).detach()
        outputs.backward(gradient=one_hot_output)
        probs, preds = outputs.detach().max(1)

        sal_maps = rescale_image(pre_imgs.grad.numpy())

        return (sal_maps, probs.numpy(), preds.numpy())


class IntegratedGradients(object):
    def __init__(self, model):
        self.model = model
        # evaluation mode
        self.model.eval()

    def generate_images_on_linear_path(self, input_images, **kwargs):
        step_list = np.arange(kwargs['steps']+1)/kwargs['steps']
        xbar_list = [input_images*step for step in step_list]
        return xbar_list 

    def generate_gradients(self, pre_imgs, targets, **kwargs):
        pre_imgs = Variable(pre_imgs, requires_grad=True)
        outputs = self.model(pre_imgs)
        
        self.model.zero_grad()

        one_hot_output = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1).detach()
        outputs.backward(gradient=one_hot_output)
        probs, preds = outputs.detach().max(1)

        imgs = pre_imgs.grad.numpy()

        return (imgs, probs.numpy(), preds.numpy())

    def generate_image(self, pre_imgs, targets, **kwargs):
        if 'steps' not in kwargs.keys():
            kwargs['steps'] = 10

        xbar_list = self.generate_images_on_linear_path(pre_imgs, **kwargs)
        sal_maps = np.zeros(pre_imgs.size())

        for xbar_image in xbar_list:
            single_integrated_grad, probs, preds = self.generate_gradients(xbar_image, targets)
            sal_maps = sal_maps + (single_integrated_grad/kwargs['steps'])
            
        sal_maps = rescale_image(sal_maps)
        
        return (sal_maps, probs, preds)


class GuidedBackprop(object):
    def __init__(self, model):
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

        def relu_forward_hook_function(module, ten_in, ten_out):
            self.forward_relu_outputs.append(ten_out)

        for pos, module in self.model.features._modules.items():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_image(self, pre_imgs, targets, **kwargs):
        pre_imgs = Variable(pre_imgs, requires_grad=True)
        outputs = self.model(pre_imgs)

        self.model.zero_grad()
        
        one_hot_output = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1).detach()
        outputs.backward(gradient=one_hot_output)
        probs, preds = outputs.detach().max(1)

        sal_maps = rescale_image(pre_imgs.grad.numpy())

        return (sal_maps, probs.numpy(), preds.numpy())

class GradCAM(object):
    def __init__(self, model):
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

        for idx, layer in enumerate(self.model._modules.get('features')):
            layer.register_forward_hook(partial(hook_forward, key=idx))
            layer.register_backward_hook(partial(hook_backward, key=idx))
            
    def generate_image(self, pre_imgs, targets, **kwargs):
        if 'color' not in kwargs.keys():
            kwargs['color'] = False

        pre_imgs = Variable(pre_imgs, requires_grad=True)
        outputs = self.model(pre_imgs)

        self.model.zero_grad()

        one_hot_output = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1).detach()
        outputs.backward(gradient=one_hot_output)
        probs, preds = outputs.detach().max(1)

        gradients = self.gradients[kwargs['layer']].numpy()
        
        # A = w * conv_output
        convs = self.conv_outputs[kwargs['layer']].detach().numpy()
        weights = np.mean(gradients, axis=(2,3))
        weights = weights.reshape(weights.shape + (1,1,))

        gradcams = weights * convs
        gradcams = gradcams.sum(axis=1)

        # relu
        gradcams = np.maximum(gradcams, 0)

        # minmax scaling * 255
        mins = gradcams.min(axis=(1,2))
        mins = mins.reshape(mins.shape + (1,1,))
        maxs = gradcams.max(axis=(1,2))
        maxs = maxs.reshape(maxs.shape + (1,1,))

        gradcams = (gradcams - mins)/(maxs - mins)
        gradcams = np.uint8(gradcams * 255)
    
        # resize to input image size
        def resize_image(gradcam, origin_image, color):
            img = np.uint8(Image.fromarray(gradcam).resize((origin_image.shape[-2:]), Image.ANTIALIAS))/255
            if color:
                # output_GC (H,W) to (H,W,C)
                img = cv2.applyColorMap(np.uint8(img*255), cv2.COLORMAP_JET)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.expand_dims(img, axis=2)
            
            return img

        colors = [kwargs['color']] * gradcams.shape[0]
        gradcams = np.array(list(map(resize_image, gradcams, pre_imgs, colors)))

        

        return (gradcams, probs.numpy(), preds.numpy())


class DeconvNet(object):
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

    def generate_image(self, pre_imgs, targets, **kwargs):
        # prediction
        outputs = self.model(pre_imgs).detach()
        probs, preds = outputs.max(1)
        
        # feature size
        num_feat = self.model.feature_maps[kwargs['layer']].shape[1]
        new_feat_map = self.model.feature_maps[kwargs['layer']].clone()

        # output deconvnet
        deconv_outputs = self.deconv_model(self.model.feature_maps[kwargs['layer']], kwargs['layer'], self.model.pool_locs)

        # denormalization
        deconv_outputs = deconv_outputs.data.numpy()
        deconv_outputs = rescale_image(deconv_outputs)
        
        return (deconv_outputs, probs.numpy(), preds.numpy())


class GuidedGradCAM(object):
    def __init__(self, model):
        self.GC_model = GradCAM(model)
        self.GB_model = GuidedBackprop(model)

    def generate_image(self, pre_imgs, targets, **kwargs):
        if 'layer' not in kwargs.keys():
            kwargs['layer'] = 8
        if 'color' not in kwargs.keys():
            kwargs['color'] = False

        output_GC, probs, preds = self.GC_model.generate_image(pre_imgs, targets, **kwargs)
        output_GB, _, _ = self.GB_model.generate_image(pre_imgs, targets, **kwargs)

        sal_maps = np.multiply(output_GC, output_GB)
        sal_maps = rescale_image(sal_maps.transpose(0,3,1,2))

        return sal_maps, probs, preds

    
class InputBackprop(object):
    def __init__(self, model):
        self.VBP_model = VanillaBackprop(model)

    def generate_image(self, pre_imgs, targets, **kwargs):
        output_VBP, probs, preds = self.VBP_model.generate_image(pre_imgs, targets, **kwargs)
        input_img = pre_imgs.detach().numpy()
        input_img = rescale_image(input_img)

        sal_maps = np.multiply(output_VBP, input_img)
        sal_maps = rescale_image(sal_maps.transpose(0,3,1,2))

        return sal_maps, probs, preds


def save_saliency_map(target, method):
    # data load
    if target == 'mnist':
        trainset, validset, testloader = mnist_load(shuffle=False)
    elif target == 'cifar10':
        trainset, validset, testloader = cifar10_load(shuffle=False, augmentation=False)

    # model load
    weights = torch.load('../checkpoint/simple_cnn_{}.pth'.format(target))
    model = SimpleCNN(target)
    model.load_state_dict(weights['model'])

    # saliency map
    attribute_method, layer = saliency_map_choice(method, model, target)
    
    # make saliency_map
    trainset_saliency_map = np.zeros(trainset.dataset.data.shape, dtype=np.float32)
    validset_saliency_map = np.zeros(validset.dataset.data.shape, dtype=np.float32)
    testset_saliency_map = np.zeros(testset.dataset.data.shape, dtype=np.float32)

    for i in tqdm(range(trainset.dataset.data), desc='trainset'):
        img = trainset.dataset.data[i]
        target = trainset.dataset.targets[i]
        pre_imgs = trainset.dataset.transform(np.array(img)).unsqueeze(0)
        output = attribute_method.generate_image(pre_imgs, layer, target)        
        if (target=='cifar10') and (method=='GC'):
            # GradCAM output shape is (W,H)
            output = cv2.applyColorMap(np.uint8(output*255), cv2.COLORMAP_JET)
            output = cv2.cvtCOLOR(output, cv2.COLOR_BGR2RGB)
        trainset_saliency_map[i] = output    
    
    for i in tqdm(range(validset.dataset.data), desc='validset'):
        img = validset.dataset.data[i]
        target = validset.dataset.targets[i]
        pre_imgs = validset.dataset.transform(np.array(img)).unsqueeze(0)
        output = attribute_method.generate_image(pre_imgs, layer, target)
        if (target=='cifar10') and (method=='GC'):
            # GradCAM output shape is (W,H)
            output = cv2.applyColorMap(np.uint8(output*255), cv2.COLORMAP_JET)
            output = cv2.cvtCOLOR(output, cv2.COLOR_BGR2RGB)        
        validset_saliency_map[i] = output    

    for i in tqdm(range(testset.dataset.data), desc='testset'):
        img = testset.dataset.data[i]
        target = testset.dataset.targets[i]
        pre_imgs = validset.dataset.transform(np.array(img)).unsqueeze(0)
        output = attribute_method.generate_image(pre_imgs, layer, target)
        if (target=='cifar10') and (method=='GC'):
            # GradCAM output shape is (W,H)
            output = cv2.applyColorMap(np.uint8(output*255), cv2.COLORMAP_JET)
            output = cv2.cvtCOLOR(output, cv2.COLOR_BGR2RGB)        
        testset_saliency_map[i] = output    

    # make saliency_map directory 
    if not os.path.isdir('../saliency_map'):
        os.mkdir('../saliency_map')

    # save saliency map to hdf5
    with h5py.File('../saliency_map/{}_{}.hdf5'.format(target, method),'w') as hf:
        hf.create_dataset('trainset',data=trainset_saliency_map)
        hf.create_dataset('validset',data=validset_saliency_map)
        hf.create_dataset('testset',data=testset_saliency_map)
        hf.close()


def saliency_map_choice(method, model, target=None):
    if method == 'VBP':
        saliency_map = VanillaBackprop(s.model)
        layer = 0 
    elif method == 'GB':
        saliency_map = GuidedBackprop(model)
        layer = 0 
    elif method == 'IG':
        saliency_map = IntegratedGradients(model)
        layer = 0 
    elif method == 'GC':
        saliency_map = GradCAM(model)
        layer = 11
    elif method == 'DeconvNet':
        deconv_model = SimpleCNNDeconv(target)
        saliency_map = DeconvNet(model, deconv_model)
        layer = 0

    return saliency_map, layer
    
    
def make_saliency_map():
    target_lst = ['mnist','cifar10']
    method_lst = ['VBP','IG','GB','GC','DeconvNet']
    for target in target_lst:
        for method in method_lst:
            save_saliency_map(target, method)