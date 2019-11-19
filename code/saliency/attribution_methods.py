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

        re_imgs = rescale_image(pre_imgs.grad.numpy())

        return (re_imgs, probs.numpy(), preds.numpy())


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
        outputs = np.zeros(pre_imgs.size())

        for xbar_image in xbar_list:
            single_integrated_grad, probs, preds = self.generate_gradients(xbar_image, targets)
            outputs = outputs + (single_integrated_grad/kwargs['steps'])
            
        outputs = rescale_image(outputs)
        
        return (outputs, probs, preds)


class GuidedBackprop(object):
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

    def generate_image(self, pre_imgs, layer=0, target_class=None):
        probs = self.model(pre_imgs)
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


class CamExtractor(object):
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

class GradCAM(object):
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_image(self, pre_imgs, layer, target_class=None):
        extractor = CamExtractor(self.model, layer)
        conv_output, probs = extractor.forward_pass(pre_imgs)

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
        output = np.uint8(output * 255)
    
        # resize to input image size
        output = np.uint8(Image.fromarray(output).resize((pre_imgs.shape[2], pre_imgs.shape[3]), Image.ANTIALIAS))/255

        return (output, prob, pred)


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

    def generate_image(self, pre_imgs, layer, target_class):
        output_GC, prob, pred = self.GC_model.generate_image(pre_imgs, 8, target_class)
        output_GB, _, _ = self.GB_model.generate_image(pre_imgs, layer, target_class)
        output = np.multiply(output_GC, output_GB.transpose(2,0,1))
        output = output.transpose(1,2,0)

        return output, prob, pred

    
class InputBackprop(object):
    def __init__(self, model):
        self.VBP_model = VanillaBackprop(model)

    def generate_image(self, pre_imgs, targets, **kwargs):
        output_VBP, prob, pred = self.VBP_model.generate_image(pre_imgs, targets)
        input_img = pre_imgs.detach().numpy()
        input_img = rescale_image(input_img)
        output = np.multiply(output_VBP, input_img)

        return output, prob, pred


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