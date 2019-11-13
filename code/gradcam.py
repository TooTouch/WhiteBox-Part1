import torch 
import torch.nn as nn

import numpy as np
from PIL import Image

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
        conv_output, model_output = extractor.forward_pass(pre_img)

        prob = model_output.max()
        pred = model_output.argmax()

        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()

        # gradients
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        guided_gradients = extractor.gradients.data.numpy()[0]
        
        # A = w * conv_output
        target = conv_output.data.numpy()[0]
        weights = np.mean(guided_gradients, axis=(1,2))

        cam = np.zeros(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        
        # minmax scaling * 255
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)
    
        # resize to input image size
        cam = np.uint8(Image.fromarray(cam).resize((pre_img.shape[2], pre_img.shape[3]), Image.ANTIALIAS))/255

        return cam, prob, pred