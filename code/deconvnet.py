from functools import partial
import numpy as np

import torch

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
        new_img = deconv_output.data.numpy()[0].transpose(1, 2, 0) # (H,W,C)
        new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
        new_img = new_img.astype(np.uint8)
        
        return new_img, prob, pred