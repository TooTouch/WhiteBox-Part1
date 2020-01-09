import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from attention_methods import cbam, cam, ran, warn

class SimpleCNN(nn.Module):
    def __init__(self, target, attention=None):
        super(SimpleCNN, self).__init__()
        # attention method
        self.attention = attention 
        # nb_channel and FCN size
        if target=='mnist':
            in_channels = 1
            fcn_size = 128*3*3
        elif target=='cifar10':
            in_channels = 3
            fcn_size = 128*4*4
        
        self.feature_maps = OrderedDict()
        self.pool_locs = OrderedDict()

        self.features = nn.Sequential(
            # layer1
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            # layer2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
 
            # layer3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=fcn_size, out_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.Softmax(dim=1)
        )

        # Attention Modules
        if attention=='CBAM':
            ch_lst = [32,64,128]
            for i in range(3):
                self.__setattr__('cbam%d' % i, cbam.CBAM(ch_lst[i]))
            self.cam_mlp = cam.CAM(128, 10)
        elif attention=='CAM':
            self.cam_mlp = cam.CAM(128, 10)

        print('Model Complete')

    
    def forward(self, x):
        nb_layer = 0
        for _, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, _ = layer(x)
            elif isinstance(layer, nn.BatchNorm2d) and (self.attention=='CBAM'): # CBAM
                x = self.__getattr__('cbam%d' % nb_layer)(x)
                # x = self.cbam_lst[nb_layer](x)
                x = layer(x)
                nb_layer += 1
            else:
                x = layer(x)

        if self.attention=='CAM': # CAM
            output = self.cam_mlp(x)
        else: # original
            x = x.view(x.size(0), -1)
            output = self.classifier(x)

        return output


class SimpleCNNDeconv(nn.Module):
    def __init__(self, target):
        super(SimpleCNNDeconv, self).__init__()

        self.target = target
        in_channels = 1 if self.target=='mnist' else 3

        self.features = nn.Sequential(
            # deconv1
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
    
            # deconv2
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),

            # deconv3
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32, out_channels=in_channels, kernel_size=3, padding=1)
        )

        self.conv2deconv_indices = {
            0:11, 4:7, 8:3
        }

        self.unpool2pool_indices = {
            0:11, 4:7, 8:3
        }

    def init_weight(self):
        # Load checkpoint
        weight = torch.load('../checkpoint/simple_cnn_{}.path'.format(self.target))['model']

        model = SimpleCNN(self.target)
        model.load_state_dict(weight)

        for idx, layer in enumerate(model.features):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv_indices[idx].weight.data] = layer.weight.data

    def forward(self, x, layer, pool_locs):
        if layer in self.conv2deconv_indices:
            start_idx = self.conv2deconv_indices[layer]
        else:
            raise ValueError('layer is not a conv feature map')

        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d):
                x = self.features[idx](x, pool_locs[self.unpool2pool_indices[idx]])
            else:
                x = self.features[idx](x)
        return x


    
class RAN(nn.Module):
    # Residual Attention Network (RAN)
    def __init__(self, target, nb_class=10, t_depth=56):
        '''
        params
            nb_class: number of class
            target: ['mnist','cifar10','imagenet']
            t_depth: [56,92] default is 56. 
        '''
        super(RAN, self).__init__()
        if target == 'mnist':
            size = 28
            avgpool_size = 4
            i_channel = 1
            kernel_size = 5
            stride = 1
            padding = 2
        if target == 'cifar10':
            size = 32
            avgpool_size = 4
            i_channel = 3
            kernel_size = 5
            stride = 1
            padding = 2
        elif target == 'imagenet':
            size = 224
            avgpool_size = 7
            i_channel = 3
            kernel_size = 7
            stride = 2
            padding = 3
        
        # interpolation size must be rounded up. 
        size_lst = np.ceil([(size/2, size/2), (size/4,size/4), (size/8,size/8)]).astype(int)
        size_lst = list(map(tuple, size_lst))

        self.conv = nn.Sequential(
            nn.Conv2d(i_channel, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(2, 2)

        if t_depth==56:
            self.stages = nn.Sequential(
                ran.ResidualUnit(64, 256),
                ran.AttentionModule(256, 128, size=size_lst, nb_skip=2),
                ran.ResidualUnit(128, 512, 2),
                ran.AttentionModule(512, 256, size=size_lst[1:], nb_skip=1),
                ran.ResidualUnit(256, 1024, 2),
                ran.AttentionModule(1024, 512, size=size_lst[2:], nb_skip=0)
            )
        elif t_depth==92:
            self.stages = nn.Sequential(
                ran.ResidualUnit(64, 256),
                ran.AttentionModule(256, 128, size=size_lst, nb_skip=2),
                ran.ResidualUnit(128, 512, 2),
                ran.AttentionModule(512, 256, size=size_lst[1:], nb_skip=1),
                ran.AttentionModule(256, 256, size=size_lst[1:], nb_skip=1),
                ran.ResidualUnit(256, 1024, 2),
                ran.AttentionModule(1024, 512, size=size_lst[2:], nb_skip=0),
                ran.AttentionModule(512, 512, size=size_lst[2:], nb_skip=0),
                ran.AttentionModule(512, 512, size=size_lst[2:], nb_skip=0)
            )

        self.avgpool = nn.Sequential(
            ran.ResidualUnit(512,2048),
            ran.ResidualUnit(2048,2048),
            ran.ResidualUnit(2048,2048),
            nn.AvgPool2d(kernel_size=avgpool_size, stride=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, nb_class),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        out = self.conv(x)
        out = self.maxpool(out)
        out = self.stages(out)
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)

        return out 


class WideResNetAttention(torch.nn.Module):
    """
    https://github.com/prlz77/attend-and-rectify
    WARN class
    """
    def __init__(self, target, depth=28, width=4, num_classes=10, dropout=0, attention_depth=3, attention_width=4, reg_w=0.001,
                 attention_type="softmax"):
        """ Constructor
        Args:
            depth: network depth
            width: network width
            num_classes: number of output classes
            dropout: dropout prob
            attention_depth: number of attention modules
            attention_width: number of attention heads per module
            reg_w: multihead attention regularization coefficient
            attention_type: gating function
        """
        super(WideResNetAttention, self).__init__()
        i_channel = 1 if target=='mnist' else 3
        self.pool_size = 7 if target=='mnist' else 8

        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        self.n = (depth - 4) // 6
        self.num_classes = num_classes
        widths = [int(x * width) for x in [16, 32, 64]]
        self.conv0 = torch.nn.Conv2d(i_channel, 16, 3, padding=1, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(16)
        torch.nn.init.kaiming_normal_(self.conv0.weight.data)
        self.group_0 = warn.Group(16, widths[0], self.n, 1, dropout)
        self.group_1 = warn.Group(widths[0], widths[1], self.n, 2, dropout)
        self.group_2 = warn.Group(widths[1], widths[2], self.n, 2, dropout)
        self.bn_g2 = torch.nn.BatchNorm2d(widths[2])
        self.classifier = torch.nn.Linear(widths[2], self.num_classes)
        torch.nn.init.kaiming_normal_(self.classifier.weight)

        self.attention_depth = attention_depth
        self.attention_width = attention_width
        self.reg_w = reg_w
        self.attention_type = attention_type

        self.attention_layers = [2 - i for i in range(self.attention_depth)]
        print("Attention after groups %s" % (str(self.attention_layers)))
        for i in self.attention_layers:
            self.__setattr__("att%d" % (i), warn.AttentionModule(widths[i], num_classes, attention_width, reg_w))

        ngates = self.attention_depth + 1

        self.output_gate = warn.Gate(widths[-1], ngates, gate_depth=1)

    def reg_loss(self):
        """ Compute regularization loss
        Returns: the total accumulated reg loss of the network
        """
        loss = 0
        for i in range(self.attention_depth):
            loss += self.__getattr__("att%i" % self.attention_layers[i]).reg_loss()
        return loss / self.attention_depth

    def forward(self, x):
        """ Pytorch Module forward
        Args:
            x: input
        Returns: network(input)
        """

        x = F.relu(self.bn0(self.conv0(x)), True)
        group0 = self.group_0(x)
        group1 = self.group_1(group0)
        group2 = F.relu(self.bn_g2(self.group_2(group1)), True)
        groups = [self.group_1.block_0.block_input, self.group_2.block_0.block_input, group2]

        attention_outputs = []
        for i in self.attention_layers:
            attention_outputs.append(self.__getattr__("att%d" % i)(groups[i]))

        o = F.avg_pool2d(group2, self.pool_size, 1, 0) # (b, width[-1])
        o = o.view(o.size(0), -1) # (b, width[-1], 1)

        gates = self.output_gate(o) # (b, ngates)

        attention_outputs.append(self.classifier(o).view(o.size(0), 1, -1))
        ret = warn.AttentionModule.aggregate(attention_outputs, gates, self.attention_type)

        # Memory save
        del groups, attention_outputs
        
        if self.training and self.reg_w > 0:
            reg_loss = self.reg_loss()
            return ret, reg_loss
        else:
            return ret

