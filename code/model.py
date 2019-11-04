import torch
import torch.nn as nn

from collections import OrderedDict

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, fcn_size):
        super(SimpleCNN, self).__init__()

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

        print('Model Complete')
        

    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)

        x = x.view(x.size()[0], -1)
        output = self.classifier(x)

        return output


class DeconvNet(nn.Module):
    def __init__(self, target, in_channels, fcn_size):
        super(DeconvNet, self).__init__()

        self.target = target
        self.in_channels = in_channels

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
            nn.ConvTranspose2d(in_channels=32, out_channels=self.in_channels, kernel_size=3, padding=1)
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

        model = SimpleCNN(self.in_channels, self.fcn_size)
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