import torch
import torch.nn as nn
import torch.nn.functional as F 


class ResidualUnit(nn.Module):
    def __init__(self, i_channel, o_channel, stride=1):
        '''
        params
            i_channel: number of input channels
            o_channel: number of output channels
            stride: stride
        '''
        super(ResidualUnit, self).__init__()
        self.i_channel = i_channel
        self.o_channel = o_channel
        self.stride = stride

        self.conv1 = nn.Conv2d(i_channel, o_channel//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(o_channel//4)
        self.conv2 = nn.Conv2d(o_channel//4, o_channel//4, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(o_channel//4)
        self.conv3 = nn.Conv2d(o_channel//4, o_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(o_channel)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(i_channel, o_channel, 1, stride=stride),
            nn.BatchNorm2d(o_channel)
        )

    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if (self.stride != 1) or (self.i_channel != self.o_channel):
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class SoftMaskBranch(nn.Module):
    def __init__(self, channel, r, size, nb_skip, pool_size, res_unit):
        '''
        params
            channel: number of channels
            r: the parameter mentioned in the paper [figure 2]
            size: upsampling image size list
            nb_skip: number of skip connections
            pool_size: pooling size
            res_unit: residual block
        '''
        super(SoftMaskBranch, self).__init__()
        self.r = r
        self.nb_skip = nb_skip
        self.skip_lst = []
        self.interpolation_lst = []

        self.max_pool = nn.MaxPool2d(pool_size, ceil_mode=True)
        self.residual_unit = res_unit(channel, channel)
        self.conv = nn.Conv2d(channel, channel, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # skip connection x n -> max pooling x (n+1)
        for i in range(nb_skip+1):
            self.__setattr__('interpolation%d' % i, nn.UpsamplingBilinear2d(size=size[i]))
    
    def forward(self, x):
        # downsampling
        out = x
        skip_lst = []
        for i in range(self.nb_skip):
            out = self.max_pool(out)
            for _ in range(self.r):
                out = self.residual_unit(out)
            skip_lst.append(out)

        # middle
        out = self.max_pool(out)
        for _ in range(2*self.r):
            out = self.residual_unit(out)

        # upsampling
        for i in range(self.nb_skip):
            out = self.__getattr__('interpolation%d' % (self.nb_skip-i))(out) + skip_lst[-(i+1)]
            for _ in range(self.r):
                out = self.residual_unit(out)
        out = self.__getattr__('interpolation0')(out)

        # sigmoid
        out = self.conv(out)
        out = self.conv(out)
        out = self.sigmoid(out)

        return out

class AttentionModule(nn.Module):
    def __init__(self, i_channel, o_channel, size, nb_skip, pool_size=2, p=1, t=2, r=1):
        '''
        params
            channel: number of channels
            size: upsampling image size list
            nb_skip: number of skip connections
            pool_size: pooling size
            p: the parameter mentioned in the paper [figure 2]
            t: the parameter mentioned in the paper [figure 2]
            r: the parameter mentioned in the paper [figure 2]
        '''
        super(AttentionModule, self).__init__()
        self.residual_unit_in = ResidualUnit(i_channel, o_channel)
        self.trunk_b = nn.Sequential(
            ResidualUnit(o_channel, o_channel),
            ResidualUnit(o_channel, o_channel)
        )
        self.softmask_b = SoftMaskBranch(o_channel, r, size, nb_skip, pool_size, ResidualUnit)
        self.residual_unit_out = ResidualUnit(o_channel, o_channel)
    
    def forward(self, x):
        out = self.residual_unit_in(x)
        # trunk branch
        T = self.trunk_b(out)
        # soft mask branch
        M = self.softmask_b(out)
        # calc H
        out = (1+M)*T

        out = self.residual_unit_out(out)

        return out

