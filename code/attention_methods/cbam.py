import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio, pool_types):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels//reduction_ratio, in_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None 
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool.view(avg_pool.size(0), -1))
            elif pool_type=='max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool.view(max_pool.size(0), -1))

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        channel_att = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * channel_att


class SpatialAttention(nn.Module):
    def __init__(self, k_size):
        super(SpatialAttention, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=k_size, padding=(k_size - 1)//2),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
    
    def forward(self, x):
        conv_concat = torch.cat((x.max(1)[0].unsqueeze(1), x.mean(1).unsqueeze(1)), dim=1)
        spatial_att_raw = self.conv_block(conv_concat)
        spatial_att = torch.sigmoid(spatial_att_raw).expand_as(x)
        return x * spatial_att


class CBAM(nn.Module):
    def __init__(self, in_channels, pool_types=['avg','max'], k_size=7, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.CA = ChannelAttention(in_channels, reduction_ratio, pool_types)
        self.SA = SpatialAttention(k_size)
        
    def forward(self, x):
        channel_att = self.CA(x)
        spatial_att = self.SA(channel_att)

        return spatial_att