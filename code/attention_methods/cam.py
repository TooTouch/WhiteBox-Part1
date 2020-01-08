import torch.nn as nn
import torch.nn.functional as F

class CAM(nn.Module):
    def __init__(self, in_channels, nb_class):
        super(CAM, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, nb_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        GAP = F.avg_pool2d(x, (x.size(2), x.size(3)))
        output = self.mlp(GAP.view(GAP.size(0), -1))
        
        return output