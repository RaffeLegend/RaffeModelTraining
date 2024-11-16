import torch.nn as nn
from ....model_list import CHANNELS

class FCDecoder(nn.Module):
    def __init__(self, name, num_classes=1):
        super(FCDecoder, self).__init__()

        # self.preprecess will not be used during training, which is handled in Dataset class 
        self.model = nn.Linear(CHANNELS[name], num_classes)
 
    def forward(self, x):
        output = self.model(x) 
        return output