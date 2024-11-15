from .clips import clip 
import torch.nn as nn

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        # self.preprecess will not be used during training, which is handled in Dataset class 
        self.model, self.preprocess = clip.load(name, device="cpu") 
 
    def forward(self, x):
        features = self.model.encode_image(x) 
        return features