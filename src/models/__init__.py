from torch import nn
from src.models.model_zoo.encoder.clip_models import CLIPModel
from src.models.model_zoo.encoder.imagenet_models import ImagenetModel
from src.models.model_zoo.decoder.classification.fc_layer import FCDecoder

from src.models.model_list import VALID_DECODER_NAMES, VALID_ENCODER_NAMES, CHANNELS

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()

        self.opt = opt
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()     

    def forward(self, x):
        feature = self.encoder(x)["penultimate"]
        output  = self.decoder(feature)
        return output

    def get_encoder(self):
        name = self.opt.encoder
        assert name in VALID_ENCODER_NAMES
        if name.startswith("Imagenet:"):
            return ImagenetModel(name[9:]) 
        elif name.startswith("CLIP:"):
            return CLIPModel(name[5:])  
        else:
            assert False 

    def get_decoder(self):
        name = self.opt.decoder
        assert name in VALID_DECODER_NAMES
        if name.startswith("fc"):
            return FCDecoder(name)
        else:
            assert False