from torch import nn
from src.models.model_zoo.encoder.clip_models import CLIPModel
from src.models.model_zoo.encoder.imagenet_models import ImagenetModel
from src.models.model_zoo.decoder.classification.fc_layer import FCDecoder

from src.models.abstract import AbstractModel

from src.models.model_list import VALID_DECODER_NAMES, VALID_ENCODER_NAMES, MMSEG_ENCODER_NAMES, MMSEG_DECODER_NAMES
from external.mmsegmentation.mmseg.models.builder import BACKBONES, NECKS, HEADS

class Model(AbstractModel):
    def __init__(self, opt):
        super(Model, self).__init__()

        self.opt = opt
        self.encoder = None
        self.decoder = None
        self.build_model()

    def build_model(self):
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()     

    def forward(self, x):
        feature = self.encoder(x)
        output  = self.decoder(feature)
        return output

    def get_encoder(self):
        name = self.opt.encoder
        assert name in VALID_ENCODER_NAMES + MMSEG_ENCODER_NAMES, f"Invalid encoder name: {name}"
        if name.startswith("Imagenet:"):
            return ImagenetModel(name[9:]) 
        elif name.startswith("CLIP:"):
            return CLIPModel(name[5:])
        elif name.startswith("mmseg:"):
            return BACKBONES.build(name[6:])
        else:
            assert False, f"Unsupported encoder name: {name}"

    def get_decoder(self):
        name = self.opt.decoder
        assert name in VALID_DECODER_NAMES + MMSEG_DECODER_NAMES
        if name.startswith("fc"):
            return FCDecoder(self.opt)
        elif name.startswith("mmseg:"):
            return HEADS.build(name[6:])
        else:
            assert False, f"Unsupported decoder name: {name}"