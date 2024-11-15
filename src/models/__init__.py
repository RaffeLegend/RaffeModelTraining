from src.models.model_zoo.decoder.clip_models import CLIPModel
from src.models.model_zoo.decoder.imagenet_models import ImagenetModel
from models.model_zoo.encoder.classification.fc_layer import FCDecoder

from model_list import VALID_DECODER_NAMES, VALID_ENCODER_NAMES, CHANNELS


def get_encoder(name):
    assert name in VALID_ENCODER_NAMES
    if name.startswith("Imagenet:"):
        return ImagenetModel(name[9:]) 
    elif name.startswith("CLIP:"):
        return CLIPModel(name[5:])  
    else:
        assert False 

def get_decoder(name):
    assert name in VALID_DECODER_NAMES
    if name.startswith("fc"):
        return FCDecoder(name)
    else:
        assert False