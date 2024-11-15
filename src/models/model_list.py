VALID_ENCODER_NAMES = [
    'Imagenet:resnet18',
    'Imagenet:resnet34',
    'Imagenet:resnet50',
    'Imagenet:resnet101',
    'Imagenet:resnet152',
    'Imagenet:vgg11',
    'Imagenet:vgg19',
    'Imagenet:swin-b',
    'Imagenet:swin-s',
    'Imagenet:swin-t',
    'Imagenet:vit_b_16',
    'Imagenet:vit_b_32',
    'Imagenet:vit_l_16',
    'Imagenet:vit_l_32',

    'CLIP:RN50', 
    'CLIP:RN101', 
    'CLIP:RN50x4', 
    'CLIP:RN50x16', 
    'CLIP:RN50x64', 
    'CLIP:ViT-B/32', 
    'CLIP:ViT-B/16', 
    'CLIP:ViT-L/14', 
    'CLIP:ViT-L/14@336px',
]

VALID_DECODER_NAMES = [
    'fc',
]

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}