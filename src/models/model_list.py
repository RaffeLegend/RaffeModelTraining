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
    "ViT-L/14" : 768,
    "Imagenet:resnet50": 2048,
    "vit_b_16": 768,
}

MMSEG_ENCODER_NAMES = [
    'mmseg:ResNet', 
    'mmseg:ResNetV1c', 
    'mmseg:ResNetV1d', 
    'mmseg:ResNeXt', 
    'mmseg:HRNet', 
    'mmseg:FastSCNN',
    'mmseg:ResNeSt', 
    'mmseg:MobileNetV2', 
    'mmseg:UNet', 
    'mmseg:CGNet', 
    'mmseg:MobileNetV3',
    'mmseg:VisionTransformer', 
    'mmseg:SwinTransformer', 
    'mmseg:MixVisionTransformer',
    'mmseg:BiSeNetV1', 
    'mmseg:BiSeNetV2', 
    'mmseg:ICNet', 
    'mmseg:TIMMBackbone', 
    'mmseg:ERFNet', 
    'mmseg:PCPVT',
    'mmseg:SVT', 
    'mmseg:STDCNet', 
    'mmseg:STDCContextPathNet', 
    'mmseg:BEiT', 
    'mmseg:MAE', 
    'mmseg:PIDNet', 
    'mmseg:MSCAN',
    'mmseg:DDRNet', 
    'mmseg:VPD',
]

MMSEG_DECODER_NAMES = [
    'mmseg:FCNHead', 
    'mmseg:PSPHead', 
    'mmseg:ASPPHead', 
    'mmseg:PSAHead', 
    'mmseg:NLHead', 
    'mmseg:GCHead', 
    'mmseg:CCHead',
    'mmseg:UPerHead', 
    'mmseg:DepthwiseSeparableASPPHead', 
    'mmseg:ANNHead', 
    'mmseg:DAHead', 
    'mmseg:OCRHead',
    'mmseg:EncHead', 
    'mmseg:DepthwiseSeparableFCNHead', 
    'mmseg:FPNHead', 
    'mmseg:EMAHead', 
    'mmseg:DNLHead',
    'mmseg:PointHead', 
    'mmseg:APCHead', 
    'mmseg:DMHead', 
    'mmseg:LRASPPHead', 
    'mmseg:SETRUPHead',
    'mmseg:SETRMLAHead', 
    'mmseg:DPTHead', 
    'mmseg:SETRMLAHead', 
    'mmseg:SegmenterMaskTransformerHead',
    'mmseg:SegformerHead', 
    'mmseg:ISAHead', 
    'mmseg:STDCHead', 
    'mmseg:IterativeDecodeHead',
    'mmseg:KernelUpdateHead', 
    'mmseg:KernelUpdator', 
    'mmseg:MaskFormerHead', 
    'mmseg:Mask2FormerHead',
    'mmseg:LightHamHead', 
    'mmseg:PIDHead', 
    'mmseg:DDRHead', 
    'mmseg:VPDDepthHead', 
    'mmseg:SideAdapterCLIPHead'
]