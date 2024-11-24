import cv2
import os
import pickle
import numpy as np
from io import BytesIO
from PIL import Image

from scipy.ndimage.filters import gaussian_filter
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from random import random, choice, shuffle

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

# scan the target dataset
def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out

# return the data list
def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


# return data
def get_data(opt):
    if opt.data_source == "folder":
        image, label = get_data_by_folder(opt)
    elif opt.data_source == "list":
        image, label = get_data_by_namelist(opt)
    else:
        raise ValueError("data should be loaded from folder or list")
    
    return image, label

# load data by folder origanization
def get_data_by_folder(opt):

    real_list = get_list(os.path.join(opt.dataset_path,opt.data_label), must_contain='0_real')
    fake_list = get_list(os.path.join(opt.dataset_path,opt.data_label), must_contain='1_fake')

    # setting the labels for the dataset
    labels_dict = dict()
    for i in real_list:
        labels_dict[i] = 0
    for i in fake_list:
        labels_dict[i] = 1

    total_list = real_list + fake_list
    shuffle(total_list)

    return total_list, labels_dict

# load data by list file
def get_data_by_namelist(opt):

    labels_dict = dict()
    total_list = list()
    with open(opt.datalist, 'r') as f:
        contents = f.readlines()
        for content in contents:
            content_split = content.split(",")
            image, label = content_split[0].strip(), int(content_split[1].strip())
            total_list.append(image)
            labels_dict[image] = label

    return total_list, labels_dict
            

def data_augment(img, opt):
    img = np.array(img)

    if not opt.isTrain:
        return Image.fromarray(img)
    
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}

def custom_resize(img, opt):
    # interp = sample_discrete(opt.rz_interp)
    # return TF.resize(img, opt.image_height, interpolation=rz_dict[interp])
    return TF.resize(img, (opt.image_height, opt.image_width))


def get_transformer(opt):
    # if opt.isTrain:
    #     crop_func = transforms.RandomCrop(opt.cropSize)
    # elif opt.no_crop:
    #     crop_func = transforms.Lambda(lambda img: img)
    # else:
    #     crop_func = transforms.CenterCrop(opt.cropSize)

    # if opt.isTrain and not opt.no_flip:
    #     flip_func = transforms.RandomHorizontalFlip()
    # else:
    #     flip_func = transforms.Lambda(lambda img: img)
    # if not opt.isTrain and opt.no_resize:
    #     rz_func = transforms.Lambda(lambda img: img)
    # else:
    rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        
    stat_from = "imagenet" if opt.encoder.lower().startswith("imagenet") else "clip"

    print("mean and std stats are from: ", stat_from)
    
    # to do ......
    # if '2b' not in opt.arch:
    print ("using Official CLIP's normalization")
    transform = transforms.Compose([
            rz_func,
            transforms.Lambda(lambda img: data_augment(img, opt)),
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])
    # else:
    #     print ("Using CLIP 2B transform")
    #     # to do
    #     transform = None # will be initialized in trainer.py

    return transform