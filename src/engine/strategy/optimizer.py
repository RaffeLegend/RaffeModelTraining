import torch

# return the type of optimizer
def get_optimizer(params, opt):
        
    if opt.optim == 'adam':
        optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    else:
        raise ValueError("optim should be [adam, sgd]")
    
    return optimizer