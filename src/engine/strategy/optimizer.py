import torch

# return the type of optimizer
def get_optimizer(params, opt):
        
    if opt.optimizer == 'adam':
        optimizer = torch.optim.AdamW(params, lr=opt.learning_rate, betas=(opt.beta, 0.999), weight_decay=opt.weight_decay)
    elif opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    else:
        raise ValueError("optim should be [adam, sgd]")
    
    return optimizer