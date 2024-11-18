import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from src.utils.tools import mkdirs


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device('cuda:{}'.format(opt.gpus[0])) if opt.gpus else torch.device('cpu')

    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, self.opt.model_name)
        mkdirs(save_path)

        # serialize model and optimizer to dict
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'total_steps' : self.total_steps,
        }

        torch.save(state_dict, os.path.join(save_path, save_filename))


    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()