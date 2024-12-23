import torch.nn as nn
from src.engine.base_trainer import BaseModel
from src.engine.strategy.weights_init import init_weights
from src.engine.strategy.optimizer import get_optimizer
from models.model_imp.base_model import Model

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt  
        self.model = Model(opt)

        if opt.fix_backbone:
            params = self.model.decoder.parameters()
        else:
            params = self.model.parameters()

        self.optimizer = get_optimizer(params, opt)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.model.to(opt.gpus[0])


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True


    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)


    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label) 
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()