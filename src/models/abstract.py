from abc import ABC, abstractmethod
import torch.nn as nn

class AbstractModel(nn.Module, ABC):
            
    def __init__(self, opt):
        super(AbstractModel, self).__init__()
        self.opt = opt
    
    @abstractmethod
    def build_model(self):
        pass
    
    @abstractmethod
    def get_encoder(self):
        pass

    @abstractmethod
    def get_decoder(self):
        pass


