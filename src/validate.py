import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.options import config_settings
from src.data import create_dataloader
from src.engine.trainer import Trainer
from src.engine.validator import Validator
from src.utils.tools import log_validation_metrics
import torch

# Assuming you have a function to load your model and data

def evaluate_model():
    # Load the model
    opt = config_settings()

    trainer = Trainer(opt.train)
    data_loader = create_dataloader(opt.test_data)

    validator = Validator(opt.val)
    validator.update(trainer.model, data_loader)
    net_state_dict = torch.load(opt.test.model_path, map_location='cuda:0')
    validator.model.load_state_dict(net_state_dict['model'])

    validator.model.eval()

    validator.forward()
    ap, r_acc, f_acc, acc = validator.evaluate()

    log_validation_metrics(0, acc, ap)

if __name__ == "__main__":
    
    evaluate_model()