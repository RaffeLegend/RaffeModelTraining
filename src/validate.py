import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.options import config_settings
from src.data import create_dataloader
from src.engine.validator import Validator
from src.utils.tools import log_validation_metrics
import torch

# Assuming you have a function to load your model and data

def evaluate_model():
    # Load the model
    opt = config_settings()

    data_loader = create_dataloader(opt.test_data)

    model = torch.load(opt.validate.model_path)
    model.eval()

    validator = Validator(opt.val)
    validator.update(model, data_loader)

    validator.forward()
    ap, r_acc, f_acc, acc = validator.evaluate()

    log_validation_metrics(0, acc, ap)

if __name__ == "__main__":
    
    evaluate_model()