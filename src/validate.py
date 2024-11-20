import os
import sys
from src.options import config_settings
from src.data import create_dataloader
from src.engine.validator import Validator
from src.utils.tools import log_validation_metrics
import torch

# Assuming you have a function to load your model and data

def evaluate_model(model_path, data_path):
    # Load the model
    opt = config_settings()
    opt.test_data.dataset_path = data_path
    opt.validate.model_path = model_path
    
    data_loader = create_dataloader(opt.test_data)

    model = torch.load(opt.validate.model_path)
    model.eval()

    validator = Validator(opt.val)
    validator.update(model, data_loader)

    validator.forward()
    ap, r_acc, f_acc, acc = validator.evaluate()

    log_validation_metrics(0, acc, ap)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validation.py <model_path> <data_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    
    evaluate_model(model_path, data_path)