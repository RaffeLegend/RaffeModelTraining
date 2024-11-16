import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_sampler) else False

    if opt.task == "classification":
        from classification.dataset import RealFakeDataset
    elif opt.task == "segmentation":
        from segmentation.dataset import RealFakeDataset

    dataset = RealFakeDataset(opt)

    sampler = get_bal_sampler(dataset) if opt.class_sampler else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader