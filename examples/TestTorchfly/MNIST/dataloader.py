import os
import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
from torchvision import datasets, transforms

def get_cwd():
    try:
        return get_original_cwd()
    except AttributeError:
        return os.getcwd()

class WarpDataloader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
  
    def __iter__(self):
        for batch in super().__iter__():
            batch = {
                "input": batch[0],
                "target": batch[1]
            }
            yield batch
    
    def __next__(self):
        return next(self.__iter__())

def get_data_loader(config):
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = WarpDataloader(
        datasets.MNIST(os.path.join(get_cwd(), 'data'), train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=config.training.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(os.path.join(get_cwd(), 'data'), train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=config.training.batch_size, shuffle=True, **kwargs)


    return train_loader, test_loader