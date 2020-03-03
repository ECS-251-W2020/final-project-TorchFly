import os
import random
import logging
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from apex import amp
from typing import Dict

# pylint:disable=no-member

logger = logging.getLogger(__name__)


def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)


def send_to_device(data, devide):
    data = data.to(devide)
    return data 



class SimpleTrainer():
    def __init__(self, config, model, optimizer, dataloader):
        self.config = config
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer


        self.progress_states = {
            "global_step": 0,
            "epochs_trained": 0,
            "steps_trained_in_current_epoch": 0,
        }

    def fit(self):
        
        for _ in range( self.config.num_epochs ):
            self.train_epoch()


    def train_iter(self, batch: Dict[torch.Tensor]) -> Dict[torch.Tensor]:
        """
        """
        # set training mode
        self.model.train()
        batch = send_to_device(batch, self.config.device)

        results = self.model(batch)
        loss = results["loss"]

        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps

        # backward loss
        if self.config.distributed:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def train_epoch(self):
   
        for data in self.dataloader:
            self.train_iter(data)
