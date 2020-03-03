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


class InitManager:
    """ Set Up Distributed Training
        Should be Distributed Agnostic (Works both distributed and non-distributed)
    """
    def __init__(self, config):
        self.config = config
        self.rank = config.local_rank
        self.main_rank = 0

        # fp16 handle
        amp.register_float_function(torch, 'multinomial')
        amp.register_float_function(torch, 'softmax')

        if config.local_rank != -1:
            torch.cuda.set_device(config.local_rank)
            device = torch.device("cuda", config.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            self.config.device = device
            # get the world size
            self.config.n_gpu = torch.distributed.get_world_size()
        else:
            # no distributed traing
            # set default cuda device
            self.config.device = torch.device("cuda")
            self.config.n_gpu = 1

    def init_training(self, models, optimizers, num_losses=1):
        """
        num_losses: useful if multiple losses, e.g. GAN training
        """
        # send to device
        if isinstance(models, list):
            models = [model.to(self.config.device) for model in models]
        else:
            models = models.to(self.config.device)

        if self.config.fp16:
            models, optimizers = amp.initialize(
                models, optimizers, opt_level=self.config.fp16_opt_level, num_losses=num_losses
            )

        # Distributed training (should be after apex fp16 initialization)
        if self.rank != -1:
            if isinstance(models, list):
                models = [
                    DistributedDataParallel(
                        model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True
                    ) for model in models
                ]
            else:
                models = DistributedDataParallel(
                    models, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True
                )

        # send to device

        logger.warning(
            f"Process rank: {self.config.local_rank}, "
            f"device: {self.config.device}, "
            f"n_gpu: {self.config.num_gpus}, "
            f"distributed training: {self.config.local_rank != -1}, "
            f"16-bits training: {self.config.fp16}"
        )

        return models, optimizers

    def is_main_rank(self):
        if self.rank in [-1, self.main_rank]:
            return True
        else:
            return False

    def backward_loss(self, loss, model, optimizer, loss_id=0):
        if self.config.fp16:
            with amp.scale_loss(loss, optimizer, loss_id=loss_id) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def clip_grad_norm(self, model, optimizer):
        if self.config.max_grad_norm > 0.0:
            if self.config.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.config.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)


class SimpleTrainer:
    def __init__(self, config, model, optimizer):
        self.config = config

        model, optimizer = amp.initialize(model, optimizer, opt_level=config.fp16_opt_level)
        # scheduler =

        self.progress_states = {
            "global_step": 0,
            "epochs_trained": 0,
            "steps_trained_in_current_epoch": 0,
        }

    def fit(self):
        pass

    def train_iter(self, batch: Dict[torch.Tensor]) -> Dict[torch.Tensor]:
        """
        """
        # set training mode
        model.train()
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

        pass
