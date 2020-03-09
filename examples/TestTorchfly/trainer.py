import os
import ray
import math
import time
import datetime
import logging
from typing import Any, List, Dict, Iterator
from omegaconf import DictConfig
from apex import amp
import numpy as np
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from hydra.utils import get_original_cwd

from torchfly_dev.training.optimization import ConstantLRSchedule
from torchfly_dev.utils import move_to_device

from model import FlyModule
from checkpointer import Checkpointer

logger = logging.getLogger(__name__)


def get_cwd(resume_mode):
    if resume_mode:
        try:
            result = get_original_cwd()
        except:
            logger.info("hydra is not used in resume mode. Using `os.getcwd` instead")
            result = os.getcwd()
        return result
    else:
        return os.getcwd()


class MovingAverage:
    def __init__(self, init_dict=None, decay=0.9):
        self.history_dict = init_dict if init_dict else {}
        self.decay = decay

    def update_key(self, key, value):
        if key in self.history_dict:
            self.history_dict[key] = self.decay * self.history_dict[key] + (1 - self.decay) * value
        else:
            self.history_dict[key] = value
        return self.history_dict[key]


logger.info(ray.init())


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        model: FlyModule = None,
        train_loader: Iterator = None,
        validation_loader: Iterator = None,
        test_loader: Iterator = None
    ):
        self.config = config
        self.device = torch.device("cuda")
        self.moving_average = MovingAverage()

        # Data Loading
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        # Distributed Training
        config.training.rank = -1

        if self.config.training.validation_interval is None:
            # save for every epoch
            self.config.training.validation_interval = len(self.train_loader) - 1

        # set up model
        if model is None:
            self.model = self.configure_model()
        else:
            self.model = model
        # move to device
        self.model = move_to_device(self.model, self.device)
        self.optimizer = self.configure_optimizer()

        # local
        self._master = config.training.rank <= 0
        self._total_num_epochs = config.training.total_num_epochs
        self._tensorboard = SummaryWriter(log_dir=os.getcwd())

        # local variables
        self._global_count = 0
        self._epochs_trained = 0

        if config.training.fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=config.training.fp16_opt_level
            )

        # set up the cheduler
        self.scheduler = self.configure_scheduler()

        # Logging
        self._log_in_seconds = False
        if self.config.training.log_iterations_interval <= 0:
            if self.config.training.log_seconds_interval is None or self.config.training.log_seconds_interval < 0:
                # default log_iterations_interval
                self.config.training.log_iterations_interval = 10
            else:
                self._log_in_seconds = True

        # Checkpointer
        self._save_in_seconds = False

        self.checkpointer = Checkpointer(
            sync_every_save=True,
            num_checkpoints_to_keep=config.training.num_checkpoints_to_keep,
            keep_checkpoint_every_num_seconds=config.training.keep_checkpoint_every_num_seconds,
            storage_dir=os.path.join(get_cwd(config.training.resume_mode), "Checkpoints")
        )

        # if nothign about saving interval is set
        if (
            self.config.training.save_iterations_interval is None and self.config.training.save_seconds_interval is None
        ) or (self.config.training.save_iterations_interval < 0 and self.config.training.save_seconds_interval < 0):
            # then save for every epoch
            self.config.training.save_iterations_interval = len(self.train_loader) - 1

        # if self.config.training.save_seconds_interval is set
        if self.config.training.save_iterations_interval < 0 and self.config.training.save_seconds_interval > 0:
            self._save_in_seconds = True

        # Resume the training
        # find the any checkpoint
        if self.config.training.resume_mode:
            self._restore_checkpoint()

        if self._master:
            logger.info(ray.init())

    def train(self) -> Dict[str, Any]:
        logger.info("Starting Training.")
        if self._master:
            if self._log_in_seconds:
                self._log_iter_start_time = time.time()
            if self._save_in_seconds:
                self._save_iter_start_time = time.time()

        # training loop
        for epoch in range(self._epochs_trained, self._total_num_epochs):
            epoch_start_time = time.time()
            epoch_results = self._train_epoch(epoch)

            epoch_elapsed_time = time.time() - epoch_start_time

            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            self._epochs_trained += 1

        # make sure pending events are flushed to disk and files are closed properly
        self._tensorboard.close()

        # test stage
        if self.test_loader:
            pass

        # return training results
        results = {}
        return results

    def _train_epoch(self, epoch):
        self.model.train()

        if self._master:
            logger.info("Epoch %d/%d", epoch + 1, self._total_num_epochs)
            logger.info("Training")

        for batch_idx, batch in enumerate(self.train_loader):
            batch = move_to_device(batch, self.device)
            results = self._train_iter(batch)

            # Update the model
            if self._global_count % self.config.training.gradient_accumulation_steps == (
                self.config.training.gradient_accumulation_steps - 1
            ):
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Checkpointing
            if self._master:
                if self._save_in_seconds:
                    current_time = time.time()
                    iter_elapsed_time = current_time - self._save_iter_start_time

                    if iter_elapsed_time > self.config.training.save_seconds_interval:
                        self._save_checkpoint()
                        self._save_iter_start_time = current_time
                else:
                    if self._global_count % self.config.training.save_iterations_interval == (
                        self.config.training.save_iterations_interval - 1
                    ):
                        self._save_checkpoint()

            # Logging
            if self._master:
                if self._log_in_seconds:
                    current_time = time.time()
                    iter_elapsed_time = current_time - self._log_iter_start_time

                    if iter_elapsed_time > self.config.training.log_seconds_interval:
                        self._log_iteration(batch_idx, results)
                        self._log_iter_start_time = current_time
                else:
                    if self._global_count % self.config.training.log_iterations_interval == (
                        self.config.training.log_iterations_interval - 1
                    ):
                        self._log_iteration(batch_idx, results)
            self._global_count += 1

            # Validation
            # if self._master:
            #     if self.validation_loader is not None and  and self._global_count % self.config.training.validation_interval == (
            #         self.config.training.validation_interval - 1
            #     ):
            #         self.validate()

        return 0

    def _train_iter(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        results = self.model(batch)
        loss = results["loss"]

        if self.config.training.gradient_accumulation_steps > 1:
            loss = loss / self.config.training.gradient_accumulation_steps

        if self.config.training.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return results

    def validate(self):
        NotImplementedError

    def configure_optimizer(self) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.training.learning_rate)

    def configure_scheduler(self) -> LambdaLR:
        return ConstantLRSchedule(self.optimizer)

    def configure_model(self) -> FlyModule:
        raise NotImplementedError

    def _save_checkpoint(self) -> None:
        states = {
            "epoch": self._epochs_trained,
            "iteration": self._global_count,
            "model_states": self.model.state_dict(),
            "optimizer_states": self.optimizer.state_dict(),
            "scheduler_states": self.scheduler.state_dict(),
            "checkpointer_states": self.checkpointer.state_dict(),
        }
        # use the current iteration number as the stamp
        self.checkpointer.save_checkpoint("iter_" + str(self._global_count), states)

    def _restore_checkpoint(self):
        states = self.checkpointer.restore_latest_checkpoint()
        if states is not None:
            self._epochs_trained = states["epoch"]
            self._global_count = states["iteration"]
            self.model.load_state_dict(states["model_states"])
            self.optimizer.load_state_dict(states["optimizer_states"])
            self.scheduler.load_state_dict(states["scheduler_states"])
            self.checkpointer.load_state_dict(states["checkpointer_states"])
            logger.info("Loaded the latest checkpoint")
        else:
            logger.info("No checkpoint found. Training from scratch!")

    def _set_random_seed(self, random_seed):
        # Reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    def _log_iteration(self, batch_idx, results: Dict[str, torch.Tensor]):
        """
        Args:
           batch_idx: 
           results:
        """
        _loss = results['loss'].item()
        avg_loss = self.moving_average.update_key("loss", _loss)
        percent = 100. * batch_idx / len(self.train_loader)
        logger.info(
            f"Train Epoch: {self._epochs_trained+1} [{self._epochs_trained+1}/{self._total_num_epochs} ({percent:.2f}%)]\tLoss: {avg_loss:.6f}"
        )

        if self._tensorboard:
            self._tensorboard.add_scalar("loss", _loss, self._global_count + 1)