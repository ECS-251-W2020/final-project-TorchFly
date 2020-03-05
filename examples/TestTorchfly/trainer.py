import os
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

from torchfly_dev.training.optimization import ConstantLRSchedule
from torchfly_dev.utils import move_to_device
from model import FlyModule

logger = logging.getLogger(__name__)


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

        # Reproducibility
        if hasattr(self.config.training, "random_seed") and self.config.training.random_seed:
            random.seed(self.config.training.random_seed)
            np.random.seed(self.config.training.random_seed)
            torch.manual_seed(self.config.training.random_seed)
            torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

        # Data Loading
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        # Distributed Training
        config.training.rank = -1

        # Checkpointer
        if self.config.training.save_checkpoint_interval is None:
            # save for every epoch
            self.config.training.save_checkpoint_interval = len(self.train_loader) - 1

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
        self.scheduler = self.configure_scheduler()

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

        # Logging
        self._log_in_seconds = False
        if self.config.training.log_iterations_interval <= 0:
            if self.config.training.log_seconds_interval is None:
                # default log_iterations_interval
                self.config.training.log_iterations_interval = 10
            else:
                self._log_in_seconds = True

    def train(self) -> Dict[str, Any]:
        logger.info("Starting Training.")

        # Resume previous training
        # try:
        #     self.epochs_trained = 0
        # except:
        #     logger.info("Could not recover training")
        #     pass

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

        if self._master and self._log_in_seconds:
            iter_start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = move_to_device(batch, self.device)
            results = self._train_iter(batch)

            # update the model
            if self._global_count % self.config.training.gradient_accumulation_steps == (
                self.config.training.gradient_accumulation_steps - 1
            ):
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if self._master and self._global_count % self.config.training.save_checkpoint_interval == (
                self.config.training.save_checkpoint_interval - 1
            ):
                self._save_checkpoint()

            if self.validation_loader is not None and self._master and self._global_count % self.config.training.validation_interval == (
                self.config.training.validation_interval - 1
            ):
                self.validate()

            # Logging
            if self._master:
                if self._log_in_seconds:
                    current_time = time.time()
                    iter_elapsed_time = current_time - iter_start_time

                    if iter_elapsed_time > self.config.training.log_seconds_interval:
                        self._log_iteration(batch_idx, results)
                        iter_start_time = current_time
                else:
                    if self._global_count % self.config.training.log_iterations_interval == (
                        self.config.training.log_iterations_interval - 1
                    ):
                        self._log_iteration(batch_idx, results)
            self._global_count += 1
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

    def _restore_checkpoint(self):
        pass

    def _save_checkpoint(self) -> None:
        pass

    def _log_iteration(self, batch_idx, results):
        _loss = results['loss'].item()
        percent = 100. * batch_idx / len(self.train_loader)
        logger.info(
            f"Train Epoch: {self._epochs_trained+1} [{self._epochs_trained+1}/{self._total_num_epochs} ({percent:.2f}%)]\tLoss: {_loss:.6f}"
        )

        if self._tensorboard:
            self._tensorboard.add_scalar("loss", _loss, self._global_count + 1)