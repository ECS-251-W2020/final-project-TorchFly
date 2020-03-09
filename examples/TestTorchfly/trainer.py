import os
import sys
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
import torch.multiprocessing as multiprocessing
import torch.distributed
# from apex.parallel import DistributedDataParallel, Reducer
from torch.nn.parallel import DistributedDataParallel
from hydra.utils import get_original_cwd

from torchfly_dev.training.optimization import ConstantLRSchedule
from torchfly_dev.utils import move_to_device, configure_logging

from model import FlyModule
from checkpointer import Checkpointer

logger = logging.getLogger(__name__)


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


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        model: FlyModule = None,
        train_loader: Iterator = None,
        validation_loader: Iterator = None,
        test_loader: Iterator = None
    ):
        """
        Do not send anything to cuda in __init__
        """
        self.config = config
        self.moving_average = MovingAverage()
        # setup saving directory
        if self.config.saving.resume_mode:
            self.config.saving.save_dir = get_original_cwd()
        else:
            self.config.saving.save_dir = os.getcwd()
        self.setup_configuration()

        # # Data Loading
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        # local variables
        self.__global_count = 0
        self.__epochs_trained = 0
        self.__total_num_epochs = config.training.total_num_epochs

        # setup model
        if model is None:
            self.model = self.configure_model()
        else:
            self.model = model

        # # Checkpointer
        self.checkpointer = Checkpointer(
            sync_every_save=True,
            num_checkpoints_to_keep=config.saving.num_checkpoints_to_keep,
            keep_checkpoint_every_num_seconds=config.saving.keep_checkpoint_every_num_seconds,
            storage_dir=os.path.join(self.config.saving.save_dir, "Checkpoints")
        )

        # # Resume the training
        # # find the any checkpoint
        if self.config.saving.resume_mode:
            self.__restore_checkpoint()

    def setup_configuration(self):
        # if self.config.training.validation_interval is None:
        #     # save for every epoch
        #     self.config.training.validation_interval = len(self.train_loader) - 1

        # Logging
        self.__log_in_seconds = False
        if self.config.logging.iterations_interval <= 0:
            if self.config.logging.seconds_interval is None or self.config.logging.seconds_interval < 0:
                # default log_iterations_interval
                self.config.logging.iterations_interval = 10
            else:
                self.__log_in_seconds = True

        # Saving
        self.__save_in_seconds = False
        # if nothign about saving interval is set
        if (self.config.saving.iterations_interval is None and self.config.saving.seconds_interval is None
           ) or (self.config.saving.iterations_interval < 0 and self.config.saving.seconds_interval < 0):
            # then save for every epoch
            self.config.saving.iterations_interval = len(self.train_loader) - 1

        if self.config.saving.iterations_interval < 0 and self.config.saving.seconds_interval > 0:
            self.__save_in_seconds = True

        logger.info(self.config.pretty())

    def __single_gpu_train(self):
        self.__set_random_seed(rank + self.config.training.random_seed)
        self.__device = torch.device("cuda")

    def _distributed_train(self, rank):
        """
        Handles the distributed training on a single node.

        Args:
            rank: also the gpu index
        """
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        self.__rank = rank
        self.__master = rank == 0
        torch.cuda.set_device(rank)
        self.__device = torch.device("cuda", self.__rank)
        self.config.training.rank = rank

        # setup the scheduler
        self.optimizer = self.configure_optimizer()

        if self.__master:
            configure_logging(self.config)
            # logger.info(ray.init())
            self.__tensorboard = SummaryWriter(log_dir=os.getcwd())

        # init distributed
        torch.distributed.init_process_group(
            backend="nccl", rank=self.__rank, world_size=self.config.training.num_gpus_per_node
        )

        # Reproducibility
        if self.config.training.random_seed:
            self.__set_random_seed(rank + self.config.training.random_seed)

        # To Device
        self.model = move_to_device(self.model, self.__device)

        # FP16
        if self.config.training.fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.config.training.fp16_opt_level
            )

        # Distributed training (should be after apex fp16 initialization)
        # self.model = DistributedDataParallel(self.model)
        self.model = DistributedDataParallel(
            self.model, device_ids=[self.__rank], output_device=self.__rank, find_unused_parameters=True
        )
        # setup the cheduler
        self.scheduler = self.configure_scheduler()

        torch.distributed.barrier()

        if self.__master:
            logger.info("Starting Training.")
            if self.__log_in_seconds:
                self.__log_iter_start_time = time.time()
            if self.__save_in_seconds:
                self.__save_iter_start_time = time.time()

        # training loop
        for epoch in range(self.__epochs_trained, self.__total_num_epochs):
            if self.__master:
                epoch_start_time = time.time()

            epoch_results = self.__train_epoch(epoch)

            if self.__master:
                epoch_elapsed_time = time.time() - epoch_start_time
                logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            self.__epochs_trained += 1

        if self.__master:
            self.__tensorboard.close()

        return None

    def train(self) -> Dict[str, Any]:
        if self.config.training.num_gpus_per_node > 1:
            logger.info("Initializing Distributed Training")

            if 'OMP_NUM_THREADS' not in os.environ and self.config.training.num_gpus_per_node > 1:
                os.environ["OMP_NUM_THREADS"] = str(1)
                logger.info(
                    "*****************************************\n"
                    "Setting OMP_NUM_THREADS environment variable for each process "
                    "to be {} in default, to avoid your system being overloaded, "
                    "please further tune the variable for optimal performance in "
                    "your application as needed. \n"
                    "*****************************************".format(os.environ["OMP_NUM_THREADS"])
                )

                os.environ["MASTER_ADDR"] = "localhost"
                os.environ["MASTER_PORT"] = str(random.randint(20000, 29000))  # use a random port, but might collide
                os.environ["WORLD_SIZE"] = str(self.config.training.num_gpus_per_node)

                multiprocessing.log_to_stderr()
                multiprocessing.spawn(self._distributed_train, args=(), nprocs=self.config.training.num_gpus_per_node)
        elif self.config.training.num_gpus_per_node == 1:
            self.__single_gpu_train()
        else:
            raise NotImplementedError("Do you mean CPU training?")

        results = {}
        return results

    def __train_iter(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
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

    def __train_epoch(self, epoch):
        self.model.train()

        if self.__master:
            logger.info("Epoch %d/%d", epoch + 1, self.__total_num_epochs)

        for batch_idx, batch in enumerate(self.train_loader):
            batch = move_to_device(batch, self.__device)
            results = self.__train_iter(batch)

            # Update the model
            if self.__global_count % self.config.training.gradient_accumulation_steps == (
                self.config.training.gradient_accumulation_steps - 1
            ):
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Checkpointing
            if self.__master:
                if self.__save_in_seconds:
                    current_time = time.time()
                    iter_elapsed_time = current_time - self.__save_iter_start_time

                    if iter_elapsed_time > self.config.saving.seconds_interval:
                        self.__save_checkpoint()
                        self.__save_iter_start_time = current_time
                else:
                    if self.__global_count % self.config.saving.iterations_interval == (
                        self.config.saving.iterations_interval - 1
                    ):
                        self.__save_checkpoint()

            # Logging
            if self.__master:
                if self.__log_in_seconds:
                    current_time = time.time()
                    iter_elapsed_time = current_time - self.__log_iter_start_time

                    if iter_elapsed_time > self.config.logging.seconds_interval:
                        self.__log_iteration(batch_idx, results)
                        self.__log_iter_start_time = current_time
                else:
                    if self.__global_count % self.config.logging.iterations_interval == (
                        self.config.logging.iterations_interval - 1
                    ):
                        self.__log_iteration(batch_idx, results)

            self.__global_count += 1

            # Validation
            # if self._master:
            #     if self.validation_loader is not None and  and self._global_count % self.config.training.validation_interval == (
            #         self.config.training.validation_interval - 1
            #     ):
            #         self.validate()

        return 0

    def validate(self):
        NotImplementedError

    def configure_optimizer(self) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.training.learning_rate)

    def configure_scheduler(self) -> LambdaLR:
        return ConstantLRSchedule(self.optimizer)

    def configure_model(self) -> FlyModule:
        raise NotImplementedError

    def __save_checkpoint(self) -> None:
        pass
        # states = {
        #     "epoch": self.__epochs_trained,
        #     "iteration": self.__global_count,
        #     "model_states": self.model.state_dict(),
        #     "optimizer_states": self.optimizer.state_dict(),
        #     "scheduler_states": self.scheduler.state_dict(),
        #     "checkpointer_states": self.checkpointer.state_dict(),
        # }
        # # save amp states
        # if self.config.training.fp16:
        #     states["amp"] = amp.state_dict()
        # # use the current iteration number as the stamp
        # self.checkpointer.save_checkpoint("iter_" + str(self.__global_count), states)

    def __restore_checkpoint(self):
        states = self.checkpointer.restore_latest_checkpoint()
        if states is not None:
            self.__epochs_trained = states["epoch"]
            self.__global_count = states["iteration"]
            self.model.load_state_dict(states["model_states"])
            self.optimizer.load_state_dict(states["optimizer_states"])
            self.scheduler.load_state_dict(states["scheduler_states"])
            self.checkpointer.load_state_dict(states["checkpointer_states"])
            # restore amp states
            if self.config.training.fp16:
                amp.load_state_dict(states["amp"])

            logger.info("Loaded the latest checkpoint")
        else:
            logger.info("No checkpoint found. Training from scratch!")

    def __set_random_seed(self, random_seed):
        # Reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    def __log_iteration(self, batch_idx, results: Dict[str, torch.Tensor]):
        """
        Args:
           batch_idx: 
           results:
        """
        _loss = results['loss'].item()
        avg_loss = self.moving_average.update_key("loss", _loss)
        percent = 100. * batch_idx / len(self.train_loader)
        logger.info(
            f"Train Epoch: {self.__epochs_trained+1} \
                [{self.__epochs_trained+1}/{self.__total_num_epochs} ({percent:.2f}%)]\tLoss: {avg_loss:.6f}"
        )

        if self.__tensorboard:
            self.__tensorboard.add_scalar("loss", _loss, self.__global_count + 1)


def set_random_port(self):
    """
    When running DDP NOT managed by SLURM, the ports might collide
    :return:
    """
    try:
        default_port = os.environ['MASTER_PORT']
    except Exception:
        import random
        default_port = random.randint(20000, 29000)
        os.environ['MASTER_PORT'] = str(default_port)
