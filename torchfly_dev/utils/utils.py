import os
import sys
import hydra
import hydra.utils
import logging
import colorlog
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def configure_logging(config: DictConfig) -> None:
    # Only setup training for node 0
    if not hasattr(config.training, "rank") or config.training.rank == 0 or config.training.rank is None:
        root = logging.getLogger()
        root.setLevel(getattr(logging, config.logging.level))
        # setup formaters
        file_formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        if config.logging.color:
            stream_formater = colorlog.ColoredFormatter(
                "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"
            )
        else:
            stream_formater = file_formatter
        # setup handlers
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(stream_formater)
        file_handler = logging.FileHandler(f"experiment.log")
        file_handler.setFormatter(file_formatter)
        # add all handlers
        root.addHandler(stream_handler)
        root.addHandler(file_handler)


def get_original_cwd(config, resume_mode) -> str:
    if resume_mode:
        os.getcwd()
    else:
        return os.getcwd()