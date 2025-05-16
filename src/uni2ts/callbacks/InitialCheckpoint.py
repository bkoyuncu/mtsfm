import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)

_PATH = Union[str, Path]

class InitialCheckpointCallback(Callback):
    """
    A PyTorch Lightning callback to save the initial model state before training begins.
    """

    def __init__(self, model_tag: str):
        """
        Args:
            filepath (Union[str, Path]): Path to save the initial model checkpoint.
        """
        self.model_tag = model_tag
        print('Initial Checkpoint ready')

    def on_train_start(self, trainer: "pl.Trainer", pl_module) -> None:
        """
        Called when the train begins. Saves the initial model state.
        """
        # Construct the filepath
        self.filepath = trainer.logger.save_dir + '/' + self.model_tag
        if not self.filepath.endswith('.ckpt'):
            self.filepath += '.ckpt'
        
        # Save the initial checkpoint
        trainer.save_checkpoint(self.filepath, weights_only=True)
        log.info(f"Initial model checkpoint saved at {self.filepath}")
