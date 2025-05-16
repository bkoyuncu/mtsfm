#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import sys


# Remove existing uni2ts paths if they exist
sys.path = [p for p in sys.path if 'uni2ts' not in p]

# Add your custom paths with higher priority
# sys.path.insert(0, #TODO)
# sys.path.insert(0, #TODO)


print(sys.path)
from functools import partial
from typing import Callable, Optional
import argparse

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils._pytree import tree_map
from torch.utils.data import Dataset, DistributedSampler

# Import the resolver from uni2ts.common
from uni2ts.common import hydra_util  # noqa: hydra resolvers
from uni2ts.data.loader import DataLoader


def setup_environment(cache_dir=None, tmp_dir=None):
    """Configure environment variables for training."""
    # Set WORLD_SIZE for single-node setup by default
    os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
    
    # Set cache and temporary directories if provided
    if tmp_dir:
        os.environ['TMPDIR'] = tmp_dir
        print(f"Using temporary directory: {tmp_dir}")
    
    if cache_dir:
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        os.environ['HF_DATASETS_CACHE'] = cache_dir
        os.environ['TORCH_HOME'] = cache_dir
        print(f"Using cache directory: {cache_dir}")
    
    # Print CUDA information
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    
    # Uncomment to display detailed GPU information if needed
    # for i in range(torch.cuda.device_count()):
    #     gpu = torch.cuda.get_device_properties(i)
    #     print(f"GPU {i}: {gpu.name} - {gpu.total_memory/1024**3:.2f} GB")
    # print(f"Current device: {torch.cuda.current_device()}")


class DataModule(L.LightningDataModule):
    """Lightning DataModule to handle data loading for training and validation."""
    
    def __init__(
        self,
        cfg: DictConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset | list[Dataset]],
    ):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = train_dataset

        if val_dataset is not None:
            self.val_dataset = val_dataset
            self.val_dataloader = self._val_dataloader

    @staticmethod
    def get_dataloader(
        dataset: Dataset,
        dataloader_func: Callable[..., DataLoader],
        shuffle: bool,
        world_size: int,
        batch_size: int,
        num_batches_per_epoch: Optional[int] = None,
    ) -> DataLoader:
        """Create a dataloader with appropriate sampler based on distributed training setup."""
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=None,
                rank=None,
                shuffle=shuffle,
                seed=0,
                drop_last=False,
            )
            if world_size > 1
            else None
        )
        return dataloader_func(
            dataset=dataset,
            shuffle=shuffle if sampler is None else None,
            sampler=sampler,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader with appropriate batch size and transforms."""
        return self.get_dataloader(
            self.train_dataset,
            instantiate(self.cfg.train_dataloader, _partial_=True),
            self.cfg.train_dataloader.shuffle,
            self.trainer.world_size,
            self.train_batch_size,
            num_batches_per_epoch=self.train_num_batches_per_epoch,
        )

    def _val_dataloader(self) -> DataLoader | list[DataLoader]:
        """Return validation dataloader(s) with appropriate batch size and transforms."""
        return tree_map(
            partial(
                self.get_dataloader,
                dataloader_func=instantiate(self.cfg.val_dataloader, _partial_=True),
                shuffle=self.cfg.val_dataloader.shuffle,
                world_size=self.trainer.world_size,
                batch_size=self.val_batch_size,
                num_batches_per_epoch=None,
            ),
            self.val_dataset,
        )

    @property
    def train_batch_size(self) -> int:
        """Calculate effective training batch size accounting for distributed training and gradient accumulation."""
        return self.cfg.train_dataloader.batch_size // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )

    @property
    def val_batch_size(self) -> int:
        """Calculate effective validation batch size accounting for distributed training and gradient accumulation."""
        return self.cfg.val_dataloader.batch_size // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )

    @property
    def train_num_batches_per_epoch(self) -> int:
        """Calculate number of training batches per epoch accounting for gradient accumulation."""
        return (
            self.cfg.train_dataloader.num_batches_per_epoch
            * self.trainer.accumulate_grad_batches
        )


def train_model(cfg: DictConfig):
    """Main training function that sets up and runs the training process."""
    # Set the environment variable to use specific GPU devices if specified
    if hasattr(cfg, "cuda_device") and cfg.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_device)
        print(f"Using CUDA devices: {cfg.cuda_device}")

    # Configure TF32 precision if specified
    if cfg.get("tf32", False):
        assert cfg.trainer.precision == 32, "TF32 requires precision to be set to 32"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Using TF32 precision")

    # Instantiate the model
    model: L.LightningModule = instantiate(cfg.model, _convert_="all")

    # Compile the model if specified
    if cfg.get("compile", False):
        compile_mode = cfg.get("compile")
        if isinstance(compile_mode, bool):
            compile_mode = "default"
        model.module.compile(mode=compile_mode)
        print(f"Model compiled with mode: {compile_mode}")

    # Instantiate the trainer
    trainer: L.Trainer = instantiate(cfg.trainer)
    print(f"World size: {trainer.world_size}")

    # Load datasets
    train_dataset: Dataset = instantiate(cfg.data).load_dataset(
        model.train_transform_map
    )
    
    # Load validation datasets if specified
    val_dataset = None
    if "val_data" in cfg:
        val_dataset = tree_map(
            lambda ds: ds.load_dataset(model.val_transform_map),
            instantiate(cfg.val_data, _convert_="all"),
        )

    # Set random seed for reproducibility
    L.seed_everything(cfg.seed + trainer.logger.version, workers=True)
    
    # Print configuration for debugging
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Start training
    trainer.fit(
        model,
        datamodule=DataModule(cfg, train_dataset, val_dataset),
        ckpt_path=cfg.get("ckpt_path", None),
    )


# Configuration for pretraining
@hydra.main(version_base="1.3", config_path="conf/pretrain", config_name="trial_pretrain_1000_16_cov.yaml")
def pretrain(cfg: DictConfig):
    """Entry point for pretraining."""
    train_model(cfg)


# Configuration for finetuning
@hydra.main(version_base="1.3", config_path="conf/finetune", config_name="trial_finetune_1000.yaml")
def finetune(cfg: DictConfig):
    """Entry point for finetuning."""
    train_model(cfg)


if __name__ == "__main__":
    # Parse command line arguments to determine which mode to run
    parser = argparse.ArgumentParser(description="Train or finetune time series models")
    parser.add_argument("--mode", type=str, default="pretrain", 
                        choices=["pretrain", "finetune", "finetune_multidataset", "finetune_lightweight"],
                        help="Training mode to use")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to use for caching models and datasets")
    parser.add_argument("--tmp_dir", type=str, default=None,
                        help="Directory to use for temporary files")
    
    args, hydra_args = parser.parse_known_args()
    
    # Setup environment with provided directories
    setup_environment(cache_dir=args.cache_dir, tmp_dir=args.tmp_dir)
    
    # Override sys.argv to pass remaining arguments to Hydra
    sys.argv = [sys.argv[0]] + hydra_args
    
    # Run the selected training mode
    if args.mode == "pretrain":
        pretrain()
    elif args.mode == "finetune":
        finetune()
    elif args.mode == "finetune_multidataset":
        finetune_multidataset()
    elif args.mode == "finetune_lightweight":
        finetune_lightweight()
