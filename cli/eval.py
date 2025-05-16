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
from functools import partial
import argparse
from pathlib import Path

import hydra
import torch
from gluonts.time_feature import get_seasonality
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

# Import the resolver from uni2ts.common
from uni2ts.common import hydra_util  # noqa: hydra resolvers
from uni2ts.eval_util.evaluation import evaluate_model


def setup_environment(cache_dir=None, tmp_dir=None):
    """Configure environment variables for evaluation."""
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


def evaluate_model_with_batch_adaptation(cfg: DictConfig):
    """Main evaluation function with automatic batch size adaptation."""
    # Set the environment variable to use specific GPU devices if specified
    if hasattr(cfg, "cuda_device") and cfg.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_device)
        print(f"Using CUDA devices: {cfg.cuda_device}")

    # Load test data and metadata
    test_data, metadata = call(cfg.data)
    
    # Starting batch size
    batch_size = cfg.batch_size
    
    # Try different batch sizes until successful or too small
    while True:
        print(f"Attempting evaluation with batch size: {batch_size}")
        
        # Instantiate the model
        model = call(
            cfg.model, 
            _partial_=True, 
            _convert_="all"
        )(
            prediction_length=metadata.prediction_length,
            target_dim=metadata.target_dim,
            feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
        )
        
        # Instantiate metrics
        metrics = instantiate(cfg.metrics, _convert_="all")
        
        try:
            # Create predictor and evaluate
            predictor = model.create_predictor(batch_size, cfg.device)
            
            print(f"Evaluating model on {metadata.split} data...")
            res = evaluate_model(
                predictor,
                test_data=test_data,
                metrics=metrics,
                batch_size=batch_size,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=get_seasonality(metadata.freq),
            )
            
            # Print results
            print("\nEvaluation Results:")
            print(res)
            
            # Log metrics to TensorBoard
            output_dir = HydraConfig.get().runtime.output_dir
            writer = SummaryWriter(log_dir=output_dir)
            
            # Add metrics to TensorBoard
            for name, metric in res.to_dict("records")[0].items():
                writer.add_scalar(f"{metadata.split}_metrics/{name}", metric)
            
            writer.close()
            print(f"Metrics saved to {output_dir}")
            
            # Successful evaluation, break the loop
            break
        
        except torch.cuda.OutOfMemoryError:
            print(f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size//2}")
            batch_size //= 2
            
            # Check if batch size is too small
            if batch_size < cfg.min_batch_size:
                print(
                    f"batch_size {batch_size} smaller than "
                    f"min_batch_size {cfg.min_batch_size}, ending evaluation"
                )
                break


@hydra.main(version_base="1.3", config_path="conf/eval", config_name="eval_trial.yaml")
def evaluate(cfg: DictConfig):
    """Entry point for evaluation."""
    evaluate_model_with_batch_adaptation(cfg)


if __name__ == "__main__":
    # Parse command line arguments to determine which mode to run
    parser = argparse.ArgumentParser(description="Evaluate time series models")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to use for caching models and datasets")
    parser.add_argument("--tmp_dir", type=str, default=None,
                        help="Directory to use for temporary files")
    parser.add_argument("--config_path", "-cp", type=str, default=None,
                        help="Path to the configuration directory")
    parser.add_argument("--config_name", type=str, default=None,
                        help="Name of the configuration file")
    
    args, hydra_args = parser.parse_known_args()
    
    # Setup environment with provided directories
    setup_environment(cache_dir=args.cache_dir, tmp_dir=args.tmp_dir)

    # Override sys.argv to pass remaining arguments to Hydra
    sys.argv = [sys.argv[0]] + hydra_args
    
    # Run the evaluation
    evaluate()
