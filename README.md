# MTSFM: Macroeconomic Time Series Foundation Model

[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)


MTSFM is a specialized PyTorch-based foundation model for macroeconomic time series forecasting. Built upon the [Uni2TS framework](https://github.com/SalesforceAIResearch/uni2ts) and [Moirai](https://arxiv.org/abs/2402.02592) architecture, MTSFM has been fine-tuned with custom optimization approaches and specialized training on macroeconomic datasets to deliver superior performance for economic forecasting tasks.

## 🌟 Key Features

* **Specialized for Macroeconomics**: Optimized specifically for forecasting economic indicators, financial metrics, and monetary policy variables
* **Custom Loss Functions**: Enhanced training objectives tailored for macroeconomic variables
* **Adaptive Forecasting**: Handles mixed-frequency data and irregular sampling common in economic time series
* **Uncertainty Quantification**: Provides robust probabilistic forecasts essential for economic policy decisions
* **Zero-Shot Capabilities**: Leverages Moirai's transfer learning abilities while excelling on macroeconomic datasets

## ⚙️ Installation

MTSFM uses the same installation process as the original Uni2TS framework:

1. Clone repository:
```shell
git clone https://github.com/bkoyuncu/mtsfm.git
cd mtsfm
```

2) Create virtual environment:
```shell
virtualenv venv
. venv/bin/activate
```

3) Build from source:
```shell
pip install -e '.[notebook]'
```

4) Create a `.env` file:
```shell
touch .env
```

MTSFM requires several environment variables to be set in your .env file. Create this file in the root project directory with the following variables:

# Required paths
PROJECT_PATH=/absolute/path/to/your/MTSFM/project
MODEL_PATH=/absolute/path/to/store/model/checkpoints
CUSTOM_DATA_PATH=/absolute/path/to/your/datasets


## 📔 Jupyter Notebooks
See process_macroecon_notebook.ipynb for creating the dataset using our HuggingFace dataset


## 🧑‍🔬 Training Workflow

MTSFM uses a two-step process: first saving the base MOIRAI model, then fine-tuning it with specialized optimizations for macroeconomic data.

### Step 1: Saving the MOIRAI Model Checkpoint

First, save the pre-trained MOIRAI model to use as your starting point:

```shell
python -m cli.train \
  -cp conf/finetune \
  --config-name finetune_save_model.yaml \
  run_name=save_base \
  model=moirai_1.0_R_small_ours_save
```

This command saves the MOIRAI checkpoint that will be the foundation for MTSFM. Make sure to modify the `finetune_save_model.yaml` configuration file to set appropriate hyperparameters and paths.

### Step 2: Fine-tuning with Specialized Optimizations

After saving the MOIRAI checkpoint, fine-tune the model with specialized optimizations for macroeconomic data:

```shell
python -m cli.train \
  -cp conf/finetune \
  --config-name finetune_saved.yaml \
  run_name=load_base \
  model=moirai_1.0_R_small_ours_load
```

Important: Before running this command, you must modify the `finetune_saved.yaml` file to:
1. Point to the correct path of your saved MOIRAI checkpoint from Step 1
2. Configure your macroeconomic dataset specifications
3. Set the specialized optimization parameters (loss functions, learning rates, etc.)

Sample YAML configuration sections that need to be updated:
```yaml
# In finetune_saved.yaml
model:
  # Path to your saved MOIRAI checkpoint
  checkpoint_path: "path/to/your/saved/moirai/checkpoint.ckpt"
  
data:
  # Your macroeconomic dataset configuration
  name: "your_economic_dataset"
  path: "${env:CUSTOM_DATA_PATH}/your_processed_data"
  
optimizer:
  # Specialized optimization parameters for macroeconomic data
  name: "adam"
  lr: 1e-5
  weight_decay: 0.01
```



## 💻 Command Line Interface
We provide a command line interface for training, evaluation, and inference. See the [cli directory](cli) for more details.

## 🙏 Acknowledgements

MTSFM builds upon the excellent work of the [Uni2TS](https://github.com/SalesforceAIResearch/uni2ts) framework and [Moirai](https://arxiv.org/abs/2402.02592) architecture by Salesforce AI Research. We are grateful for their open-source contribution that made our specialized model possible.