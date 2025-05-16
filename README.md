# MTSFM: Macroeconomic Time Series Foundation Model

[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)


MTSFM is a specialized PyTorch-based foundation model for macroeconomic time series forecasting. Built upon the [Uni2TS framework](https://github.com/SalesforceAIResearch/uni2ts) and [Moirai](https://arxiv.org/abs/2402.02592) architecture, MacroTS has been fine-tuned with custom optimization approaches and specialized training on macroeconomic datasets to deliver superior performance for economic forecasting tasks.

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

## 📔 Jupyter Notebooks
See read_dataset.ipynb for creating the dataset.
<!-- See the [example folder](example) for examples specific to macroeconomic forecasting tasks. -->

## 🧑‍🔬 Fine-tuning on Your Data

To fine-tune MacroTS on your own economic dataset:

1. Process your data into the required format:
```shell
echo "CUSTOM_DATA_PATH=PATH_TO_SAVE" >> .env
```

2. Run the fine-tuning script:
```shell
python -m cli.train \
  -cp conf/finetune \
  run_name=econ_finetune \ 
  model=macrots_1.0_base \ 
  data=your_econ_data
```


## 💻 Command Line Interface
We provide a command line interface for training, evaluation, and inference. See the [cli directory](cli) for more details.

## 🙏 Acknowledgements

MacroTS builds upon the excellent work of the [Uni2TS](https://github.com/SalesforceAIResearch/uni2ts) framework and [Moirai](https://arxiv.org/abs/2402.02592) architecture by Salesforce AI Research. We are grateful for their open-source contribution that made our specialized model possible.