# Personal Training Notebooks

Welcome to my personal training repository! This directory serves as a centralized collection of Jupyter Notebooks that I use to train, optimize, and export various machine learning models. 

By open-sourcing these notebooks, I hope to share my workflows and allow the community to learn, fine-tune, experiment, and deploy their own versions of these models for custom use cases.

Currently, this repository features models from the **L3mcore** (LEMoE - Light Easy Mix Of Experts) ecosystem, but more notebooks and model architectures will be added over time.

## Repository Structure

| Model Directory | Description | Links |
| :--- | :--- | :--- |
| [**lemoeppc**](https://github.com/jrodriiguezg/train-notebooks/tree/main/lemoeppc) | Sequence Classification Pipeline | [README](https://github.com/jrodriiguezg/train-notebooks/blob/main/lemoeppc/README.md) \| [Notebook](https://github.com/jrodriiguezg/train-notebooks/blob/main/lemoeppc/train.ipynb) |
| [**lemoe-query-distiler**](https://github.com/jrodriiguezg/train-notebooks/tree/main/lemoe-query-distiler) | Token Classification / Query Distillation | [README](https://github.com/jrodriiguezg/train-notebooks/blob/main/lemoe-query-distiler/README.md) \| [Notebook](https://github.com/jrodriiguezg/train-notebooks/blob/main/lemoe-query-distiler/train.ipynb) |

### 1. `lemoeppc`
Contains the complete workflow for fine-tuning a BERT-based sequence classification model. This model is typically used to categorize, filter, or route incoming text queries. The notebook covers everything from dataset preparation to ONNX export and INT8 dynamic quantization for high-speed CPU inference.

### 2. `lemoe-query-distiler`
Contains the pipeline for training a DeBERTa-based token classification model. This "distiller" acts as an advanced filter that evaluates each word in a query and decides whether to keep it or discard it, effectively extracting the core intent of a prompt.

## Getting Started

> **Hardware Note:** The original training of these models was performed locally on a laptop equipped with an NVIDIA RTX 5070 GPU. However, the notebooks are highly optimized and perfectly suited to be run in **Google Colab**, AWS SageMaker, or any local JupyterLab instance with CUDA support.

To use these notebooks:

1. Clone this repository.
2. Navigate to the model directory you want to train (`lemoeppc` or `lemoe-query-distiler`).
3. Prepare your custom datasets in the `.jsonl` format as specified in each sub-directory's README.
4. Run the notebook cells sequentially. Dependencies will be installed automatically in the first cell.

## License
These notebooks are open-sourced for the community to learn, train, and build upon these models. Feel free to use and adapt them for your own projects!
