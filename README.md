# LoRA-FineTuning-GPT2-QA

## Overview

This repository contains the implementation of parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) on the GPT-2 model for extractive question answering tasks with the SQuAD dataset. Weights & Biases (W&B) is used for tracking experiments and logging metrics.

## Purpose

This project tackles the computational challenges of fine-tuning large language models like GPT-2 by applying LoRA to adapt only a small subset of parameters while achieving competitive performance on extractive question answering. It evaluates the impact of hyperparameters (e.g., rank, learning rate, target modules) on convergence, gradient flow, and metrics such as F1-score and Exact Match.

## Usage

### Dependencies

- PyTorch 2.0+
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- Evaluate (Hugging Face)
- Weights & Biases (wandb)

Install dependencies via:

```bash
pip install -r requirements.txt
```

### Running the Code

1. Clone the repository:

```bash
git clone https://github.com/AmirAAZ818/GPT2-LoRA-QA.git
cd GPT2-LoRA-QA
```

2. Set up Weights & Biases (optional but recommended for experiment tracking):
    - Sign up at [wandb.ai](https://wandb.ai) and obtain your API key.
    - Run `wandb login` and paste your API key.

3. Run the notebook:
    - Open `parameter-effcient-fine-tuning-with-lora_Experiments.ipynb` in Jupyter Notebook or Colab.
    - Experiments log metrics (e.g., F1, Exact Match, loss) to W&B; adjust configurations for hyperparameters like *rank* and *learning rate*.
