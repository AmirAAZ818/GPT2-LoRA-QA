# LoRA-FineTuning-GPT2-QA

## Overview

This repository contains the implementation of parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) on the GPT-2 model for extractive question answering tasks with the SQuAD dataset. Weights & Biases (W&B) is used for tracking experiments and logging metrics.

## What is LoRA?

LoRA reduces the computational cost of fine-tuning LLMa by updating only a low-rank decomposition of the weight matrix, leaving the original weights frozen. The formula is: $$ W' = W + \Delta W = W + BA $$ where: 
- \( W \) is the original weight matrix
- \( \Delta W = BA \) with \( B \in \mathbb{R}^{d \times r} \) and \( A \in \mathbb{R}^{r \times k} \)
- \( r \ll \min(d, k) \) controls the rank, minimizing trainable parameters.

## Purpose

This project tackles the computational challenges of fine-tuning large language models like GPT-2 by applying LoRA to adapt only a small subset of parameters while achieving competitive performance on extractive question answering. It evaluates the impact of hyperparameters on convergence, gradient flow, and metrics such as F1-score and Exact Match.

## Experiments

The training process used the following hyperparameters, detailed in the table below:

| Feature            | Value                  |
|--------------------|-------------------------|
| Batch Size         | 8                      |
| Number of Epochs   | 3                      |
| Optimizer          | AdamW                  |
| Learning Rate      | 0.0001, 0.0002, 0.0005|
| LoRA Rank          | 4, 8, 16, 32           |
| Target Modules     | Attention, Attention + Projection |
| Alpha              | 16                     |
| LoRA Scaling Factor| 16                     |

For instance, the effect of varying target modules on loss is illustrated below, with Attention + Projection showing superior convergence.

| ![Evaluation Loss](GPT2-LoRA-QA\Assets\eval_loss_tm.png) | ![Train Loss](GPT2-LoRA-QA\Assets\train_loss_tm.png) |
|---------------------------|---------------------------|
| Evaluation Loss               | Train Loss                 |

Explore additional visualizations in this [Weights & Biases Project](https://wandb.ai/amiraaz/Parameter-Efficient%20Fine-tuning%20of%20GPT-2%20with%20LoRA?nw=nwuseramiraaz)!

## Results

The **best** configuration achieved the following performance, summarized in the table below:

| LoRA Rank | Target Module     | Learning Rate | F1-Score | Exact Match (EM) |
|-----------|-------------------|---------------|----------|------------------|
| 32        | Attention + Projection | 0.0005        | 90.67    | 80               |
| 8         | Attention         | 0.0002        | 80       | 80               |

## Setup

### Dependencies

- PyTorch
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
   - Experiments log metrics (e.g., F1, Exact Match, loss) to W&B; adjust configurations for hyperparameters like rank and learning rate.