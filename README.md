# Transformers
[![CI](https://github.com/SverreNystad/transfomer/actions/workflows/ci.yml/badge.svg)](https://github.com/SverreNystad/transfomer/actions/workflows/ci.yml)

This repository provides a PyTorch implementation of the Transformer as introduced in Vaswani et al.’s 2017 paper “Attention Is All You Need,” which forgoes recurrence and convolution in favor of purely attention-based encoder–decoder stacks to achieve state-of-the-art machine translation performance with greatly improved parallelizability. At its core are scaled dot-product self-attention and multi-head attention mechanisms that dynamically weight token interactions, augmented by sinusoidal positional encodings to inject sequence order information. As the foundational architecture behind BERT, GPT, and other leading large-scale language models, the Transformer has reshaped NLP and beyond by enabling efficient learning of long-range dependencies.


## Prerequisites

- Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- Python 3.10 or higher is required for the backend. [Download Python](https://www.python.org/downloads/)
- CUDA Toolkit (optional): For GPU acceleration with PyTorch

## Installation
Clone the repository using the following command in the terminal:
```bash
git clone https://github.com/SverreNystad/transformer.git
cd transformer
```

### Environment variables
Copy the `.env.example` file to `.env` and set the required environment variables. The `.env` file is used to configure the WandB monitoring.
```bash
cp .env.example .env
```


Configure the pre-commit hooks by running the following command in the terminal:
```bash
pip install pre-commit
pre-commit install
```

Setup the virtual environment by running the following command in the terminal:
```bash
python -m venv venv
```
Activate the virtual environment by running the following command in the terminal:
```bash
source venv/bin/activate
```

Install the required packages by running the following command in the terminal:
```bash
pip install -r requirements.txt
```

# Usage
To run the training script, use the following command:
```bash
python training_casual_modeling.py
```

# Testing
To run the tests, use the following command:

```bash
pytest
```
