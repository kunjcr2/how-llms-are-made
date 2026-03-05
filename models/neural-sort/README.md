# Neural Sort using Pointer Networks

A PyTorch implementation of a Pointer Network capable of sorting integer arrays natively. The model receives an array of integers and predicts a permutation (reordering of indices) that sorts the elements.

## Project Structure

- `data/generator.py`: Synthetic sequence generation
- `model/`: Attention-based Transformer Pointer Network
- `train.py`: Training script with curriculum learning
- `evaluate.py`: Evaluation and exact match metrics script
- `config.py`: Single source of truth for all hyperparams
- `notebook.ipynb`: Self-contained end-to-end Jupyter Notebook

## Running the Code

Install dependencies:
`pip install torch torch-vision numpy`

Train the model:
`python train.py`

Evaluate the model:
`python evaluate.py`
