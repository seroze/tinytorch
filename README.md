# TinyTorch

A from-scratch implementation of a minimal ML framework — tensors, autograd, layers, losses, and a training loop — built module by module.

---

> **Inspired by** the [TinyTorch textbook](https://mlsysbook.ai/tinytorch/preface.html) from the Machine Learning Systems book (Harvard, open access).
> The textbook walks through every design decision; this repo is the working code built alongside it.

> **Personal notes & writeup:** [seroze.github.io/build-your-own-pytorch](https://seroze.github.io/build-your-own-pytorch/)

---

## What's built

| Module | What it covers |
|--------|---------------|
| `storage.py` | Flat 1D buffer — raw memory |
| `ndarray.py` | Multi-dimensional array with strides, arithmetic, matmul |
| `tensor.py` | Wraps NDArray, adds `requires_grad`, `grad`, `grad_fn` |
| `activations.py` | ReLU, Sigmoid, Tanh, GELU, Softmax (numerically stable) |
| `layers.py` | Linear (Xavier init), Sequential |
| `losses.py` | MSELoss, CrossEntropyLoss (log-sum-exp), BinaryCrossEntropyLoss |
| `dataloader.py` | Dataset, DataLoader with batching and shuffle |
| `autograd.py` | Reverse-mode autograd — computation graph + backward pass |

## Quickstart

```bash
cd tinytorch
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

```python
from tinytorch import Linear, Sequential, ReLU, Softmax, CrossEntropyLoss
from tinytorch import NDArray, Tensor

model = Sequential(Linear(4, 8), ReLU(), Linear(8, 3), Softmax())
x     = Tensor(NDArray((4,), data=[1.0, 2.0, 3.0, 4.0]))
y     = model(x)   # shape (3,), sums to 1.0
```

## Run tests

```bash
python -m unittest discover tests/
```
