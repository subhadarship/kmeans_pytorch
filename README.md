# kmeans using PyTorch
PyTorch implementation of kmeans for utilizing GPU

# Getting started
See `example.ipynb`

# Requirments
* [PyTorch](http://pytorch.org/) version >= 1.0.0
* Python version >= 3.6

# Installation

install with `pip`:
```
pip install fairseq
```

**Installing from source**

To install from source and develop locally:
```
git clone https://github.com/subhadarship/kmeans_pytorch
cd kmeans_pytorch
pip install --editable .
```

# Notes
- useful when clustering large number of samples
- utilizes GPU for faster matrix computations
- support euclidean and cosine distances (for now)
