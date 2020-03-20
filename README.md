# K Means using PyTorch
PyTorch implementation of kmeans for utilizing GPU

![Alt Text](https://media.giphy.com/media/WsYIwIHHXUcuiR8BeS/giphy.gif)

# Getting Started
```

import torch
import numpy as np
from kmeans_pytorch import kmeans

# data
data_size, dims, num_clusters = 1000, 2, 3
x = np.random.randn(data_size, dims) / 6
x = torch.from_numpy(x)

# kmeans
cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
)
```

see [`example.ipynb`](https://github.com/subhadarship/kmeans_pytorch/blob/master/example.ipynb) for a more elaborate example

# Requirements
* [PyTorch](http://pytorch.org/) version >= 1.0.0
* Python version >= 3.6

# Installation

install with `pip`:
```
pip install kmeans-pytorch
```

**Installing from source**

To install from source and develop locally:
```
git clone https://github.com/subhadarship/kmeans_pytorch
cd kmeans_pytorch
pip install --editable .
```

# CPU vs GPU
see [`cpu_vs_gpu.ipynb`](https://github.com/subhadarship/kmeans_pytorch/blob/master/cpu_vs_gpu.ipynb) for a comparison between CPU and GPU

# Notes
- useful when clustering large number of samples
- utilizes GPU for faster matrix computations
- support euclidean and cosine distances (for now)

# Credits
- This implementation closely follows the style of [this](https://github.com/overshiki/kmeans_pytorch)
- Documentation is done using the awesome theme [jekyllbook](https://github.com/ebetica/jekyllbook)

# License
[MIT](https://github.com/subhadarship/kmeans_pytorch/blob/master/LICENSE)
