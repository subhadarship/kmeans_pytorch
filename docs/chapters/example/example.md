---
layout: default
title: Example
---

## Installation
Install easily using ```pip```


```python
!pip install kmeans-pytorch
```

    Collecting kmeans-pytorch
      Downloading https://files.pythonhosted.org/packages/b5/c9/eb5b82e7e9741e61acf1aff70530a08810aa0c7e2272c534ff7a150fc5bd/kmeans_pytorch-0.3-py3-none-any.whl
    Installing collected packages: kmeans-pytorch
    Successfully installed kmeans-pytorch-0.3
    

## Import packages
```kmeans_pytorch``` and other packages


```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans, kmeans_predict
```

## Set random seed
For reproducibility


```python
# set random seed
np.random.seed(123)
```

## Generate data
1. Generate data from a random distribution
2. Convert to torch.tensor


```python
# data
data_size, dims, num_clusters = 1000, 2, 3
x = np.random.randn(data_size, dims) / 6
x = torch.from_numpy(x)
```

## Set Device
If available, set device to GPU


```python
# set device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
```

## Perform K-Means


```python
# k-means
cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='euclidean', device=device
)
```

    running k-means on cuda:0..
    

    [running kmeans]: 7it [00:00, 29.79it/s, center_shift=0.000068, iteration=7, tol=0.000100]
    

### Cluster IDs and Cluster Centers


```python
# cluster IDs and cluster centers
print(cluster_ids_x)
print(cluster_centers)
```

    tensor([2, 0, 2, 0, 1, 0, 1, 0, 1, 1, 2, 2, 0, 1, 0, 0, 0, 1, 2, 2, 0, 2, 1, 1,
            2, 0, 1, 2, 2, 1, 2, 0, 1, 1, 2, 1, 1, 2, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2,
            0, 1, 0, 2, 1, 0, 0, 2, 2, 1, 0, 1, 0, 2, 1, 1, 1, 0, 2, 1, 2, 1, 2, 1,
            1, 2, 2, 1, 0, 2, 1, 1, 1, 2, 1, 1, 1, 0, 2, 2, 1, 2, 2, 1, 0, 0, 2, 1,
            1, 0, 0, 0, 1, 1, 1, 0, 2, 1, 0, 2, 1, 2, 0, 0, 1, 0, 2, 2, 2, 1, 1, 1,
            1, 0, 1, 0, 2, 1, 0, 1, 1, 2, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 2,
            2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 0, 2, 2, 0, 0,
            2, 1, 0, 1, 0, 2, 2, 0, 0, 0, 2, 0, 2, 2, 2, 1, 1, 0, 1, 2, 2, 0, 1, 0,
            2, 2, 1, 1, 0, 0, 2, 2, 1, 0, 2, 0, 2, 1, 2, 1, 1, 0, 2, 0, 0, 2, 2, 2,
            0, 1, 0, 1, 1, 2, 1, 2, 1, 0, 0, 2, 2, 2, 2, 0, 1, 1, 1, 2, 1, 0, 2, 0,
            0, 2, 2, 1, 1, 0, 0, 2, 1, 1, 1, 2, 1, 0, 0, 1, 1, 2, 2, 1, 0, 0, 2, 1,
            1, 0, 1, 2, 1, 2, 0, 2, 2, 0, 2, 1, 0, 1, 1, 1, 2, 0, 1, 2, 2, 1, 1, 1,
            0, 1, 0, 1, 2, 0, 2, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 0, 2, 1,
            0, 0, 2, 0, 2, 0, 1, 2, 1, 2, 0, 0, 2, 1, 1, 1, 1, 0, 2, 0, 2, 2, 1, 0,
            1, 2, 2, 1, 1, 1, 2, 2, 0, 0, 1, 2, 1, 1, 0, 1, 2, 1, 2, 0, 0, 2, 0, 1,
            1, 1, 2, 2, 1, 2, 0, 2, 0, 0, 2, 0, 2, 1, 2, 1, 1, 2, 2, 0, 1, 0, 0, 0,
            0, 1, 0, 2, 2, 1, 0, 2, 0, 0, 2, 2, 2, 0, 1, 2, 0, 2, 2, 1, 2, 1, 2, 1,
            0, 0, 0, 2, 0, 2, 2, 2, 0, 1, 1, 0, 2, 2, 0, 2, 2, 1, 0, 0, 2, 2, 0, 0,
            1, 0, 1, 2, 0, 2, 0, 1, 0, 0, 0, 1, 2, 2, 1, 1, 2, 1, 1, 1, 0, 0, 2, 0,
            0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 2, 0, 0, 1, 2, 1, 0, 1, 0, 2, 2,
            0, 0, 0, 0, 2, 1, 0, 2, 1, 1, 2, 1, 0, 2, 0, 2, 0, 2, 1, 1, 2, 1, 0, 0,
            0, 1, 2, 1, 1, 0, 2, 0, 2, 1, 2, 1, 1, 2, 0, 1, 0, 0, 2, 0, 2, 2, 2, 1,
            1, 2, 1, 1, 2, 1, 1, 1, 1, 0, 0, 2, 1, 1, 2, 1, 1, 2, 0, 0, 2, 1, 2, 1,
            1, 1, 1, 1, 0, 0, 2, 2, 1, 0, 1, 2, 2, 0, 1, 0, 2, 0, 2, 2, 2, 0, 1, 2,
            2, 0, 1, 2, 1, 1, 2, 1, 2, 1, 0, 2, 0, 2, 1, 0, 2, 0, 1, 2, 1, 2, 1, 0,
            1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 2, 0, 0, 2, 1, 0, 1, 1, 1, 0, 0,
            2, 1, 0, 2, 1, 1, 0, 2, 1, 2, 0, 2, 2, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 1,
            1, 0, 2, 2, 2, 1, 0, 0, 2, 1, 1, 1, 2, 1, 0, 1, 1, 1, 2, 2, 1, 1, 2, 1,
            0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 1, 0, 1, 2, 1, 1, 1, 1, 0, 1, 0, 0, 2,
            1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 2, 2, 0,
            2, 2, 0, 2, 2, 1, 1, 1, 1, 0, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 0, 2,
            2, 0, 2, 2, 0, 2, 1, 0, 0, 2, 0, 0, 1, 0, 2, 2, 0, 1, 2, 0, 0, 1, 1, 2,
            2, 2, 0, 1, 2, 0, 0, 1, 2, 2, 0, 1, 0, 0, 2, 2, 0, 2, 1, 0, 1, 1, 2, 1,
            0, 2, 1, 1, 0, 1, 1, 0, 2, 2, 2, 2, 1, 0, 0, 0, 2, 1, 2, 2, 0, 0, 0, 2,
            1, 2, 1, 0, 2, 0, 0, 1, 1, 2, 2, 1, 2, 1, 2, 0, 0, 2, 1, 0, 1, 0, 0, 2,
            2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 1, 0, 0, 1, 1, 0, 2, 0, 2, 0, 2, 0, 0, 1,
            0, 0, 0, 2, 0, 2, 1, 2, 0, 1, 0, 2, 0, 0, 0, 1, 0, 1, 1, 1, 0, 2, 2, 0,
            1, 2, 0, 1, 1, 2, 2, 1, 2, 1, 0, 1, 0, 2, 1, 1, 2, 1, 1, 2, 2, 0, 1, 0,
            2, 2, 0, 2, 2, 2, 1, 1, 0, 1, 2, 0, 2, 1, 0, 2, 1, 0, 1, 0, 2, 2, 2, 2,
            2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 0, 0, 1, 1, 2, 1, 0, 1, 1, 1, 0, 2, 2,
            2, 2, 1, 2, 0, 1, 2, 0, 1, 1, 1, 1, 2, 2, 0, 0, 2, 1, 1, 1, 0, 2, 0, 2,
            2, 2, 0, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0])
    tensor([[-0.1075, -0.1522],
            [ 0.1544, -0.0137],
            [-0.0833,  0.1454]])
    

### Create More Data Just for Prediction


```python
# more data
y = np.random.randn(5, dims) / 6
y = torch.from_numpy(y)
```

## Predict


```python
# predict cluster ids for y
cluster_ids_y = kmeans_predict(
    y, cluster_centers, 'euclidean', device=device
)
```

    predicting on cuda:0..
    

### Show Predicted Cluster IDs


```python
print(cluster_ids_y)
```

    tensor([1, 2, 0, 1, 2])
    

## Plot
plot the samples


```python
# plot
plt.figure(figsize=(4, 3), dpi=160)
plt.scatter(x[:, 0], x[:, 1], c=cluster_ids_x, cmap='cool')
plt.scatter(y[:, 0], y[:, 1], c=cluster_ids_y, cmap='cool', marker='X')
plt.scatter(
    cluster_centers[:, 0], cluster_centers[:, 1],
    c='white',
    alpha=0.6,
    edgecolors='black',
    linewidths=2
)
plt.axis([-1, 1, -1, 1])
plt.tight_layout()
plt.show()
```


![png](output_21_0.png)


### Hope the example was useful !!


```python

```
