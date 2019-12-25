import numpy as np
import torch

from kmeans import kmeans

np.random.seed(123)
dims, num_clusters = 100, 5

for data_size in [100000, 1000000]:
    print(f'\ndata size: {data_size}')

    # training data
    x = np.random.randn(data_size, dims) / 6
    x = torch.from_numpy(x)

    ############################
    from time import time

    start_gpu = time()
    kmeans_gpu = kmeans(X=x, num_clusters=num_clusters, device=torch.device('cuda:0'))
    print(f'gpu time: {time() - start_gpu}')
    start_cpu = time()
    kmeans_cpu = kmeans(X=x, num_clusters=num_clusters, device=torch.device('cpu'))
    print(f'cpu time: {time() - start_cpu}')
    #############################
