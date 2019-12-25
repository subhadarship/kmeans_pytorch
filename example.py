import matplotlib.pyplot as plt
import numpy as np
import torch

from kmeans import kmeans, kmeans_predict

np.random.seed(123)
data_size, dims, num_clusters = 1000, 2, 3

# set device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# data
x = np.random.randn(data_size, dims) / 6
x = torch.from_numpy(x)

# k-means
cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='cosine', device=device
)

# cluster IDs and cluster centers
print(cluster_ids_x)
print(cluster_centers)

# more data
y = np.random.randn(5, dims) / 6
y = torch.from_numpy(y)

# predict cluster ids for y
cluster_ids_y = kmeans_predict(
    y, cluster_centers, 'cosine', device=device
)
print(cluster_ids_y)

plt.figure(figsize=(6, 5), dpi=160)
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
