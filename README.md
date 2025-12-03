# KMeans Clustering from Scratch with NumPy

## Introduction
This project implements the **KMeans clustering algorithm** from scratch using `NumPy`.  
The algorithm partitions a dataset into `k` clusters by minimizing the within-cluster sum of squares.

## Algorithm
Given a dataset \$(X = \{x_1, x_2, \dots, x_n\}\$) and a number of clusters `k`:

1. Initialize `k` centroids randomly.
2. Repeat until convergence:
   1. Assign each data point to the nearest centroid:
     $c_i = argmin_j ||x_i - μ_j||$ 
   2. Update each centroid to the mean of assigned points:
     $μ_j = (1 / |C_j|) * sum(x_i  in  C_j) x_i$


## Usage

```python
from kmeans import KMeansClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate synthetic data
data = make_blobs(n_samples=100, n_features=2, centers=3)
X = data[0]

# Fit KMeans
model = KMeansClustering(k=3)
labels = model.fit(X)

# Visualize
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(model.centroids[:,0], model.centroids[:,1], marker="*", s=200)
plt.show()
```
## Requirements
Numpy
Matplotlib
Scikit-learn (only for data generation)

## Notes
This implementation is purely for learning purposes and may not be as optimized as scikit-learn's version.

Works for 2D and higher-dimensional data
