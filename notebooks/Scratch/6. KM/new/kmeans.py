import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def kmeans(data, k, max_iters=100):
    """K-means clustering implementation."""
    n, d = data.shape
    # Initialize centroids randomly
    centroids = data[np.random.choice(n, k, replace=False)]

    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels, centroids

# Generate a sample dataset
data, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)

# Set parameters
k = 3  # Number of clusters

# Run K-means
labels, centroids = kmeans(data, k)

# Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title('K-means Clustering')
plt.legend()
plt.show()

print("Cluster Centers:\n", centroids)
