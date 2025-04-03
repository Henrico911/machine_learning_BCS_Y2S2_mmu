import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

# --- Divisive Hierarchical Clustering (Math & Implementation) ---
def euclidean_distance(a, b):
    """Calculates Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))

def divisive_clustering(X, num_clusters):
    """
    Performs divisive hierarchical clustering (simplified).

    Math:
    1. Initialize all points in a single cluster.
    2. Find the most dissimilar cluster and split it.
    3. Repeat step 2 until the desired number of clusters is reached.

    Implementation:
    - Uses Euclidean distance as the distance metric.
    - Uses a simplified splitting method (dividing the largest cluster in half).
      Note: This is a very basic implementation for demonstration.
      Real-world divisive clustering uses more sophisticated methods.
    """
    clusters = [list(X)]  # Start with one cluster containing all points

    while len(clusters) < num_clusters:
        largest_cluster = max(clusters, key=len)
        clusters.remove(largest_cluster)

        # Simplified split: divide the largest cluster into two halves
        midpoint = len(largest_cluster) // 2
        clusters.append(largest_cluster[:midpoint])
        clusters.append(largest_cluster[midpoint:])

    # Convert to dictionary format for consistency
    result_clusters = {i: cluster for i, cluster in enumerate(clusters)}
    return result_clusters

def divisive_iris():
    iris = load_iris()
    X = iris.data[:, :2]
    clusters = divisive_clustering(X, 3)
    labels = np.zeros(len(X))
    for cluster_idx, cluster in enumerate(clusters.values()):
        for point in cluster:
            labels[np.where((X == point).all(axis=1))] = cluster_idx
    score = silhouette_score(X, labels)
    print(f"Divisive Clustering Silhouette Score: {score:.4f}")
    plt.figure(figsize=(8, 6))
    for cluster in clusters.values():
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1])
    plt.title("Divisive Clustering (Iris)")
    plt.show()

# Run the divisive clustering algorithm
divisive_iris()