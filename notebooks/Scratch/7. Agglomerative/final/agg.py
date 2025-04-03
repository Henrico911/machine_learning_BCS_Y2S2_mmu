import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

# --- Agglomerative Hierarchical Clustering (Math & Implementation) ---
def euclidean_distance(a, b):
    """Calculates Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))

def agglomerative_clustering(X, num_clusters):
    """
    Performs agglomerative hierarchical clustering.

    Math:
    1. Initialize each point as a single cluster.
    2. Calculate the distance between all pairs of clusters.
    3. Merge the two closest clusters.
    4. Repeat steps 2 and 3 until the desired number of clusters is reached.

    Implementation:
    - Uses Euclidean distance as the distance metric.
    - Uses the average linkage method (average distance between points in two clusters).
    """
    clusters = {i: [x] for i, x in enumerate(X)}  # Initialize clusters
    distances = {(i, j): euclidean_distance(X[i], X[j]) for i in range(len(X)) for j in range(i)}  # Initial distances

    while len(clusters) > num_clusters:
        i, j = min(distances, key=distances.get)  # Find closest clusters
        clusters[i].extend(clusters[j])  # Merge clusters
        del clusters[j]

        # Update distances (average linkage)
        new_distances = {}
        for a in clusters:
            if a != i:
                new_distances[(min(i, a), max(i, a))] = np.mean([euclidean_distance(p1, p2) for p1 in clusters[i] for p2 in clusters[a]])
        distances = new_distances

    return clusters

def agglomerative_iris():
    iris = load_iris()
    X = iris.data[:, :2]
    clusters = agglomerative_clustering(X, 3)
    labels = np.zeros(len(X))
    for cluster_idx, cluster in enumerate(clusters.values()):
        for point in cluster:
            labels[np.where((X == point).all(axis=1))] = cluster_idx
    score = silhouette_score(X, labels)
    print(f"Agglomerative Clustering Silhouette Score: {score:.4f}")
    plt.figure(figsize=(8, 6))
    for cluster in clusters.values():
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1])
    plt.title("Agglomerative Clustering (Iris)")
    plt.show()

# Run the agglomerative clustering algorithm
agglomerative_iris()