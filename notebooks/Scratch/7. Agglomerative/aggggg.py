import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
import pandas as pd

# --- Agglomerative Hierarchical Clustering (From Scratch) ---
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def hierarchical_clustering(X, num_clusters):
    clusters = {i: [x] for i, x in enumerate(X)}
    distances = {(i, j): euclidean_distance(X[i], X[j]) for i in range(len(X)) for j in range(i)}
    
    while len(clusters) > num_clusters:
        i, j = min(distances, key=distances.get)
        clusters[i].extend(clusters[j])
        del clusters[j]
        
        # Update distances
        new_distances = {}
        for a in clusters:
            if a != i:
                new_distances[(min(i, a), max(i, a))] = euclidean_distance(np.mean(clusters[i], axis=0), np.mean(clusters[a], axis=0))
        distances = new_distances
    
    return clusters

# --- Gaussian Mixture Model (From Scratch) ---
def gaussian_pdf(x, mean, cov):
    d = len(x)
    coeff = 1 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(cov)))
    exponent = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    return coeff * np.exp(exponent)

def gmm(X, k, max_iter=100):
    n, d = X.shape
    np.random.seed(42)
    means = X[np.random.choice(n, k, replace=False)]
    covs = [np.eye(d) for _ in range(k)]
    pis = np.ones(k) / k
    
    for _ in range(max_iter):
        responsibilities = np.zeros((n, k))
        
        for i in range(n):
            for j in range(k):
                responsibilities[i, j] = pis[j] * gaussian_pdf(X[i], means[j], covs[j])
            responsibilities[i] /= np.sum(responsibilities[i])
        
        Nk = np.sum(responsibilities, axis=0)
        means = [np.sum(responsibilities[:, j].reshape(-1, 1) * X, axis=0) / Nk[j] for j in range(k)]
        covs = [sum(responsibilities[i, j] * np.outer(X[i] - means[j], X[i] - means[j]) for i in range(n)) / Nk[j] for j in range(k)]
        pis = Nk / n
    
    return np.argmax(responsibilities, axis=1)

def cluster_and_evaluate(data, num_clusters=3):
    """
    Performs hierarchical and GMM clustering, evaluates the results.

    Args:
        data (pd.DataFrame or np.ndarray): The input data.
        num_clusters (int): The number of clusters to form.
    """

    # Ensure data is a numpy array
    if isinstance(data, pd.DataFrame):
        X = data.values
    else:
        X = data

    # Hierarchical Clustering
    clusters = hierarchical_clustering(X, num_clusters)
    labels_hierarchical = np.zeros(len(X))
    for cluster_idx, cluster in enumerate(clusters.values()):
        for point in cluster:
            labels_hierarchical[np.where((X == point).all(axis=1))] = cluster_idx

    # GMM Clustering
    labels_gmm = gmm(X, num_clusters)

    # Evaluation
    silhouette_hierarchical = silhouette_score(X, labels_hierarchical)
    silhouette_gmm = silhouette_score(X, labels_gmm)

    print(f"Silhouette Score for Hierarchical Clustering: {silhouette_hierarchical:.4f}")
    print(f"Silhouette Score for GMM: {silhouette_gmm:.4f}")

    return X, clusters, labels_gmm

def plot_hierarchical(X, clusters, title="Hierarchical Clustering"):
    """Plots the results of hierarchical clustering."""
    plt.figure(figsize=(8, 6))
    for cluster in clusters.values():
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1])
    plt.title(title)
    plt.show()

def plot_gmm(X, labels_gmm, title="GMM Clustering"):
    """Plots the results of GMM clustering."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels_gmm, cmap='viridis')
    plt.title(title)
    plt.show()

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Using only the first two features for visualization

# Cluster and evaluate the Iris dataset
X_data, hierarchical_clusters, gmm_labels = cluster_and_evaluate(X, num_clusters=3)

# Plot the results separately
plot_hierarchical(X_data, hierarchical_clusters)
plot_gmm(X_data, gmm_labels)

# Example of using the full iris dataset.
X_full = iris.data
X_full_data, full_hierarchical_clusters, full_gmm_labels = cluster_and_evaluate(X_full, num_clusters=3)

#Full iris dataset cannot be plotted in 2d, so only the scores are printed.