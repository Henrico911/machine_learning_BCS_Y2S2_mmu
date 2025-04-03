import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

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

def cluster_and_evaluate(data, num_clusters=3, plot_results=True):
    """
    Performs hierarchical and GMM clustering, evaluates the results, and optionally plots them.

    Args:
        data (pd.DataFrame or np.ndarray): The input data.
        num_clusters (int): The number of clusters to form.
        plot_results (bool): Whether to plot the clustering results.
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

    # Plotting
    if plot_results:
        plt.figure(figsize=(16, 6))

        # Hierarchical Plot
        plt.subplot(1, 2, 1)
        for cluster in clusters.values():
            cluster = np.array(cluster)
            plt.scatter(cluster[:, 0], cluster[:, 1])
        plt.title("Hierarchical Clustering")

        # GMM Plot
        plt.subplot(1, 2, 2)
        plt.scatter(X[:, 0], X[:, 1], c=labels_gmm, cmap='viridis')
        plt.title("GMM Clustering")

        plt.show()

# Example Usage with Sample Data (as before)
np.random.seed(42)
X1 = np.random.randn(100, 2) + np.array([5, 5])
X2 = np.random.randn(100, 2) + np.array([-5, -5])
X3 = np.random.randn(100, 2) + np.array([5, -5])
X = np.vstack([X1, X2, X3])

cluster_and_evaluate(X)

# Example Usage with IRI-like Data (CSV)
def process_iri_data(filepath, features=['feature1', 'feature2'], num_clusters=3):
    """
    Loads, processes, and clusters IRI-like data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.
        features (list): List of feature names to use for clustering.
        num_clusters (int): Number of clusters to create.
    """
    try:
        iri_data = pd.read_csv(filepath)
        if not all(feature in iri_data.columns for feature in features):
            print(f"Error: Features {features} not found in the CSV file.")
            return

        data = iri_data[features].dropna() # Remove NaN values, and only keep the specified features.
        cluster_and_evaluate(data, num_clusters=num_clusters)

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example of calling the IRI function. Replace 'your_iri_data.csv' and the features with your actual data.
#process_iri_data("your_iri_data.csv", features=['Sales', 'Price'], num_clusters=4)