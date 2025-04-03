import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Step 1: Load and preprocess data
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Actual labels (for comparison)

# Standardize the data (important for distance-based algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Implement K-Means from scratch
def kmeans(X, k, max_iter=100):
    # Randomly initialize the centroids by selecting k random points from X
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iter):
        # Step 1: Assign each point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Shape: (n_samples, k)
        labels = np.argmin(distances, axis=1)  # Assign labels based on closest centroid

        # Step 2: Recompute centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Step 3: Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return centroids, labels

# Step 3: Define a function for Divisive Clustering
def divisive_clustering(X, num_clusters=3):
    # Step 3a: Start with the entire dataset as one cluster
    clusters = [X]
    
    # Step 3b: Recursively split the clusters
    while len(clusters) < num_clusters:
        # Step 3c: Find the cluster to split (the cluster with the most variance)
        cluster_to_split = max(clusters, key=lambda c: np.var(c))
        clusters.remove(cluster_to_split)
        
        # Step 3d: Use KMeans (implemented from scratch) to split the selected cluster into 2 subclusters
        centroids, labels = kmeans(cluster_to_split, k=2)
        
        # Step 3e: Add the two new subclusters back to the list of clusters
        clusters.append(cluster_to_split[labels == 0])
        clusters.append(cluster_to_split[labels == 1])
        
    return clusters

# Step 4: Apply Divisive Clustering
num_clusters = 3  # We want to create 3 clusters
clusters = divisive_clustering(X_scaled, num_clusters=num_clusters)

# Step 5: Visualize the clusters
# To visualize, we'll reduce dimensions to 2D using PCA for better visualization (optional step)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize the results
plt.figure(figsize=(8, 6))
for cluster in clusters:
    cluster_pca = pca.transform(cluster)
    plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1])
    
plt.title(f'Divisive Clustering with {num_clusters} Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Step 6: Print out the clusters (index-wise, for reference)
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1} contains {len(cluster)} points.")
