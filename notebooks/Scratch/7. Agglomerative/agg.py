import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score

# Step 1: Load and preprocess data
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Define a function for Agglomerative Clustering (single linkage)
def agglomerative_clustering(X, n_clusters):
    n = X.shape[0]
    clusters = [[i] for i in range(n)]
    dist_matrix = cdist(X, X, metric='euclidean')
    np.fill_diagonal(dist_matrix, np.inf)

    merges = []

    while len(clusters) > n_clusters:
        min_dist = np.min(dist_matrix)
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        new_cluster = clusters[i] + clusters[j]
        merges.append([i, j, min_dist, len(new_cluster)])
        clusters = [clusters[k] for k in range(len(clusters)) if k != i and k != j] + [new_cluster]

        # Corrected: Calculate new_dist_row based on the *current* dist_matrix
        new_dist_row = np.min(np.array([dist_matrix[i, :], dist_matrix[j, :]]), axis=0)

        # Remove rows and columns *before* creating the new column
        dist_matrix = np.delete(dist_matrix, [i, j], axis=0)
        dist_matrix = np.delete(dist_matrix, [i, j], axis=1)

        # Ensure new_dist_row has the correct size
        new_dist_col = np.append(new_dist_row, np.inf)

        dist_matrix = np.vstack([dist_matrix, new_dist_row])
        dist_matrix = np.column_stack([dist_matrix, new_dist_col])

    return merges

# Step 3: Apply Agglomerative Clustering to the data with desired number of clusters
n_clusters = 3
merges = agglomerative_clustering(X_scaled, n_clusters)

# Step 4: Use scipy's fcluster to get cluster labels
labels = fcluster(np.array(merges), n_clusters, criterion='maxclust')

# Step 5: Plotting a dendrogram for visualization
linked = linkage(X_scaled, method='single')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Data points')
plt.ylabel('Euclidean Distance')
plt.show()

# Step 6: Evaluate the clustering using silhouette score
silhouette_avg = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {silhouette_avg}")

# Step 7: Display the clusters
print("Cluster labels:", labels)

# Step 8: Visualize the clusters (using the first two features for simplicity)
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.title("Agglomerative Clustering Results")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.show()
