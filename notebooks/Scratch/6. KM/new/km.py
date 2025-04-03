import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Step 1: Load Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Actual labels (for comparison)

# Step 2: Preprocess the Data (Standardization)
# Since K-Means is sensitive to the scale of the data, we scale it to have zero mean and unit variance.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scaling the features

# Step 3: Define Euclidean Distance Function
# This function computes the Euclidean distance between two points (vectors).
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Step 4: Implement K-Means Algorithm
# This function performs the K-Means clustering.
def kmeans(X, K, max_iters=100):
    # Step 4a: Initialize centroids randomly by selecting K points from the dataset
    np.random.seed(42)  # For reproducibility
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    
    # Step 4b: Iterative process to assign points to clusters and update centroids
    for i in range(max_iters):
        # Step 4b1: Assign each point to the nearest centroid based on Euclidean distance
        labels = np.array([np.argmin([euclidean_distance(x, centroid) for centroid in centroids]) for x in X])
        
        # Step 4b2: Recalculate centroids as the mean of the points in each cluster
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(K)])
        
        # Step 4b3: Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            print(f"Converged after {i+1} iterations.")
            break
        
        centroids = new_centroids
    
    return centroids, labels

# Step 5: Apply K-Means to the scaled Iris data
K = 3  # We know there are 3 classes in the Iris dataset
centroids, labels = kmeans(X_scaled, K)

# Output the final centroids and the first few cluster labels
print("Final Centroids:\n", centroids)
print("Labels for the first few data points:", labels[:10])

# Step 6: Visualize the Results
# Plotting the first two features (sepal length vs sepal width) for simplicity
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o', s=100, alpha=0.7)

# Plot the centroids on top
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.title('K-Means Clustering on Iris Dataset')
plt.legend()
plt.show()

# Step 7: Compare the Clustering Results with the Actual Labels
# Since K-Means is unsupervised, it does not know the actual class labels.
# We can compare the cluster labels with the true labels for an assessment.
print("True labels for the first few data points:", y[:10])

