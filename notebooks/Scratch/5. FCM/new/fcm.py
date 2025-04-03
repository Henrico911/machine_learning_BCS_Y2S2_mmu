import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Actual labels (for comparison)

# Step 2: Preprocess the Data (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scaling the features

# Step 3: Fuzzy C-Means Implementation
def fuzzy_c_means(X, K, m=2, max_iter=100, tol=1e-6):
    n, d = X.shape  # n is the number of points, d is the number of features
    # Step 3a: Initialize the membership matrix randomly
    U = np.random.rand(n, K)
    U = U / np.sum(U, axis=1, keepdims=True)  # Normalize to ensure each row sums to 1

    # Step 3b: Initialize centroids
    centroids = np.dot(U.T, X) / np.sum(U, axis=0)[:, None]  # Weighted average of points
    
    for i in range(max_iter):
        # Step 3c: Update the membership matrix
        dist = np.linalg.norm(X[:, None] - centroids, axis=2)  # Compute distances between points and centroids
        dist = np.maximum(dist, 1e-10)  # Avoid division by zero

        # Update the membership matrix U using the fuzzy formula
        U_new = (1 / dist**2) ** (1 / (m-1))
        U_new = U_new / np.sum(U_new, axis=1, keepdims=True)  # Normalize to sum to 1

        # Step 3d: Check for convergence
        if np.max(np.abs(U_new - U)) < tol:
            print(f"Converged after {i+1} iterations.")
            break
        
        U = U_new
        
        # Step 3e: Update centroids
        centroids = np.dot(U.T, X) / np.sum(U, axis=0)[:, None]  # Weighted average of points
    
    return centroids, U

# Step 4: Apply Fuzzy C-Means to the scaled Iris data
K = 3  # Number of clusters (since Iris has 3 species)
centroids, U = fuzzy_c_means(X_scaled, K)

# Step 5: Plotting the results (using the first two features for simplicity)
# The clusters are visualized based on their membership values
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=np.argmax(U, axis=1), cmap='viridis', marker='o', s=100, alpha=0.7)

# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.title('Fuzzy C-Means Clustering on Iris Dataset')
plt.legend()
plt.show()

# Step 6: Compare the clustering results with the actual labels
print("True labels for the first few data points:", y[:10])
