import numpy as np

# Euclidean Distance Function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# K-Means Algorithm Implementation
def kmeans(X, K, max_iters=100):
    # Step 1: Randomly initialize centroids (choosing K points from X)
    np.random.seed(42)  # For reproducibility
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]

    for _ in range(max_iters):
        # Step 2: Assign each point to the nearest centroid
        labels = np.array([np.argmin([euclidean_distance(x, centroid) for centroid in centroids]) for x in X])
        
        # Step 3: Update centroids by calculating the mean of points in each cluster
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(K)])
        
        # Step 4: Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, labels

# Example usage:
if __name__ == "__main__":
    # Generating random data points
    X = np.random.rand(100, 2)  # 100 data points in 2D
    
    K = 3  # Number of clusters
    centroids, labels = kmeans(X, K)
    
    # Output the final centroids and cluster labels for each point
    print("Final Centroids:", centroids)
    print("Cluster Labels:", labels)
