import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

# --- Gaussian Mixture Model (Math & Implementation) ---
def gaussian_pdf(x, mean, cov):
    """Calculates Gaussian probability density function."""
    d = len(x)
    coeff = 1 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(cov)))
    exponent = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    return coeff * np.exp(exponent)

def gmm(X, k, max_iter=100):
    """
    Performs Gaussian Mixture Model clustering.

    Math:
    1. Initialize means, covariances, and mixing coefficients.
    2. E-step: Calculate responsibilities (probability of each point belonging to each cluster).
    3. M-step: Update means, covariances, and mixing coefficients based on responsibilities.
    4. Repeat steps 2 and 3 until convergence.

    Implementation:
    - Uses Gaussian PDF to calculate probabilities.
    - Implements the EM algorithm for parameter estimation.
    """
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

def gmm_iris():
    iris = load_iris()
    X = iris.data[:, :2]
    labels = gmm(X, 3)
    score = silhouette_score(X, labels)
    print(f"GMM Clustering Silhouette Score: {score:.4f}")
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title("GMM Clustering (Iris)")
    plt.show()

# Run the GMM clustering algorithm
gmm_iris()