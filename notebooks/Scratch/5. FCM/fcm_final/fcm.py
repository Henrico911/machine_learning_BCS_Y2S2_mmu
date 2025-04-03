# fcm.py
import numpy as np

class FuzzyCMeans:
    """
    Fuzzy C-Means Clustering Algorithm.

    Parameters:
    C : int
        The number of clusters.
    m : float
        The fuzziness parameter (m > 1).
    epsilon : float, optional
        The convergence criterion. Default is 0.01.
    max_iter : int, optional
        The maximum number of iterations. Default is 100.
    """

    def __init__(self, C, m, epsilon=0.01, max_iter=100):
        self.C = C
        self.m = m
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.U = None
        self.V = None

    def fit(self, X):
        """
        Fit the Fuzzy C-Means model to the data.

        Parameters:
        X : numpy array, shape (n_samples, n_features)
            The input data to cluster.
        """
        n_samples, n_features = X.shape

        # Step 1: Initialize the membership matrix U randomly
        self.U = np.random.rand(n_samples, self.C)
        self.U = self.U / np.sum(self.U, axis=1, keepdims=True)

        for iteration in range(self.max_iter):
            # Step 2: Calculate cluster centers V
            U_m = self.U ** self.m
            self.V = np.dot(U_m.T, X) / np.sum(U_m.T, axis=1, keepdims=True)

            # Step 3: Calculate the distance matrix D
            D = np.zeros((n_samples, self.C))
            for i in range(self.C):
                D[:, i] = np.linalg.norm(X - self.V[i], axis=1)

            # Step 4: Update the membership matrix U
            U_new = np.zeros((n_samples, self.C))
            for i in range(self.C):
                for j in range(self.C):
                    U_new[:, i] += (D[:, i] / D[:, j]) ** (2 / (self.m - 1))
            U_new = 1 / U_new

            # Step 5: Check for convergence
            if np.linalg.norm(U_new - self.U) < self.epsilon:
                break

            self.U = U_new

    def predict(self, X):
        """
        Predict the cluster membership for new data points.

        Parameters:
        X : numpy array, shape (n_samples, n_features)
            The input data to predict.

        Returns:
        U : numpy array, shape (n_samples, C)
            The membership matrix for the input data.
        """
        n_samples, n_features = X.shape
        D = np.zeros((n_samples, self.C))
        for i in range(self.C):
            D[:, i] = np.linalg.norm(X - self.V[i], axis=1)

        U_new = np.zeros((n_samples, self.C))
        for i in range(self.C):
            for j in range(self.C):
                U_new[:, i] += (D[:, i] / D[:, j]) ** (2 / (self.m - 1))
        U_new = 1 / U_new

        return U_new

    def get_cluster_centers(self):
        """
        Get the cluster centers.

        Returns:
        V : numpy array, shape (C, n_features)
            The cluster centers.
        """
        return self.V

    def get_membership_matrix(self):
        """
        Get the membership matrix.

        Returns:
        U : numpy array, shape (n_samples, C)
            The membership matrix.
        """
        return self.U