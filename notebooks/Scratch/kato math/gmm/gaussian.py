import numpy as np
from scipy.stats import multivariate_normal
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt  
from matplotlib.patches import Ellipse  # For drawing ellipses

class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-3, random_state=None):
        """
        Initializes a Gaussian Mixture Model.

        Args:
            n_components (int): The number of Gaussian components (clusters) in the mixture.
            max_iter (int): The maximum number of iterations for the EM algorithm.
            tol (float): The convergence tolerance.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state  # Add random state for reproducibility
        self.weights = None
        self.means = None
        self.covariances = None
        self.log_likelihood_history = []

    def _initialize_parameters(self, X):
        """
        Initializes the parameters of the GMM.
        Uses k-means for better initialization.

        Args:
            X (numpy.ndarray): The training data.
        """
        n_samples, n_features = X.shape

        # Initialize weights uniformly
        self.weights = np.ones(self.n_components) / self.n_components

        # Initialize means using k-means
        from sklearn.cluster import KMeans  # Import here to avoid global sklearn dependency
        if self.random_state is not None:
            kmeans = KMeans(n_clusters=self.n_components, n_init=10, random_state=self.random_state)  # Specify n_init and random_state
        else:
            kmeans = KMeans(n_clusters=self.n_components, n_init=10)
        kmeans.fit(X)
        self.means = kmeans.cluster_centers_

        # Initialize covariances using the sample covariance of each cluster
        self.covariances = []
        labels = kmeans.labels_
        for k in range(self.n_components):
            X_cluster = X[labels == k]
            self.covariances.append(np.cov(X_cluster.T) + 1e-6 * np.eye(n_features)) # Add small value for numerical stability


    def _e_step(self, X):
        """
        Expectation step.

        Args:
            X (numpy.ndarray): The training data.

        Returns:
            numpy.ndarray: Responsibilities.
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covariances[k]
            )

        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        return responsibilities


    def _m_step(self, X, responsibilities):
        """
        Maximization step.

        Args:
            X (numpy.ndarray): The training data.
            responsibilities (numpy.ndarray): Responsibilities.
        """
        n_samples, n_features = X.shape
        Nk = np.sum(responsibilities, axis=0)

        self.weights = Nk / n_samples

        for k in range(self.n_components):
            self.means[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / Nk[k]

        for k in range(self.n_components):
            X_centered = X - self.means[k]
            self.covariances[k] = np.dot(responsibilities[:, k:k+1].T * X_centered.T, X_centered) / Nk[k]
            self.covariances[k] += 1e-6 * np.eye(n_features)  # Add for numerical stability


    def _calculate_log_likelihood(self, X):
        """
        Calculates the log-likelihood.

        Args:
            X (numpy.ndarray): The training data.

        Returns:
            float: The log-likelihood.
        """
        log_likelihood = 0
        for k in range(self.n_components):
            log_likelihood += self.weights[k] * multivariate_normal.logpdf(X, self.means[k], self.covariances[k]).sum()
        return log_likelihood


    def fit(self, X):
        """
        Fits the GMM to the training data using the EM algorithm.

        Args:
            X (numpy.ndarray): The training data.
        """
        self._initialize_parameters(X)

        previous_log_likelihood = -np.inf

        for i in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

            current_log_likelihood = self._calculate_log_likelihood(X)
            self.log_likelihood_history.append(current_log_likelihood)

            if np.abs(current_log_likelihood - previous_log_likelihood) < self.tol:
                print(f"Converged after {i+1} iterations.")
                break

            previous_log_likelihood = current_log_likelihood
        else:
            print(f"GMM did not converge after {self.max_iter} iterations.")


    def predict(self, X):
        """
        Predicts cluster assignments.

        Args:
            X (numpy.ndarray): The data to predict.

        Returns:
            numpy.ndarray: Cluster assignments.
        """
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        """
        Predicts the probability of each data point belonging to each cluster.

        Args:
            X (numpy.ndarray): The data to predict.

        Returns:
            numpy.ndarray: Probabilities of belonging to each cluster.
        """
        return self._e_step(X)

def visualize_data(X, y, title):
    """
    Visualizes the data points with labels.
    Args:
        X (numpy.ndarray): The data to visualize.
        y (numpy.ndarray): The labels for the data points.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Class")
    plt.show()

def visualize_clusters(X, cluster_assignments, means, covariances, title):
    """
    Visualizes the clusters and their Gaussian distributions.
    Args:
        X (numpy.ndarray): The data to visualize.
        cluster_assignments (numpy.ndarray): The cluster assignments for the data points.
        means (numpy.ndarray): The means of the Gaussian distributions.
        covariances (numpy.ndarray): The covariances of the Gaussian distributions.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Cluster")

    # Plot ellipses representing the Gaussian distributions
    for i in range(len(means)):
        mean = means[i]
        covariance = covariances[i]
        v, w = np.linalg.eigh(covariance)
        angle = np.arctan2(w[1, 0], w[0, 0])
        angle = np.degrees(angle)
        width, height = 2 * np.sqrt(v) * 2  # Scaling factor for visualization
        ellipse = Ellipse(xy=mean[:2], width=width, height=height, angle=angle,
                          edgecolor='red', fc='None', lw=2)
        plt.gca().add_patch(ellipse)

    plt.show()

def plot_log_likelihood(log_likelihood_history):
    """
    Plots the log-likelihood over iterations.
    Args:
        log_likelihood_history (list): The list of log-likelihood values.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(log_likelihood_history)
    plt.title("Log-Likelihood over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.show()


# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use only the first two features for visualization
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Visualize the original data
visualize_data(X_train, y_train, "Original Training Data")

# Create a GMM object.  Note: Iris has 3 classes, so we use 3 components.
gmm = GaussianMixtureModel(n_components=3, max_iter=100, tol=1e-3, random_state=42)

# Fit the GMM to the training data
gmm.fit(X_train)

# Plot the log-likelihood
plot_log_likelihood(gmm.log_likelihood_history)


# Predict cluster assignments for the test data
cluster_assignments = gmm.predict(X_test)

# Visualize the clusters
visualize_clusters(X_test, cluster_assignments, gmm.means, gmm.covariances, "GMM Clusters on Test Data")



# Evaluate the model (GMM is unsupervised, so we need to map clusters to classes)
# This is a simple mapping, and might not be optimal.
# In a real-world scenario, you might need a more sophisticated mapping.

# Create a mapping from cluster to class based on the training data
from collections import defaultdict
cluster_to_class = defaultdict(list)
train_assignments = gmm.predict(X_train)
for i in range(len(train_assignments)):
    cluster_to_class[train_assignments[i]].append(y_train[i])

# Get the most frequent class for each cluster
mapping = {}
for cluster, classes in cluster_to_class.items():
    mapping[cluster] = max(set(classes), key=classes.count) # Find most frequent class

# Apply the mapping to the test predictions
predicted_labels = np.array([mapping[cluster] for cluster in cluster_assignments])


# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Accuracy: {accuracy}")
