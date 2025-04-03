# main.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from fcm import FuzzyCMeans  # Import the FCM library

# Load the Iris dataset
data = load_iris()
X = data.data[:, :2]  # Use only the first two features for visualization

# Step 1: Plot the unclustered data points
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Data Points', alpha=0.7)
plt.title('Before Clustering: Unclustered Data')  # Label for unclustered data
plt.xlabel('Feature 1 (Sepal Length)')
plt.ylabel('Feature 2 (Sepal Width)')
plt.legend()
plt.grid(True)
plt.show()

# Step 2: Apply Fuzzy C-Means Clustering
fcm = FuzzyCMeans(C=3, m=2, epsilon=0.01, max_iter=100)  # Initialize FCM
fcm.fit(X)  # Fit the model to the data

# Get the results
U = fcm.get_membership_matrix()
V = fcm.get_cluster_centers()

# Assign each data point to the cluster with the highest membership
cluster_assignments = np.argmax(U, axis=1)

# Step 3: Plot the clustered data points
plt.figure(figsize=(6, 6))
for i in range(fcm.C):
    plt.scatter(
        X[cluster_assignments == i, 0],  # Feature 1
        X[cluster_assignments == i, 1],  # Feature 2
        label=f'Cluster {i + 1}',
        alpha=0.7
    )

# Plot the cluster centers
plt.scatter(
    V[:, 0],  # Cluster center feature 1
    V[:, 1],  # Cluster center feature 2
    s=200,  # Size of the markers
    c='red',  # Color
    marker='X',  # Marker style
    label='Cluster Centers'
)

# Add labels and legend
plt.title('After Clustering: Fuzzy C-Means Results')  # Label for clustered data
plt.xlabel('Feature 1 (Sepal Length)')
plt.ylabel('Feature 2 (Sepal Width)')
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Model Evaluation
def fuzzy_partition_coefficient(U):
    """Calculate the Fuzzy Partition Coefficient (FPC)."""
    return np.mean(U**2)

def fuzzy_partition_entropy(U):
    """Calculate the Fuzzy Partition Entropy (FPE)."""
    return -np.mean(U * np.log(U))

# Calculate evaluation metrics
fpc = fuzzy_partition_coefficient(U)
fpe = fuzzy_partition_entropy(U)

print("Model Evaluation:")
print(f"Fuzzy Partition Coefficient (FPC): {fpc:.4f}")
print(f"Fuzzy Partition Entropy (FPE): {fpe:.4f}")

# Interpretation of metrics
print("\nInterpretation:")
print("- FPC ranges from 0 to 1. A value closer to 1 indicates better clustering.")
print("- FPE ranges from 0 to infinity. A value closer to 0 indicates better clustering.")