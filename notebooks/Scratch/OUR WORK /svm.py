import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the Iris dataset
def load_iris_data(filepath):
    """Load and prepare the Iris dataset."""
    # Read the data
    data = pd.read_csv(filepath)
    
    # Extract features (all columns except Id and Species)
    X = data.iloc[:, 1:5].values
    
    # Extract target and convert to numerical values
    # For SVM implementation, we'll convert to binary labels (-1, 1)
    # We'll first handle a binary classification problem: setosa vs non-setosa
    y = data.iloc[:, 5].values
    y_binary = np.where(y == 'Iris-setosa', 1, -1)
    
    return X, y_binary, y

# Kernel functions as described in the SVM document
def linear_kernel(x1, x2):
    """Compute linear kernel: K(x_i, x_j) = x_i · x_j"""
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3, c=1.0):
    """Compute polynomial kernel: K(x_i, x_j) = (x_i · x_j + c)^d"""
    return (np.dot(x1, x2) + c) ** degree

def rbf_kernel(x1, x2, gamma=0.1):
    """Compute RBF kernel: K(x_i, x_j) = exp(-γ||x_i - x_j||²)"""
    return np.exp(-gamma * np.sum((x1 - x2) ** 2))

class SVM:
    def __init__(self, kernel=linear_kernel, C=1.0, tol=1e-3, max_iter=100):
        """
        Implementation of Support Vector Machine using SMO algorithm
        
        Parameters:
        kernel: the kernel function to use
        C: regularization parameter (for soft margin)
        tol: tolerance for stopping criterion
        max_iter: maximum number of iterations
        """
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_indices = None
        
    def fit(self, X, y):
        """
        Fit the SVM model according to the training data using SMO algorithm
        
        Parameters:
        X: training features
        y: training labels (-1, 1)
        """
        n_samples, n_features = X.shape
        
        # Initialize alphas (Lagrange multipliers)
        self.alpha = np.zeros(n_samples)
        
        # Precompute kernel matrix for efficiency
        self.kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                self.kernel_matrix[i, j] = self.kernel(X[i], X[j])
        
        # Simplified SMO algorithm
        iter_count = 0
        while iter_count < self.max_iter:
            alpha_changed = 0
            
            for i in range(n_samples):
                # Calculate error for sample i
                E_i = self._decision_function_internal(i, X, y) - y[i]
                
                # Check if example violates KKT conditions
                if ((y[i] * E_i < -self.tol and self.alpha[i] < self.C) or 
                    (y[i] * E_i > self.tol and self.alpha[i] > 0)):
                    
                    # Randomly select second sample j ≠ i
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    # Calculate error for sample j
                    E_j = self._decision_function_internal(j, X, y) - y[j]
                    
                    # Save old alphas
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # Compute bounds L and H
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2.0 * self.kernel_matrix[i, j] - self.kernel_matrix[i, i] - self.kernel_matrix[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    self.alpha[j] = alpha_j_old - (y[j] * (E_i - E_j)) / eta
                    
                    # Clip alpha_j
                    self.alpha[j] = min(H, self.alpha[j])
                    self.alpha[j] = max(L, self.alpha[j])
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    self.alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # Compute b
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * self.kernel_matrix[i, i] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * self.kernel_matrix[i, j]
                    
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * self.kernel_matrix[i, j] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * self.kernel_matrix[j, j]
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    alpha_changed += 1
            
            if alpha_changed == 0:
                iter_count += 1
            else:
                iter_count = 0
        
        # Extract support vectors
        sv_indices = np.where(self.alpha > 1e-5)[0]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.support_vector_indices = sv_indices
        self.alpha = self.alpha[sv_indices]
        
        print(f"Number of support vectors: {len(self.support_vectors)}")
        
        return self
    
    def _decision_function_internal(self, i, X, y):
        """Internal decision function used during training"""
        return np.sum(self.alpha * y * self.kernel_matrix[i]) + self.b
    
    def decision_function(self, X):
        """
        Compute the decision function for samples in X
        
        Parameters:
        X: test samples
        
        Returns:
        Array of decisions (scores)
        """
        n_samples = X.shape[0]
        decision = np.zeros(n_samples)
        
        for i in range(n_samples):
            decision[i] = self.b
            for alpha, sv, sv_y in zip(self.alpha, self.support_vectors, self.support_vector_labels):
                decision[i] += alpha * sv_y * self.kernel(X[i], sv)
        
        return decision
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        X: test samples
        
        Returns:
        Array of predicted class labels (-1 or 1)
        """
        return np.sign(self.decision_function(X))

# One-vs-Rest SVM for multiclass classification
class OneVsRestSVM:
    def __init__(self, kernel=linear_kernel, C=1.0):
        """
        One-vs-Rest SVM for multiclass classification
        
        Parameters:
        kernel: the kernel function to use
        C: regularization parameter
        """
        self.kernel = kernel
        self.C = C
        self.models = {}
        self.classes = None
    
    def fit(self, X, y):
        """
        Fit One-vs-Rest SVM model
        
        Parameters:
        X: training features
        y: training labels (can be multiclass)
        """
        self.classes = np.unique(y)
        
        # Train one SVM for each class
        for cls in self.classes:
            print(f"Training SVM for class: {cls}")
            # Create binary labels (1 for current class, -1 for rest)
            y_binary = np.where(y == cls, 1, -1)
            
            # Create and train SVM model
            svm = SVM(kernel=self.kernel, C=self.C)
            svm.fit(X, y_binary)
            
            # Store the model
            self.models[cls] = svm
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        X: test samples
        
        Returns:
        Array of predicted class labels
        """
        n_samples = X.shape[0]
        # Decision matrix: rows = samples, columns = classes
        decision = np.zeros((n_samples, len(self.classes)))
        
        # Calculate decision scores for each model
        for i, cls in enumerate(self.classes):
            decision[:, i] = self.models[cls].decision_function(X)
        
        # Return class with highest decision score
        return self.classes[np.argmax(decision, axis=1)]

# Function to evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Evaluate model performance on both training and test sets
    
    Parameters:
    model: trained SVM model
    X_train, X_test: training and test features
    y_train, y_test: training and test labels
    model_name: name of the model for printing
    """
    # Evaluate on training data
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Evaluate on test data
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"{model_name} - Training Accuracy: {train_accuracy:.4f}")
    print(f"{model_name} - Test Accuracy: {test_accuracy:.4f}")
    
    return y_test_pred

# Function to visualize decision boundaries for 2D data
def plot_decision_boundary(model, X_train, X_test, y_train, y_test, title="Decision Boundary"):
    """
    Plot decision boundary for a 2D SVM model and show train/test data
    
    Parameters:
    model: trained SVM model
    X_train, X_test: training and test features
    y_train, y_test: training and test labels
    title: plot title
    """
    # Combine data for mesh grid boundaries
    X_combined = np.vstack((X_train, X_test))
    
    # Create a mesh grid
    x_min, x_max = X_combined[:, 0].min() - 1, X_combined[:, 0].max() + 1
    y_min, y_max = X_combined[:, 1].min() - 1, X_combined[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Get predictions for all mesh grid points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and points
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.3)
    
    # Plot training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', 
                marker='o', s=80, alpha=0.7, label='Training data')
    
    # Plot test points with different marker
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k',
                marker='^', s=80, alpha=0.7, label='Test data')
    
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# Main function
def main():
    # Load the Iris dataset
    X, y_binary, y_original = load_iris_data("Iris.csv")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train_binary, y_test_binary = train_test_split(
        X_scaled, y_binary, test_size=0.3, random_state=42)
    
    # For multiclass classification
    _, _, y_train_multi, y_test_multi = train_test_split(
        X_scaled, y_original, test_size=0.3, random_state=42)
    
    print("==== Binary Classification: Setosa vs Non-Setosa ====")
    
    # Train SVM model with linear kernel
    print("Training SVM with Linear Kernel...")
    svm_linear = SVM(kernel=linear_kernel, C=1.0)
    svm_linear.fit(X_train, y_train_binary)
    
    # Evaluate the model on both training and test data
    print("\nEvaluating Linear Kernel SVM:")
    y_pred_linear = evaluate_model(svm_linear, X_train, X_test, 
                                 y_train_binary, y_test_binary, 
                                 "Linear Kernel SVM")
    
    # Train SVM model with RBF kernel
    print("\nTraining SVM with RBF Kernel...")
    svm_rbf = SVM(kernel=rbf_kernel, C=1.0)
    svm_rbf.fit(X_train, y_train_binary)
    
    # Evaluate the model on both training and test data
    print("\nEvaluating RBF Kernel SVM:")
    y_pred_rbf = evaluate_model(svm_rbf, X_train, X_test, 
                               y_train_binary, y_test_binary, 
                               "RBF Kernel SVM")
    
    # For visualization, let's use only the first two features
    X_2d = X_scaled[:, :2]
    X_train_2d, X_test_2d, y_train_binary_2d, y_test_binary_2d = train_test_split(
        X_2d, y_binary, test_size=0.3, random_state=42)
    
    # Train 2D model for visualization
    print("\nTraining 2D SVM for Visualization...")
    svm_2d = SVM(kernel=linear_kernel, C=1.0)
    svm_2d.fit(X_train_2d, y_train_binary_2d)
    
    # Evaluate 2D model
    print("\nEvaluating 2D SVM:")
    y_pred_2d = evaluate_model(svm_2d, X_train_2d, X_test_2d, 
                              y_train_binary_2d, y_test_binary_2d, 
                              "2D Linear Kernel SVM")
    
    # Plot decision boundary showing both training and test data
    plot_decision_boundary(svm_2d, X_train_2d, X_test_2d, 
                          y_train_binary_2d, y_test_binary_2d, 
                          "SVM Decision Boundary (Linear Kernel) - Training and Test Data")
    
    print("\n==== Multiclass Classification: All Iris Species ====")
    
    # Train One-vs-Rest SVM for multiclass classification
    print("Training One-vs-Rest SVM with RBF Kernel...")
    ovr_svm = OneVsRestSVM(kernel=rbf_kernel, C=1.0)
    ovr_svm.fit(X_train, y_train_multi)
    
    # Evaluate the multiclass model on both training and test data
    print("\nEvaluating One-vs-Rest SVM:")
    # Training accuracy
    y_train_pred_multi = ovr_svm.predict(X_train)
    train_accuracy_multi = accuracy_score(y_train_multi, y_train_pred_multi)
    print(f"One-vs-Rest SVM - Training Accuracy: {train_accuracy_multi:.4f}")
    
    # Test accuracy
    y_test_pred_multi = ovr_svm.predict(X_test)
    test_accuracy_multi = accuracy_score(y_test_multi, y_test_pred_multi)
    print(f"One-vs-Rest SVM - Test Accuracy: {test_accuracy_multi:.4f}")
    
    # Confusion matrix for test data
    cm = confusion_matrix(y_test_multi, y_test_pred_multi)
    print("\nConfusion Matrix (Test Data):")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = np.unique(y_original)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations in the confusion matrix
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    print("\nModel Evaluation Complete!")

if __name__ == "__main__":
    main()