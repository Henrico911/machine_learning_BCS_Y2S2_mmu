import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Load the dataset
def load_data(file_path):
    """
    Load the dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    df = pd.read_csv("heart_attack.csv")
    return df

# Data preprocessing
def preprocess_data(df):
    """
    Preprocess the dataset for logistic regression.
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        tuple: X (features), y (target), feature_names
    """
    # Make a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Drop non-relevant columns
    if 'Patient ID' in data.columns:
        data = data.drop('Patient ID', axis=1)
    
    # Check for the target column
    target_column = 'Heart Attack Risk'
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Convert categorical variables to numerical using one-hot encoding
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Extract target variable
    y = data[target_column].values
    X = data.drop(target_column, axis=1)
    feature_names = X.columns
    X = X.values
    
    return X, y, feature_names

# Sigmoid function (logistic function)
def sigmoid(z):
    """
    Sigmoid activation function.
    
    Args:
        z (numpy.ndarray): Input values
        
    Returns:
        numpy.ndarray: Output values between 0 and 1
    """
    # Clip z to avoid overflow in exp(-z)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Cost function (Log Loss or Binary Cross-Entropy)
def compute_cost(X, y, theta):
    """
    Compute the cost function for logistic regression.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector
        theta (numpy.ndarray): Model parameters
        
    Returns:
        float: The cost value
    """
    m = len(y)
    h = sigmoid(X @ theta)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    
    cost = -1/m * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
    return cost

# Gradient of the cost function
def compute_gradient(X, y, theta):
    """
    Compute the gradient of the cost function.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector
        theta (numpy.ndarray): Model parameters
        
    Returns:
        numpy.ndarray: The gradient vector
    """
    m = len(y)
    h = sigmoid(X @ theta)
    gradient = 1/m * (X.T @ (h - y))
    return gradient

# Gradient Descent
def gradient_descent(X, y, theta, alpha, num_iterations):
    """
    Perform gradient descent to optimize the parameters.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector
        theta (numpy.ndarray): Initial model parameters
        alpha (float): Learning rate
        num_iterations (int): Number of iterations
        
    Returns:
        tuple: Optimized parameters and cost history
    """
    cost_history = []
    
    for i in range(num_iterations):
        gradient = compute_gradient(X, y, theta)
        theta = theta - alpha * gradient
        
        # Compute cost every 100 iterations to save computation time
        if i % 100 == 0:
            cost = compute_cost(X, y, theta)
            cost_history.append(cost)
            
    return theta, cost_history

# Prediction function
def predict(X, theta, threshold=0.5):
    """
    Make binary predictions using the trained model.
    
    Args:
        X (numpy.ndarray): Feature matrix
        theta (numpy.ndarray): Model parameters
        threshold (float): Probability threshold for classification
        
    Returns:
        numpy.ndarray: Predicted binary labels
    """
    probabilities = sigmoid(X @ theta)
    return (probabilities >= threshold).astype(int)

# Model evaluation
def evaluate_model(X, y, theta):
    """
    Evaluate the logistic regression model.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): True labels
        theta (numpy.ndarray): Model parameters
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    y_pred = predict(X, theta)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics

# Logistic Regression Model Class
class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch.
    """
    
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the model.
        
        Args:
            learning_rate (float): Learning rate for gradient descent
            num_iterations (int): Number of gradient descent iterations
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.cost_history = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """
        Train the logistic regression model.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
            
        Returns:
            self: The trained model
        """
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Add intercept term
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        
        # Initialize parameters
        self.theta = np.zeros(X_b.shape[1])
        
        # Optimize parameters using gradient descent
        self.theta, self.cost_history = gradient_descent(
            X_b, y, self.theta, self.learning_rate, self.num_iterations
        )
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probabilities using the trained model.
        
        Args:
            X (numpy.ndarray): Feature matrix
            
        Returns:
            numpy.ndarray: Predicted probabilities
        """
        X_scaled = self.scaler.transform(X)
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        return sigmoid(X_b @ self.theta)
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions using the trained model.
        
        Args:
            X (numpy.ndarray): Feature matrix
            threshold (float): Probability threshold for classification
            
        Returns:
            numpy.ndarray: Predicted binary labels
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): True labels
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        X_scaled = self.scaler.transform(X)
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        return evaluate_model(X_b, y, self.theta)

# Visualize the cost history
def plot_cost_history(cost_history):
    """
    Plot the cost function over iterations.
    
    Args:
        cost_history (list): List of cost values over iterations
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(cost_history)) * 100, cost_history)
    plt.title('Cost Function Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    """
    Plot the confusion matrix.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Plot ROC curve
def plot_roc_curve(y_true, y_prob):
    """
    Plot the ROC curve.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_prob (numpy.ndarray): Predicted probabilities
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Plot feature importance
def plot_feature_importance(feature_names, theta):
    """
    Plot the feature importance.
    
    Args:
        feature_names (list): Names of the features
        theta (numpy.ndarray): Model parameters
    """
    # Skip the intercept term
    feature_importance = np.abs(theta[1:])
    feature_names = list(feature_names)
    
    # Sort by importance
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title('Feature Importance')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.show()

# Data visualization
def visualize_data(df):
    """
    Visualize the dataset.
    
    Args:
        df (pandas.DataFrame): The dataset
    """
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Target distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Heart Attack Risk', data=df)
    plt.title('Heart Attack Risk Distribution')
    plt.xlabel('Heart Attack Risk')
    plt.ylabel('Count')
    plt.show()
    
    # Age vs. Heart Attack Risk
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Heart Attack Risk', y='Age', data=df)
    plt.title('Age vs. Heart Attack Risk')
    plt.show()
    
    # Cholesterol vs. Heart Attack Risk
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Heart Attack Risk', y='Cholesterol', data=df)
    plt.title('Cholesterol vs. Heart Attack Risk')
    plt.show()
    
    # Blood Pressure vs. Heart Attack Risk
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Heart Attack Risk', y='Blood Pressure', data=df)
    plt.title('Blood Pressure vs. Heart Attack Risk')
    plt.show()

# Main function
def main():
    # Load the dataset
    file_path = 'heart_rate.csv'  # Update with your file path
    df = load_data(file_path)
    
    # Display basic information about the dataset
    print("Dataset Overview:")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Visualize the data
    print("\nVisualizing the data...")
    visualize_data(df)
    
    # Preprocess the data
    X, y, feature_names = preprocess_data(df)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Train the logistic regression model
    print("\nTraining the logistic regression model...")
    model = LogisticRegression(learning_rate=0.01, num_iterations=10000)
    model.fit(X_train, y_train)
    
    # Plot the cost history
    print("\nPlotting the cost history...")
    plot_cost_history(model.cost_history)
    
    # Evaluate the model on the test set
    print("\nEvaluating the model...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Plot confusion matrix
    print("\nPlotting the confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)
    
    # Plot ROC curve
    print("\nPlotting the ROC curve...")
    plot_roc_curve(y_test, y_prob)
    
    # Plot feature importance
    print("\nPlotting feature importance...")
    plot_feature_importance(feature_names, model.theta)
    
if __name__ == "__main__":
    main()