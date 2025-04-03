import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Set basic style for visualizations
plt.style.use('ggplot')

# Sigmoid function
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))

# Logistic Regression Model
class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_=0.0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_ = lambda_
        self.theta = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Add intercept term
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        
        # Initialize parameters
        self.theta = np.zeros(X_b.shape[1])
        
        # Gradient descent
        for i in range(self.num_iterations):
            y_pred = sigmoid(X_b @ self.theta)
            error = y_pred - y
            gradient = (1/len(y)) * (X_b.T @ error)
            gradient[1:] += (self.lambda_ / len(y)) * self.theta[1:]
            self.theta -= self.learning_rate * gradient
            
            if i % 500 == 0:
                print(f"Iteration {i}: Training in progress...")
        
        return self
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        return sigmoid(X_b @ self.theta)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

# Data loading and preprocessing function
def prepare_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype.kind in 'ifc':  # numeric
                df[col] = df[col].fillna(df[col].median())
            else:  # categorical
                df[col] = df[col].fillna(df[col].mode()[0])
    
    # Check for target column
    target_column = 'Heart Attack Risk'
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Extract features and target
    y = df[target_column].values
    X = df.drop(target_column, axis=1)
    feature_names = X.columns
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, feature_names

# Function to evaluate model and detect overfitting/underfitting
def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names):
    # Get predictions
    y_train_prob = model.predict_proba(X_train)
    y_test_prob = model.predict_proba(X_test)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Print some test predictions to examine
    print("\n=== Test Data Predictions (First 10 samples) ===")
    print("True Values | Predicted | Probability")
    print("-" * 40)
    for i in range(min(10, len(y_test))):
        print(f"{y_test[i]:^10} | {y_test_pred[i]:^9} | {y_test_prob[i]:.4f}")
    
    # Calculate metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    train_values = [
        accuracy_score(y_train, y_train_pred),
        precision_score(y_train, y_train_pred),
        recall_score(y_train, y_train_pred),
        f1_score(y_train, y_train_pred)
    ]
    
    test_values = [
        accuracy_score(y_test, y_test_pred),
        precision_score(y_test, y_test_pred),
        recall_score(y_test, y_test_pred),
        f1_score(y_test, y_test_pred)
    ]
    
    # Print metrics
    print("\n=== Performance Comparison ===")
    print(f"{'Metric':<10} {'Training':<10} {'Testing':<10} {'Gap':<10}")
    print("-" * 40)
    for i, metric in enumerate(metrics):
        gap = train_values[i] - test_values[i]
        print(f"{metric:<10} {train_values[i]:.4f}     {test_values[i]:.4f}     {gap:.4f}")
    
    # Calculate average metrics for detecting overfitting/underfitting
    avg_train = sum(train_values) / len(train_values)
    avg_test = sum(test_values) / len(test_values)
    avg_gap = avg_train - avg_test
    
    # Detect fitting status
    if avg_gap > 0.1:
        fitting_status = "Overfitting"
    elif avg_test < 0.6:  # Poor performance on both training and testing
        fitting_status = "Underfitting"
    elif avg_gap > 0.05:
        fitting_status = "Slight Overfitting"
    elif avg_test < 0.7:
        fitting_status = "Slight Underfitting"
    else:
        fitting_status = "Good Fit"
    
    print(f"\n=== Model Fitting Status: {fitting_status} ===")
    print(f"Average train score: {avg_train:.4f}")
    print(f"Average test score: {avg_test:.4f}")
    print(f"Performance gap: {avg_gap:.4f}")
    
    # Create visualizations
    # 1. Metrics comparison bar chart
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Bar chart for metrics
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, train_values, width, label='Training')
    ax1.bar(x + width/2, test_values, width, label='Testing')
    
    ax1.set_ylabel('Score')
    ax1.set_title('Training vs Testing Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Confusion Matrix for test data
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'], ax=ax2)
    ax2.set_title('Testing Confusion Matrix')
    
    # 3. ROC Curve
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    roc_auc_test = auc(fpr_test, tpr_test)
    
    ax3.plot(fpr_train, tpr_train, lw=2, label=f'Training (AUC = {roc_auc_train:.2f})')
    ax3.plot(fpr_test, tpr_test, lw=2, label=f'Testing (AUC = {roc_auc_test:.2f})')
    ax3.plot([0, 1], [0, 1], 'k--', lw=2)
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve Comparison')
    ax3.legend(loc="lower right")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 4. Feature importance
    plt.figure(figsize=(10, 6))
    
    # Skip the intercept term
    coeffs = model.theta[1:]
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coeffs,
        'Absolute Value': np.abs(coeffs)
    })
    
    # Sort by absolute value
    importance_df = importance_df.sort_values('Absolute Value', ascending=False).head(10)
    
    # Plot
    bars = plt.barh(importance_df['Feature'], importance_df['Coefficient'])
    
    # Color bars based on coefficient sign
    for i, bar in enumerate(bars):
        if importance_df['Coefficient'].iloc[i] < 0:
            bar.set_color('red')
        else:
            bar.set_color('green')
            
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.show()
    
    return fitting_status, avg_train, avg_test, avg_gap

# Function to train and auto-tune model
def train_and_tune(X_train, X_test, y_train, y_test, feature_names, max_iterations=3):
    # Initial parameters
    learning_rate = 0.01
    num_iterations = 1000
    lambda_ = 0.1
    
    for iteration in range(max_iterations):
        print(f"\n=== Training Model (Iteration {iteration+1}) ===")
        print(f"Parameters: learning_rate={learning_rate}, iterations={num_iterations}, lambda={lambda_}")
        
        # Train model
        model = SimpleLogisticRegression(learning_rate=learning_rate, num_iterations=num_iterations, lambda_=lambda_)
        model.fit(X_train, y_train)
        
        # Evaluate model
        fitting_status, avg_train, avg_test, gap = evaluate_model(model, X_train, y_train, X_test, y_test, feature_names)
        
        # Check if model is already good
        if fitting_status == "Good Fit":
            print("\n=== Model is well tuned! ===")
            return model, fitting_status
            
        # Auto-tune based on fitting status
        if "Overfitting" in fitting_status:
            # Increase regularization and reduce complexity
            lambda_ *= 2
            print(f"\n=== Detected {fitting_status} ===")
            print(f"Tuning strategy: Increasing regularization to lambda={lambda_}")
            
        elif "Underfitting" in fitting_status:
            # Reduce regularization and increase complexity
            lambda_ = max(lambda_ / 2, 0.01)
            num_iterations = min(num_iterations * 2, 5000)
            print(f"\n=== Detected {fitting_status} ===")
            print(f"Tuning strategy: Decreasing regularization to lambda={lambda_} and increasing iterations to {num_iterations}")
        
        # If this is the last iteration and we haven't found a good fit
        if iteration == max_iterations - 1:
            print("\n=== Maximum tuning iterations reached ===")
            return model, fitting_status
    
    return model, fitting_status

# Main function
def main(file_path):
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data(file_path)
    
    # Train and auto-tune model
    final_model, fitting_status = train_and_tune(X_train, X_test, y_train, y_test, feature_names)
    
    print("\n=== Final Model Summary ===")
    print(f"Fitting Status: {fitting_status}")
    print(f"Regularization (lambda): {final_model.lambda_}")
    print(f"Number of features: {len(feature_names)}")
    
    return final_model

# Run the program
if __name__ == "__main__":
    file_path = "heart_attack.csv"  # Replace with your file path
    model = main(file_path)