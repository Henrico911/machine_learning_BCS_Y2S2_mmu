import numpy as np

class Logistic_Regression():    
    
    # 1. initiate the parameters for the logistic regression model
    def __init__(self, learning_rate, no_of_iterations):
        
        # declaring learning rate $ number of iterations
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        
        
        
    # 2. fit the dataset into our model
    def fit(self, X, Y):
        
        # number of data points in the dataset (number of rows) --> m
        # number of input features in the dataset (number of columns) --> n
        self.m, self.n = X.shape
        
        # initiating weight value and bias value        
        self.w = np.zeros(self.n)
        
        self.b = 0
        
        self.X = X
        
        self.Y = Y
        
        # implementing gradient descent for optimization
        for i in range(self.no_of_iterations):
            self.update_weights()
        
        
        
    # 3. update weight and bias : Gradient Descent 
    def update_weights(self):
        
        # Y hat formula (sigmoid fx)
        z = self.X.dot(self.w) + self.b      # z = wX +  b
        Y_hat = 1 / (1 + np.exp( - z )) 
    
        #derivatives    
        dw = (1 / self.m) * np.dot(self.X.T, (Y_hat - self.Y))

        db = (1 / self.m) * np.sum(Y_hat - self.Y)

        # 3. updating weight and bias using Gradient Descent 
        self.w = self.w - self.learning_rate * dw

        self.b = self.b - self.learning_rate * db
    
    
    
        
    # 4. make predictions
    def predict(self, X):
        #sigmoid equation & decision boundary
        Y_pred = 1 / (1 + np.exp( - ( X.dot(self.w) + self.b ) ))
        Y_pred = np.where( Y_pred > 0.5, 1, 0)
        return Y_pred
        
        
        
        
