import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, theta):
    return sigmoid(np.dot(X, theta))

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # Convert to binary classification (0 or 1)

# Add a bias term to the feature matrix
X = np.c_[np.ones(X.shape[0]), X]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize parameters
theta = np.zeros(X_train.shape[1])

# Set learning rate and number of iterations
learning_rate = 0.01
num_iterations = 1000

# Gradient Descent
for _ in range(num_iterations):
    predictions = logistic_regression(X_train, theta)
    errors = predictions - y_train
    gradient = np.dot(X_train.T, errors) / len(y_train)
    theta -= learning_rate * gradient

# Make predictions on the test data
test_predictions = (logistic_regression(X_test, theta) >= 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy:", accuracy)

# Print classification report for detailed metrics
print("Classification Report:")
print(classification_report(y_test, test_predictions))
