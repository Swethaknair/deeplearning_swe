# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Initialize a Random Forest classifier
random_forest_classifier = RandomForestClassifier()

# Specify the number of folds for cross-validation
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform cross-validation
cross_val_results = cross_val_score(random_forest_classifier, X, y, cv=kfold)

# Plot the cross-validation results
plt.figure(figsize=(8, 5))
plt.bar(np.arange(1, num_folds + 1), cross_val_results, color='skyblue')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Results')
plt.ylim([0, 1])
plt.show()
