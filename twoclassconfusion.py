import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load breast cancer dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train a logistic regression model (you can use any classification model)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, predictions)

# Display confusion matrix
print("Confusion Matrix:")
print(cm)

# Display confusion matrix as a basic plot
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels (0: Malignant, 1: Benign)')
plt.ylabel('Actual Labels (0: Malignant, 1: Benign)')
plt.xticks(np.arange(2), ['Malignant', 'Benign'])
plt.yticks(np.arange(2), ['Malignant', 'Benign'])

# Display values in the plot
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

plt.show()
