import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
np.random.seed(42)
X = np.random.normal(0, 1, 100)
y = (X + np.random.normal(0, 0.1, 100) > 0).astype(int)
X = X.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
plt.scatter(X, y, label='Data Points')
x_values = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
y_probabilities = model.predict_proba(x_values)[:, 1]
plt.plot(x_values, y_probabilities, color='red', label='Decision Boundary (Probability)')
plt.xlabel('X')
plt.ylabel('Probability')
plt.legend()
plt.show()
