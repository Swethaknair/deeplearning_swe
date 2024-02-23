from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
predictions = svm_classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test,predictions )
accuracy = accuracy_score(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

print(f"Accuracy: {accuracy}")

