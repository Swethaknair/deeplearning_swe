from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
texts = ["I love programming", "Machine learning is fascinating", "Python is a versatile language",
         "I dislike bugs in my code", "Data science is the future", "Programming is fun"]

labels = ["positive", "positive", "positive", "negative", "positive", "positive"]
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_vectorized, y_train)
predictions = naive_bayes_classifier.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy}")

