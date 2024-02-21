import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Actual labels
actual_labels = np.array(['Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog', 'Not Dog'])

# Predicted labels
predicted_labels = np.array(['Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog', 'Not Dog'])

# Create confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels, labels=['Dog', 'Not Dog'])

# Display confusion matrix as a basic plot
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.xticks(np.arange(len(['Dog', 'Not Dog'])), ['Dog', 'Not Dog'])
plt.yticks(np.arange(len(['Dog', 'Not Dog'])), ['Dog', 'Not Dog'])

# Display values in the plot
for i in range(len(['Dog', 'Not Dog'])):
    for j in range(len(['Dog', 'Not Dog'])):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

plt.show()
