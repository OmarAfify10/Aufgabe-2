# train_model.py

import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from decorators import my_logger, my_timer
from Unit_download import Normalize
from the_algorithm import TheAlgorithm


# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target'].astype(np.int8)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = TheAlgorithm(X_train, y_train, X_test, y_test)
# train_accuracy = model.fit()

# Initialize and train the algorithm
model = TheAlgorithm(X_train, y_train, X_test, y_test)
train_accuracy = model.fit()
print(f'Training Accuracy: {train_accuracy}%')

# Save metrics
joblib.dump({
    'train_accuracy': train_accuracy,
    'train_confusion_matrix': model.train_confusion_matrix
}, 'train_metrics.pkl')

print("Training metrics saved to 'train_metrics.pkl'")
