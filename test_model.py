# test_model.py

import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from decorators import my_logger, my_timer
from Unit_download import Normalize
from the_algorithm import TheAlgorithm

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target'].astype(np.int8)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the algorithm
model = TheAlgorithm(X_train, y_train, X_test, y_test)
train_accuracy = model.fit()
print(f'Training Accuracy: {train_accuracy}%')

# Load saved metrics
saved_metrics = joblib.load('train_metrics.pkl')
saved_accuracy = saved_metrics['train_accuracy']
saved_confusion_matrix = saved_metrics['train_confusion_matrix']

# Test if metrics match
if np.isclose(train_accuracy, saved_accuracy):
    print("Training accuracy matches the saved accuracy.")
else:
    print("Training accuracy does not match the saved accuracy.")

if np.array_equal(model.train_confusion_matrix, saved_confusion_matrix):
    print("Confusion matrix matches the saved matrix.")
else:
    print("Confusion matrix does not match the saved matrix.")
