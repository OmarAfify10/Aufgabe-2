# train_time.py

import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from decorators import my_logger, my_timer
from Unit_download import Normalize
from the_algorithm import TheAlgorithm
import time

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target'].astype(np.int8)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = TheAlgorithm(X_train, y_train, X_test, y_test)

# Time the training process
start_time = time.time()
model.fit()
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print(f'Training Time: {training_time} seconds')

# Save the training time to a file
joblib.dump(training_time, 'training_time.pkl')
print("Training time saved to 'training_time.pkl'")
