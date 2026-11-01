# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 20:45:22 2026

@author: SAMRUDDHI
"""

# Practical 3: Image Classification using CNN

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# Stage 1: Loading and Preprocessing Image Data
# -------------------------------------------------

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Normalize pixel values between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add channel dimension for grayscale images
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Display sample images
plt.figure(figsize=(8, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i].squeeze(), cmap="gray")
    plt.title("Digit: " + str(y_train[i]))
    plt.axis("off")

plt.tight_layout()
plt.show()

# -------------------------------------------------
# Stage 2: Defining the CNN Model Architecture
# -------------------------------------------------

model = Sequential([
    Conv2D(32, (3, 3), activation="relu",
           input_shape=(28, 28, 1)),

    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(100, activation="relu"),

    Dense(10, activation="softmax")
])

# Compile the model using SGD optimizer
optimizer = SGD(learning_rate=0.01, momentum=0.9)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model architecture
model.summary()

# -------------------------------------------------
# Stage 3: Training the Model
# -------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# -------------------------------------------------
# Stage 4: Estimating Model Performance
# -------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")

# Predict digits from test images
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy Score:", accuracy * 100, "%")

# Plot training and validation accuracy
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------------------------
# Display Predictions
# -------------------------------------------------

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap="gray")
    plt.title("Actual: " + str(y_test[i]) +
              "\nPredicted: " + str(y_pred[i]))
    plt.axis("off")

plt.tight_layout()
plt.show()

# Save the trained model
model.save("practical3_cnn.keras")

print("\nModel saved successfully!")
