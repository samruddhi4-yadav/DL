
# Practical 4: Anomaly Detection Using Autoencoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# Step 1: Import Libraries and Load ECG Dataset
# -------------------------------------------------

url = "https://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"

df = pd.read_csv(url, header=None)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# Separate ECG features and labels
data = df.iloc[:, :-1].values.astype("float32")
labels = df.iloc[:, -1].values

# In this dataset, label 1 represents normal ECG
normal_data = data[labels == 1]
abnormal_data = data[labels != 1]

print("\nNormal ECG samples:", len(normal_data))
print("Abnormal ECG samples:", len(abnormal_data))

# -------------------------------------------------
# Step 2: Preprocess the Data
# -------------------------------------------------

# Split normal ECG samples into training and testing data
X_train, X_normal_test = train_test_split(
    normal_data,
    test_size=0.2,
    random_state=42
)

# Include abnormal samples in the test data
X_test = np.concatenate(
    [X_normal_test, abnormal_data],
    axis=0
)

# Create actual labels for evaluation
y_test = np.concatenate([
    np.zeros(len(X_normal_test)),  # Normal = 0
    np.ones(len(abnormal_data))    # Anomaly = 1
])

# Normalize using training data statistics
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# -------------------------------------------------
# Step 3: Build the Autoencoder
# -------------------------------------------------

input_dim = X_train.shape[1]

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation="relu")(input_layer)
encoded = Dense(16, activation="relu")(encoded)
encoded = Dense(8, activation="relu")(encoded)

# Decoder
decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(32, activation="relu")(decoded)
decoded = Dense(input_dim, activation="linear")(decoded)

# Create the autoencoder model
autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(
    optimizer="adam",
    loss="mae",
    metrics=["mae"]
)

print("\nAutoencoder Architecture:")
autoencoder.summary()

# -------------------------------------------------
# Step 4: Train the Autoencoder
# -------------------------------------------------

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)

# -------------------------------------------------
# Step 5: Calculate Reconstruction Error
# -------------------------------------------------

# Reconstruct the test data
X_reconstructed = autoencoder.predict(X_test, verbose=0)

# Calculate mean absolute reconstruction error
reconstruction_error = np.mean(
    np.abs(X_test - X_reconstructed),
    axis=1
)

# Calculate threshold using normal training data
train_reconstructed = autoencoder.predict(X_train, verbose=0)

train_error = np.mean(
    np.abs(X_train - train_reconstructed),
    axis=1
)

threshold = train_error.mean() + train_error.std()

print("\nAnomaly Detection Threshold:", threshold)

# Error above threshold means anomaly
predictions = (reconstruction_error > threshold).astype(int)

# -------------------------------------------------
# Step 6: Evaluate the Model
# -------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Normal", "Anomaly"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# -------------------------------------------------
# Step 7: Plot Training Loss
# -------------------------------------------------

plt.figure(figsize=(8, 4))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.show()

# -------------------------------------------------
# Step 8: Plot Reconstruction Errors
# -------------------------------------------------

plt.figure(figsize=(8, 4))

plt.hist(
    reconstruction_error[y_test == 0],
    bins=40,
    alpha=0.7,
    label="Normal ECG"
)

plt.hist(
    reconstruction_error[y_test == 1],
    bins=40,
    alpha=0.7,
    label="Abnormal ECG"
)

plt.axvline(
    threshold,
    linestyle="--",
    label="Anomaly Threshold"
)

plt.title("ECG Anomaly Detection")
plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Samples")
plt.legend()
plt.show()

print("\nECG Anomaly Detection Completed!")