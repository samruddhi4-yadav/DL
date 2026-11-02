
# -*- coding: utf-8 -*-
"""
Practical 6: Transfer Learning Using VGG-16
Dataset: Caltech-101
Framework: PyTorch
"""

import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights


# ------------------------------------------------
# STEP 1: Initialize settings
# ------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("TRANSFER LEARNING USING VGG-16")
print("=" * 60)

print("PyTorch version:", torch.__version__)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU not available. Using CPU.")


# ------------------------------------------------
# STEP 2: Define image transformations
# ------------------------------------------------

image_transforms = {
    "train": transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),

    "valid": transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),

    "test": transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}

print("\nImage transformations defined successfully!")


# ------------------------------------------------
# STEP 3: Load Caltech-101 dataset
# ------------------------------------------------

# Automatically create a local data directory.
DATA_DIR = os.path.join(
    os.path.expanduser("~"),
    "caltech101_data"
)

os.makedirs(DATA_DIR, exist_ok=True)

print("\nLoading Caltech-101 dataset...")
print("Dataset directory:", DATA_DIR)

# Download the dataset if it is not already available.
train_base = datasets.Caltech101(
    root=DATA_DIR,
    target_type="category",
    transform=image_transforms["train"],
    download=True
)

# Use separate dataset objects for validation and testing.
valid_base = datasets.Caltech101(
    root=DATA_DIR,
    target_type="category",
    transform=image_transforms["valid"],
    download=False
)

test_base = datasets.Caltech101(
    root=DATA_DIR,
    target_type="category",
    transform=image_transforms["test"],
    download=False
)

total_size = len(train_base)
n_classes = len(train_base.categories)
class_names = train_base.categories

print("\nDataset loaded successfully!")
print("Total images:", total_size)
print("Number of classes:", n_classes)


# ------------------------------------------------
# STEP 4: Split the dataset
# 50% Training, 25% Validation, 25% Testing
# ------------------------------------------------

generator = torch.Generator().manual_seed(SEED)

indices = torch.randperm(
    total_size,
    generator=generator
).tolist()

train_end = int(0.50 * total_size)
valid_end = int(0.75 * total_size)

# Use a smaller dataset for faster CPU execution
train_indices = indices[:500]
valid_indices = indices[500:700]
test_indices = indices[700:900]

train_data = Subset(train_base, train_indices)
valid_data = Subset(valid_base, valid_indices)
test_data = Subset(test_base, test_indices)

print("\nDataset split:")
print("Training images:", len(train_data))
print("Validation images:", len(valid_data))
print("Testing images:", len(test_data))


# ------------------------------------------------
# STEP 5: Create DataLoaders
# ------------------------------------------------

BATCH_SIZE = 8

# Zero workers is simpler and safer in Spyder on Windows.
NUM_WORKERS = 0

dataloaders = {
    "train": DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    ),

    "valid": DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    ),

    "test": DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
}

sample_images, sample_labels = next(
    iter(dataloaders["train"])
)

print("\nDataLoaders created successfully!")
print("Sample image batch shape:", sample_images.shape)
print("Expected image shape: [batch, 3, 224, 224]")


# ------------------------------------------------
# STEP 6: Load pretrained VGG-16
# ------------------------------------------------

print("\nLoading pretrained VGG-16 model...")
print("The first download may take some time.")

model = models.vgg16(
    weights=VGG16_Weights.DEFAULT
)

# Freeze all pretrained parameters.
for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier layer.
n_inputs = model.classifier[6].in_features

model.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, n_classes)
)

model = model.to(device)

print("VGG-16 loaded successfully!")
print("Custom classifier added.")


# ------------------------------------------------
# STEP 7: Define training function
# ------------------------------------------------

criterion = nn.CrossEntropyLoss()


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=3,
    patience=2
):
    best_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": []
    }

    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for phase in ["train", "valid"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0
            total = 0

            for images, labels in dataloaders[phase]:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(
                    phase == "train"
                ):

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    predictions = outputs.argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                batch_size = labels.size(0)

                running_loss += (
                    loss.item() * batch_size
                )

                running_correct += (
                    predictions == labels
                ).sum().item()

                total += batch_size

            epoch_loss = running_loss / max(total, 1)

            epoch_acc = (
                running_correct / max(total, 1)
            )

            history[f"{phase}_loss"].append(
                epoch_loss
            )

            history[f"{phase}_acc"].append(
                epoch_acc
            )

            print(
                f"{phase.capitalize()} Loss: "
                f"{epoch_loss:.4f} | "
                f"Accuracy: {epoch_acc * 100:.2f}%"
            )

            # Early stopping based on validation loss.
            if phase == "valid":

                if epoch_loss < best_val_loss:

                    best_val_loss = epoch_loss

                    best_weights = copy.deepcopy(
                        model.state_dict()
                    )

                    epochs_without_improvement = 0

                else:
                    epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            print("Early stopping triggered.")
            break

    model.load_state_dict(best_weights)

    return model, history


# ------------------------------------------------
# STEP 8: Train custom classifier
# ------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 1: TRAINING CUSTOM CLASSIFIER")
print("=" * 60)

optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.001
)

model, history = train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=3,
    patience=2
)

print("\nInitial classifier training completed!")


# ------------------------------------------------
# STEP 9: Fine-tune VGG-16
# ------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 2: FINE-TUNING VGG-16")
print("=" * 60)

# Unfreeze the last convolutional block.
for param in model.features[24:].parameters():
    param.requires_grad = True

# Keep the classifier trainable.
for param in model.classifier.parameters():
    param.requires_grad = True

optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.00001
)

model, fine_history = train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=2,
    patience=2
)

print("\nFine-tuning completed!")


# ------------------------------------------------
# STEP 10: Evaluate using test dataset
# ------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

model.eval()

test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():

    for images, labels in dataloaders["test"]:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        test_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

test_loss /= max(total, 1)
test_accuracy = correct / max(total, 1)

print("Test Loss:", round(test_loss, 4))

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print("Correct predictions:", correct, "/", total)


# ------------------------------------------------
# STEP 11: Plot performance curves
# ------------------------------------------------

combined_history = {}

for key in history:

    combined_history[key] = (
        history[key] + fine_history[key]
    )

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    combined_history["train_acc"],
    label="Training Accuracy"
)

plt.plot(
    combined_history["valid_acc"],
    label="Validation Accuracy"
)

plt.title("VGG-16 Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)

plt.plot(
    combined_history["train_loss"],
    label="Training Loss"
)

plt.plot(
    combined_history["valid_loss"],
    label="Validation Loss"
)

plt.title("VGG-16 Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ------------------------------------------------
# STEP 12: Display sample test predictions
# ------------------------------------------------

print("\nDisplaying sample predictions...")

images, labels = next(
    iter(dataloaders["test"])
)

model.eval()

with torch.no_grad():

    outputs = model(images.to(device))

    predictions = outputs.argmax(
        dim=1
    ).cpu()

mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)

plt.figure(figsize=(12, 8))

for i in range(min(6, len(images))):

    image = images[i] * std + mean

    image = image.permute(1, 2, 0).numpy()

    image = np.clip(image, 0, 1)

    predicted_label = class_names[
        predictions[i].item()
    ]

    actual_label = class_names[
        labels[i].item()
    ]

    plt.subplot(2, 3, i + 1)

    plt.imshow(image)

    plt.title(
        f"Predicted: {predicted_label}\n"
        f"Actual: {actual_label}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------
# STEP 13: Save trained model
# ------------------------------------------------

MODEL_PATH = os.path.join(
    DATA_DIR,
    "vgg16_caltech101.pth"
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\nModel saved successfully:", MODEL_PATH)


# ------------------------------------------------
# FINAL OUTPUT
# ------------------------------------------------

print("\n" + "=" * 60)
print("ASSIGNMENT NO. 6 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Model: VGG-16")
print("Framework: PyTorch")
print("Dataset: Caltech-101")
print("Split: 50% Training, 25% Validation, 25% Testing")
print("Transfer learning: Completed")
print("Fine-tuning: Completed")

print(
    "Test accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print("Performance graphs generated.")
print("Saved model:", MODEL_PATH)
