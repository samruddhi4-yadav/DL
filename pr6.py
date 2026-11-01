# ============================================================
# PRACTICAL 6: TRANSFER LEARNING USING PRETRAINED VGG-16
# Dataset: Caltech-101
# Framework: PyTorch
# ============================================================

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


# ============================================================
# STEP 1: INITIALIZE SETTINGS
# ============================================================

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
    print("GPU not available. Training will use CPU.")


# ============================================================
# STEP 2: DEFINE IMAGE TRANSFORMATIONS
# ============================================================

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


# ============================================================
# STEP 3: LOAD CALTECH-101 DATASET
# ============================================================

# Dataset will be downloaded into this folder.
# This relative path works in Windows, Spyder, and VS Code.

DATA_DIR = "./caltech101_data"

os.makedirs(DATA_DIR, exist_ok=True)

print("\nLoading Caltech-101 dataset...")
print("The first download may take several minutes.")

train_base = datasets.Caltech101(
    root=DATA_DIR,
    target_type="category",
    transform=image_transforms["train"],
    download=True
)

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


# ============================================================
# STEP 4: SPLIT DATASET
# 50% TRAINING, 25% VALIDATION, 25% TESTING
# ============================================================

generator = torch.Generator().manual_seed(SEED)

indices = torch.randperm(
    total_size,
    generator=generator
).tolist()

train_end = int(0.50 * total_size)
valid_end = int(0.75 * total_size)

train_indices = indices[:train_end]
valid_indices = indices[train_end:valid_end]
test_indices = indices[valid_end:]

train_data = Subset(
    train_base,
    train_indices
)

valid_data = Subset(
    valid_base,
    valid_indices
)

test_data = Subset(
    test_base,
    test_indices
)

print("\nDataset split completed!")

print("Training images:", len(train_data))
print("Validation images:", len(valid_data))
print("Testing images:", len(test_data))


# ============================================================
# STEP 5: CREATE DATALOADERS
# ============================================================

BATCH_SIZE = 16

# Using 0 workers improves compatibility with Windows and Spyder.
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
print("Sample batch shape:", sample_images.shape)
print("Sample labels shape:", sample_labels.shape)


# ============================================================
# STEP 6: LOAD PRETRAINED VGG-16
# ============================================================

print("\nLoading pretrained VGG-16 model...")

model = models.vgg16(
    weights=VGG16_Weights.DEFAULT
)

# Freeze all pretrained model parameters.

for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier with a custom classifier.

n_inputs = model.classifier[6].in_features

model.classifier[6] = nn.Sequential(

    nn.Linear(n_inputs, 256),

    nn.ReLU(),

    nn.Dropout(0.4),

    nn.Linear(256, n_classes),

    nn.LogSoftmax(dim=1)
)

model = model.to(device)

print("VGG-16 loaded successfully!")
print("Custom classifier added!")

print("\nCustom classifier:")
print(model.classifier[6])


# ============================================================
# STEP 7: DEFINE LOSS FUNCTION
# ============================================================

criterion = nn.NLLLoss()


# ============================================================
# STEP 8: DEFINE TRAINING FUNCTION
# WITH EARLY STOPPING
# ============================================================

def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=5,
    patience=2
):

    best_weights = copy.deepcopy(
        model.state_dict()
    )

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

        # Train and validation phases.

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

                    loss = criterion(
                        outputs,
                        labels
                    )

                    predictions = outputs.argmax(
                        dim=1
                    )

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

            epoch_loss = running_loss / total

            epoch_acc = running_correct / total

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

            # Check validation loss for early stopping.

            if phase == "valid":

                if epoch_loss < best_val_loss:

                    best_val_loss = epoch_loss

                    best_weights = copy.deepcopy(
                        model.state_dict()
                    )

                    epochs_without_improvement = 0

                else:

                    epochs_without_improvement += 1

        # Check early stopping after both phases.

        if epochs_without_improvement >= patience:

            print("Early stopping triggered.")
            break

    # Restore the best model weights.

    model.load_state_dict(best_weights)

    return model, history


# ============================================================
# STEP 9: TRAIN CUSTOM CLASSIFIER
# ============================================================

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
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=5,
    patience=2
)

print("\nCustom classifier training completed!")


# ============================================================
# STEP 10: FINE-TUNE LAST CONVOLUTIONAL BLOCK
# ============================================================

print("\n" + "=" * 60)
print("STAGE 2: FINE-TUNING VGG-16")
print("=" * 60)

# Unfreeze the final convolutional block.

for param in model.features[24:].parameters():
    param.requires_grad = True

# Ensure all classifier parameters are trainable.

for param in model.classifier.parameters():
    param.requires_grad = True

# Use a smaller learning rate for fine-tuning.

optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.00001
)

model, fine_history = train_model(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=3,
    patience=2
)

print("\nFine-tuning completed!")


# ============================================================
# STEP 11: EVALUATE ON TEST DATA
# ============================================================

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

        loss = criterion(
            outputs,
            labels
        )

        batch_size = labels.size(0)

        test_loss += (
            loss.item() * batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

test_loss = test_loss / total

test_accuracy = correct / total

print("Test Loss:", round(test_loss, 4))

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print(
    "Correct predictions:",
    correct,
    "/",
    total
)


# ============================================================
# STEP 12: PLOT ACCURACY AND LOSS
# ============================================================

combined_history = {}

for key in history:

    combined_history[key] = (
        history[key] + fine_history[key]
    )

epochs = range(
    1,
    len(combined_history["train_acc"]) + 1
)

plt.figure(figsize=(12, 5))

# Accuracy graph.

plt.subplot(1, 2, 1)

plt.plot(
    epochs,
    combined_history["train_acc"],
    label="Training Accuracy"
)

plt.plot(
    epochs,
    combined_history["valid_acc"],
    label="Validation Accuracy"
)

plt.title("VGG-16 Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss graph.

plt.subplot(1, 2, 2)

plt.plot(
    epochs,
    combined_history["train_loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
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


# ============================================================
# STEP 13: DISPLAY SAMPLE TEST PREDICTIONS
# ============================================================

print("\nDisplaying sample test predictions...")

images, labels = next(
    iter(dataloaders["test"])
)

model.eval()

with torch.no_grad():

    outputs = model(
        images.to(device)
    )

    predictions = outputs.argmax(
        dim=1
    ).cpu()

# Values used to reverse image normalization.

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


# ============================================================
# STEP 14: SAVE TRAINED MODEL
# ============================================================

MODEL_PATH = "./vgg16_caltech101.pth"

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\nModel saved successfully:", MODEL_PATH)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("PRACTICAL 6 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Model: VGG-16")
print("Framework: PyTorch")
print("Dataset: Caltech-101")

print("Training split: 50%")
print("Validation split: 25%")
print("Testing split: 25%")

print("Transfer learning: Completed")
print("Fine-tuning: Completed")

print(
    "Test accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print("Performance graphs generated.")
print("Saved model:", MODEL_PATH)