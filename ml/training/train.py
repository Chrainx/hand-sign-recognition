import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.training.dataset_digits import DigitsCSVDataset, DigitsDatasetConfig
from ml.training.model_mlp import DigitMLP
from ml.training.split import stratified_split


# ==========================
# CONFIG
# ==========================
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 20
MODEL_DIR = "models"
MODEL_NAME = "digit_mlp.pth"


# ==========================
# TRAIN FUNCTION
# ==========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ==========================
# VALIDATION FUNCTION
# ==========================
def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total

    return val_loss, val_acc


# ==========================
# MAIN TRAINING ENTRYPOINT
# ==========================
def main():
    print("Starting training pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------
    # Load Dataset
    # ----------------------
    dataset_config = DigitsDatasetConfig(
        csv_path="data/raw/digits_dataset.csv"
    )

    dataset = DigitsCSVDataset(dataset_config)

    train_dataset, val_dataset = stratified_split(dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------
    # Initialize Model
    # ----------------------
    input_size = dataset.num_features
    model = DigitMLP(input_dim=input_size, num_classes=10)
    model.to(device)

    # ----------------------
    # Loss & Optimizer
    # ----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ----------------------
    # Training Loop
    # ----------------------
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # ----------------------
    # Save Model
    # ----------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()