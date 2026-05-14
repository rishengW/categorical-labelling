"""Training, evaluation, and plotting helpers."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from dataset import CustomerDataset
from features import CATEGORICAL_FEATURES, LABEL_COLUMN, NUMERIC_FEATURES
from model import EnhancedWideAndDeep


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau | None = None,
    num_epochs: int = 50,
    patience: int = 10,
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
) -> tuple[list[float], list[float]]:
    device = device or get_device()
    model.to(device)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            numeric_x = batch["numeric"].to(device)
            categorical_x = batch["categorical"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(numeric_x, categorical_x)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * numeric_x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        val_epoch_loss, val_accuracy = _validation_step(model, val_loader, criterion, device)
        val_losses.append(val_epoch_loss)

        if scheduler:
            scheduler.step(val_epoch_loss)

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path:
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {epoch_loss:.4f}, "
                f"Val Loss: {val_epoch_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}"
            )

    model.load_state_dict(best_state)
    return train_losses, val_losses


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | None = None,
    confusion_matrix_path: str | Path | None = None,
) -> dict[str, object]:
    device = device or get_device()
    model.to(device)
    model.eval()

    all_probs: list[float] = []
    all_preds: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for batch in data_loader:
            numeric_x = batch["numeric"].to(device)
            categorical_x = batch["categorical"].to(device)
            labels = batch["label"].to(device)

            outputs = model(numeric_x, categorical_x)
            probs = outputs.detach().cpu().numpy()
            preds = (outputs > 0.5).float().detach().cpu().numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    report = classification_report(all_labels, all_preds)
    matrix = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Classification report:")
    print(report)

    if confusion_matrix_path:
        plot_confusion_matrix(matrix, confusion_matrix_path)

    return {
        "accuracy": accuracy,
        "auc": auc,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "probs": all_probs,
        "preds": all_preds,
        "labels": all_labels,
    }


def cross_validate(
    model_class: type[EnhancedWideAndDeep],
    data,
    labels,
    numeric_cols: list[str],
    categorical_cols: list[str],
    categorical_dims: list[int],
    n_splits: int = 5,
    batch_size: int = 32,
    num_epochs: int = 50,
    device: torch.device | None = None,
) -> list[dict[str, object]]:
    device = device or get_device()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results: list[dict[str, object]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels), start=1):
        print(f"\nTraining fold {fold}/{n_splits}...")

        x_train, x_val = data.iloc[train_idx], data.iloc[val_idx]
        y_train, y_val = labels.iloc[train_idx], labels.iloc[val_idx]

        train_dataset = CustomerDataset(x_train, y_train, numeric_cols, categorical_cols)
        val_dataset = CustomerDataset(x_val, y_val, numeric_cols, categorical_cols)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = model_class(
            numeric_dim=len(numeric_cols),
            categorical_dims=categorical_dims,
            embedding_dim=8,
            hidden_dims=(64, 32),
            num_attention_heads=2,
            dropout_rate=0.3,
        )
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=5,
            factor=0.5,
        )

        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs=num_epochs,
            device=device,
        )
        metrics = evaluate_model(model, val_loader, device=device)
        fold_results.append(
            {
                "accuracy": metrics["accuracy"],
                "auc": metrics["auc"],
                "train_losses": train_losses,
                "val_losses": val_losses,
            }
        )

    return fold_results


def make_data_loader(
    data,
    labels=None,
    batch_size: int = 32,
    shuffle: bool = False,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> DataLoader:
    dataset = CustomerDataset(
        data,
        labels,
        numeric_cols or list(NUMERIC_FEATURES),
        categorical_cols or list(CATEGORICAL_FEATURES),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_validation_split(data, test_size: float = 0.2):
    features = data.drop(LABEL_COLUMN, axis=1)
    labels = data[LABEL_COLUMN]
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )


def plot_training_curve(
    train_losses: list[float],
    val_losses: list[float],
    output_path: str | Path,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def plot_probability_distribution(
    probs: list[float],
    labels: list[float],
    output_path: str | Path,
) -> None:
    negative = [prob for prob, label in zip(probs, labels) if label == 0]
    positive = [prob for prob, label in zip(probs, labels) if label == 1]

    plt.figure(figsize=(10, 6))
    plt.hist(negative, alpha=0.5, label="Negative", bins=20)
    plt.hist(positive, alpha=0.5, label="Positive", bins=20)
    plt.title("Prediction Probability Distribution")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(matrix: np.ndarray, output_path: str | Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(output_path)
    plt.close()


def save_json(data: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(data, output, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _validation_step(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    val_loss = 0.0
    all_preds: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for batch in val_loader:
            numeric_x = batch["numeric"].to(device)
            categorical_x = batch["categorical"].to(device)
            labels = batch["label"].to(device)

            outputs = model(numeric_x, categorical_x)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * numeric_x.size(0)

            preds = (outputs > 0.5).float()
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    val_epoch_loss = val_loss / len(val_loader.dataset)
    return val_epoch_loss, accuracy_score(all_labels, all_preds)
