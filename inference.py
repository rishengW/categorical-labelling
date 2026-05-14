"""Prediction helpers for trained customer labelling models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features import LABEL_COLUMN
from model import create_model
from preprocessing import CustomerPreprocessor
from training import get_device, make_data_loader


def predict_model(
    model: torch.nn.Module,
    data_loader,
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    device = device or get_device()
    model.to(device)
    model.eval()

    probabilities: list[float] = []
    predictions: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            numeric_x = batch["numeric"].to(device)
            categorical_x = batch["categorical"].to(device)
            outputs = model(numeric_x, categorical_x)
            probs = outputs.detach().cpu().numpy()
            preds = (outputs > threshold).int().detach().cpu().numpy()

            probabilities.extend(probs.tolist())
            predictions.extend(preds.tolist())

    return np.asarray(predictions, dtype=np.int64), np.asarray(probabilities, dtype=np.float32)


def load_trained_components(
    artifacts_dir: str | Path,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, CustomerPreprocessor]:
    artifacts_dir = Path(artifacts_dir)
    device = device or get_device()

    with (artifacts_dir / "model_config.json").open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    model = create_model(config)
    state_dict = torch.load(artifacts_dir / "best_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    preprocessor = CustomerPreprocessor.load(artifacts_dir / "preprocessor.pkl")
    return model, preprocessor


def predict_dataframe(
    data: pd.DataFrame,
    model: torch.nn.Module,
    preprocessor: CustomerPreprocessor,
    batch_size: int = 64,
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    working_data = data.drop(columns=[LABEL_COLUMN], errors="ignore")
    processed = preprocessor.transform(working_data)
    loader = make_data_loader(processed, batch_size=batch_size, shuffle=False)
    return predict_model(model, loader, threshold=threshold, device=device)
