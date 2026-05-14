"""PyTorch dataset for customer features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class CustomerDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        labels: pd.Series | np.ndarray | None = None,
        numeric_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
    ) -> None:
        self.numeric_cols = numeric_cols or list(NUMERIC_FEATURES)
        self.categorical_cols = categorical_cols or list(CATEGORICAL_FEATURES)
        self.numeric_data = data[self.numeric_cols].to_numpy(dtype=np.float32)
        self.categorical_data = data[self.categorical_cols].to_numpy(dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.numeric_data)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.float32]:
        item: dict[str, np.ndarray | np.float32] = {
            "numeric": self.numeric_data[index],
            "categorical": self.categorical_data[index],
        }
        if self.labels is not None:
            item["label"] = self.labels[index]
        return item
