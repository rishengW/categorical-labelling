"""Data loading and preprocessing helpers."""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from features import CATEGORICAL_FEATURES, LABEL_COLUMN, NUMERIC_FEATURES


@dataclass
class CustomerPreprocessor:
    """Fit and apply the preprocessing used by the Wide & Deep model.

    Numeric columns are median-imputed and standardized. Categorical columns are
    most-frequent-imputed and encoded as integer ids. Id 0 is reserved for
    unseen categories during inference.
    """

    numeric_features: list[str] = field(default_factory=lambda: list(NUMERIC_FEATURES))
    categorical_features: list[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))
    num_imputer: SimpleImputer = field(default_factory=lambda: SimpleImputer(strategy="median"))
    cat_imputer: SimpleImputer = field(default_factory=lambda: SimpleImputer(strategy="most_frequent"))
    scaler: StandardScaler = field(default_factory=StandardScaler)
    category_maps: dict[str, dict[str, int]] = field(default_factory=dict)

    def fit(self, data: pd.DataFrame) -> "CustomerPreprocessor":
        self._validate_columns(data)

        numeric_values = self.num_imputer.fit_transform(data[self.numeric_features])
        self.scaler.fit(numeric_values)

        categorical_values = self.cat_imputer.fit_transform(data[self.categorical_features])
        self.category_maps = {}
        for index, column in enumerate(self.categorical_features):
            values = pd.Series(categorical_values[:, index]).astype(str)
            categories = sorted(values.dropna().unique().tolist())
            self.category_maps[column] = {
                category: encoded + 1 for encoded, category in enumerate(categories)
            }

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(data)
        transformed = data.copy()

        numeric_values = self.num_imputer.transform(transformed[self.numeric_features])
        scaled_numeric = self.scaler.transform(numeric_values).astype(np.float32)
        for index, column in enumerate(self.numeric_features):
            transformed[column] = scaled_numeric[:, index]

        categorical_values = self.cat_imputer.transform(transformed[self.categorical_features])
        for index, column in enumerate(self.categorical_features):
            values = pd.Series(categorical_values[:, index]).astype(str)
            mapping = self.category_maps[column]
            encoded = values.map(mapping).fillna(0).astype(np.int64)

            unseen_count = int((encoded == 0).sum())
            if unseen_count:
                warnings.warn(
                    f"{unseen_count} unseen value(s) in {column!r} were encoded as 0.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            transformed.loc[:, column] = encoded.to_numpy()

        return transformed

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)

    @property
    def categorical_dims(self) -> list[int]:
        return [len(self.category_maps[column]) + 1 for column in self.categorical_features]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, path: str | Path) -> "CustomerPreprocessor":
        with Path(path).open("rb") as input_file:
            return pickle.load(input_file)

    def _validate_columns(self, data: pd.DataFrame) -> None:
        missing = [
            column
            for column in [*self.numeric_features, *self.categorical_features]
            if column not in data.columns
        ]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def load_customer_csv(path: str | Path, encoding: str = "gbk") -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding)


def split_model_and_holdout(
    data: pd.DataFrame,
    train_size: int = 4800,
    label_column: str = LABEL_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if label_column not in data.columns:
        raise ValueError(f"Expected a {label_column!r} column in training data.")

    model_data = data.iloc[:train_size].copy()
    holdout_data = data.iloc[train_size:].copy()
    return model_data, holdout_data
