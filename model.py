"""Enhanced Wide & Deep binary classifier architecture."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureEmbedding(nn.Module):
    """Create one embedding table per categorical feature."""

    def __init__(self, categorical_dims: list[int], embedding_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality in categorical_dims]
        )

    def forward(self, categorical_x: torch.Tensor) -> torch.Tensor:
        embeddings = [
            embedding(categorical_x[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        return torch.cat(embeddings, dim=1)


class FeatureAttention(nn.Module):
    """Self-attention block over the concatenated deep feature vector."""

    def __init__(self, feature_dim: int, num_heads: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequence = features.unsqueeze(1)
        attention_output, _ = self.attention(sequence, sequence, sequence)
        attention_output = self.dropout(attention_output)
        return self.norm(sequence + attention_output).squeeze(1)


class EnhancedWideAndDeep(nn.Module):
    """Wide categorical branch plus deep embedding/MLP branch."""

    def __init__(
        self,
        numeric_dim: int,
        categorical_dims: list[int],
        embedding_dim: int = 8,
        wide_dim: int | None = None,
        hidden_dims: tuple[int, ...] = (64, 32),
        num_attention_heads: int = 2,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        self.wide_dim = wide_dim or len(categorical_dims)
        self.wide = nn.Linear(self.wide_dim, 1)

        self.embedding = FeatureEmbedding(categorical_dims, embedding_dim)
        self.deep_input_dim = numeric_dim + len(categorical_dims) * embedding_dim
        self.feature_attention = FeatureAttention(
            self.deep_input_dim,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
        )

        layers: list[nn.Module] = []
        input_dim = self.deep_input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            input_dim = hidden_dim

        self.deep_mlp = nn.Sequential(*layers)
        self.interaction_layer = nn.Linear(1 + hidden_dims[-1], 16)
        self.interaction_activation = nn.ReLU()
        self.output = nn.Linear(16, 1)

    def forward(self, numeric_x: torch.Tensor, categorical_x: torch.Tensor) -> torch.Tensor:
        wide_output = self.wide(categorical_x.float())

        embedded = self.embedding(categorical_x)
        deep_input = torch.cat([numeric_x, embedded], dim=1)
        deep_attended = self.feature_attention(deep_input)
        deep_output = self.deep_mlp(deep_attended)

        combined = torch.cat([wide_output, deep_output], dim=1)
        interaction = self.interaction_activation(self.interaction_layer(combined))
        output = torch.sigmoid(self.output(interaction))
        return output.squeeze(dim=-1)


def build_model_config(
    numeric_dim: int,
    categorical_dims: list[int],
    embedding_dim: int = 8,
    hidden_dims: tuple[int, ...] = (64, 32),
    num_attention_heads: int = 2,
    dropout_rate: float = 0.3,
) -> dict[str, object]:
    return {
        "numeric_dim": numeric_dim,
        "categorical_dims": categorical_dims,
        "embedding_dim": embedding_dim,
        "hidden_dims": list(hidden_dims),
        "num_attention_heads": num_attention_heads,
        "dropout_rate": dropout_rate,
    }


def create_model(config: dict[str, object]) -> EnhancedWideAndDeep:
    return EnhancedWideAndDeep(
        numeric_dim=int(config["numeric_dim"]),
        categorical_dims=list(config["categorical_dims"]),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dims=tuple(config["hidden_dims"]),
        num_attention_heads=int(config["num_attention_heads"]),
        dropout_rate=float(config["dropout_rate"]),
    )
