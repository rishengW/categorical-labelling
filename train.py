"""Command-line entry point for training the customer labelling model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from features import CATEGORICAL_FEATURES, LABEL_COLUMN, NUMERIC_FEATURES
from model import EnhancedWideAndDeep, build_model_config, create_model
from preprocessing import CustomerPreprocessor, load_customer_csv, split_model_and_holdout
from training import (
    cross_validate,
    evaluate_model,
    get_device,
    make_data_loader,
    plot_probability_distribution,
    plot_training_curve,
    save_json,
    set_seed,
    train_model,
    train_validation_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the customer labelling model.")
    parser.add_argument(
        "--train-csv",
        default="/kaggle/input/traintestdataset/train.csv",
        help="Path to the labelled training CSV.",
    )
    parser.add_argument("--encoding", default="gbk", help="CSV input encoding.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for trained artifacts.")
    parser.add_argument("--train-size", type=int, default=4800, help="Rows used for modelling.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    train_df = load_customer_csv(args.train_csv, encoding=args.encoding)
    model_df, holdout_df = split_model_and_holdout(train_df, train_size=args.train_size)

    preprocessor = CustomerPreprocessor()
    processed_model_df = preprocessor.fit_transform(model_df)
    categorical_dims = preprocessor.categorical_dims

    features = processed_model_df.drop(LABEL_COLUMN, axis=1)
    labels = processed_model_df[LABEL_COLUMN]

    cv_results = []
    if not args.skip_cv and args.folds > 1:
        print("Starting cross-validation...")
        cv_results = cross_validate(
            EnhancedWideAndDeep,
            features,
            labels,
            NUMERIC_FEATURES,
            CATEGORICAL_FEATURES,
            categorical_dims,
            n_splits=args.folds,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            device=device,
        )
        avg_accuracy = float(np.mean([result["accuracy"] for result in cv_results]))
        avg_auc = float(np.nanmean([result["auc"] for result in cv_results]))
        print(f"Average CV accuracy: {avg_accuracy:.4f}")
        print(f"Average CV AUC: {avg_auc:.4f}")

    x_train, x_val, y_train, y_val = train_validation_split(processed_model_df)
    train_loader = make_data_loader(x_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_loader = make_data_loader(x_val, y_val, batch_size=args.batch_size, shuffle=False)

    model_config = build_model_config(
        numeric_dim=len(NUMERIC_FEATURES),
        categorical_dims=categorical_dims,
        embedding_dim=8,
        hidden_dims=(64, 32),
        num_attention_heads=2,
        dropout_rate=0.3,
    )
    model = create_model(model_config)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5,
    )

    print("Training final model...")
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=args.epochs,
        checkpoint_path=output_dir / "best_model.pth",
        device=device,
    )

    state_dict = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    metrics = evaluate_model(
        model,
        val_loader,
        device=device,
        confusion_matrix_path=output_dir / "confusion_matrix.png",
    )
    plot_training_curve(train_losses, val_losses, output_dir / "training_curve.png")
    plot_probability_distribution(
        metrics["probs"],
        metrics["labels"],
        output_dir / "probability_distribution.png",
    )

    holdout_metrics = None
    if not holdout_df.empty and LABEL_COLUMN in holdout_df.columns:
        processed_holdout = preprocessor.transform(holdout_df)
        holdout_x = processed_holdout.drop(LABEL_COLUMN, axis=1)
        holdout_y = processed_holdout[LABEL_COLUMN]
        holdout_loader = make_data_loader(
            holdout_x,
            holdout_y,
            batch_size=args.batch_size,
            shuffle=False,
        )
        print("Evaluating holdout rows...")
        holdout_metrics = evaluate_model(model, holdout_loader, device=device)

    preprocessor.save(output_dir / "preprocessor.pkl")
    save_json(model_config, output_dir / "model_config.json")
    save_json(
        {
            "validation_accuracy": metrics["accuracy"],
            "validation_auc": metrics["auc"],
            "cross_validation": cv_results,
            "holdout_accuracy": None if holdout_metrics is None else holdout_metrics["accuracy"],
            "holdout_auc": None if holdout_metrics is None else holdout_metrics["auc"],
        },
        output_dir / "metrics.json",
    )
    print(f"Saved model artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
