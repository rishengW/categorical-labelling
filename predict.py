"""Command-line entry point for generating labels with a trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

from features import LABEL_COLUMN
from inference import load_trained_components, predict_dataframe
from preprocessing import load_customer_csv
from training import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict customer labels.")
    parser.add_argument(
        "--input-csv",
        default="/kaggle/input/traintestdataset/test.csv",
        help="Path to the unlabelled input CSV.",
    )
    parser.add_argument("--output-csv", default="test_with_pred.csv", help="Prediction output CSV.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory with trained artifacts.")
    parser.add_argument("--encoding", default="gbk", help="CSV input encoding.")
    parser.add_argument("--batch-size", type=int, default=64, help="Prediction batch size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model, preprocessor = load_trained_components(args.artifacts_dir, device=device)

    data = load_customer_csv(args.input_csv, encoding=args.encoding)
    predictions, _ = predict_dataframe(
        data,
        model,
        preprocessor,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=device,
    )

    output_data = data.copy()
    output_data[LABEL_COLUMN] = predictions.astype(int)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved predictions to {output_path.resolve()}")


if __name__ == "__main__":
    main()
