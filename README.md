# Categorical Labelling Neural Network

This project trains a PyTorch binary classifier for customer labelling. The
original notebook has been replaced with Python modules so the workflow can be
run from scripts, reused in other code, and versioned without notebook state.

## Files

- `features.py`: shared feature and label column definitions.
- `preprocessing.py`: CSV loading, imputation, scaling, and categorical encoding.
- `dataset.py`: PyTorch `Dataset` for numeric and categorical feature tensors.
- `model.py`: neural network architecture.
- `training.py`: training, evaluation, cross-validation, plotting, and metrics helpers.
- `inference.py`: model loading and prediction helpers.
- `train.py`: command-line training entry point.
- `predict.py`: command-line prediction entry point.

## Network Architecture

The model is `EnhancedWideAndDeep`, a binary Wide & Deep classifier.

Input preprocessing:

- 10 numeric features are median-imputed and standardized.
- 8 categorical features are most-frequent-imputed and integer encoded.
- Categorical id `0` is reserved for unseen inference categories.

Model flow:

1. Wide branch: encoded categorical features are cast to float and passed through
   `Linear(8, 1)`.
2. Embedding branch: each categorical feature has its own embedding table with
   embedding dimension `8`.
3. Deep input: the 10 standardized numeric features are concatenated with the 8
   categorical embeddings, giving `10 + 8 * 8 = 74` deep features.
4. Attention block: `MultiheadAttention(embed_dim=74, num_heads=2)` with dropout,
   residual connection, and layer normalization.
5. MLP: `Linear(74, 64) -> BatchNorm -> ReLU -> Dropout`, then
   `Linear(64, 32) -> BatchNorm -> ReLU -> Dropout`.
6. Interaction layer: the 1-dimensional wide output is concatenated with the
   32-dimensional deep output and passed through `Linear(33, 16) -> ReLU`.
7. Output layer: `Linear(16, 1) -> Sigmoid` returns the probability for label `1`.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --train-csv /kaggle/input/traintestdataset/train.csv
```

Useful options:

- `--output-dir artifacts`: where model files, plots, and metrics are saved.
- `--train-size 4800`: number of initial rows used for modelling, matching the
  original notebook.
- `--epochs 50`: maximum training epochs.
- `--folds 5`: number of stratified cross-validation folds.
- `--skip-cv`: train only the final model.

Training writes:

- `artifacts/best_model.pth`
- `artifacts/model_config.json`
- `artifacts/preprocessor.pkl`
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`
- `artifacts/training_curve.png`
- `artifacts/probability_distribution.png`

## Predict

```bash
python predict.py \
  --input-csv /kaggle/input/traintestdataset/test.csv \
  --artifacts-dir artifacts \
  --output-csv test_with_pred.csv
```

The prediction script loads the saved model configuration, weights, and
preprocessor, then writes a UTF-8 CSV with the `label` column populated.
