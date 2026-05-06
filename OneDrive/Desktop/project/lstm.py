from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import TideThresholds, build_feature_dataset, get_numeric_columns


ARTIFACTS_DIR = Path("artifacts_lstm")
PLOTS_DIR = ARTIFACTS_DIR / "plots"
THRESHOLDS = TideThresholds(safe_threshold=0.5, severe_threshold=1.0)
SEQUENCE_LENGTH = 24
RANDOM_STATE = 42
BATCH_SIZE = 128
EPOCHS = 5
PATIENCE = 3
HIDDEN_SIZE = 32
NUM_LAYERS = 1
LSTM_DROPOUT = 0.0


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def severity_from_water_level(values: np.ndarray, thresholds: TideThresholds = THRESHOLDS) -> np.ndarray:
    return np.where(
        values < thresholds.safe_threshold,
        0,
        np.where(values < thresholds.severe_threshold, 1, 2),
    )


def create_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    seq_df = df.copy()
    seq_df["hour"] = seq_df.index.hour
    seq_df["day_of_week"] = seq_df.index.dayofweek
    seq_df["month"] = seq_df.index.month
    seq_df["day_of_year"] = seq_df.index.dayofyear
    seq_df["hour_sin"] = np.sin(2 * np.pi * seq_df["hour"] / 24)
    seq_df["hour_cos"] = np.cos(2 * np.pi * seq_df["hour"] / 24)
    seq_df["dayofyear_sin"] = np.sin(2 * np.pi * seq_df["day_of_year"] / 365.25)
    seq_df["dayofyear_cos"] = np.cos(2 * np.pi * seq_df["day_of_year"] / 365.25)
    numeric_columns = get_numeric_columns(seq_df)
    time_columns = ["hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos", "day_of_week", "month"]
    return seq_df[numeric_columns + time_columns]


def split_timewise(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def fit_standardization(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    mean = train_df.mean()
    std = train_df.std(ddof=0).replace(0, 1.0)
    return {"mean": mean.to_dict(), "std": std.to_dict()}


def transform_standardization(df: pd.DataFrame, scaler_state: dict[str, dict[str, float]]) -> pd.DataFrame:
    mean = pd.Series(scaler_state["mean"])
    std = pd.Series(scaler_state["std"])
    return (df - mean) / std


def make_sequences(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    X_sequences = []
    y_values = []
    timestamps = []

    for end_idx in range(sequence_length, len(feature_df)):
        start_idx = end_idx - sequence_length
        X_sequences.append(feature_df.iloc[start_idx:end_idx].values)
        y_values.append(target_series.iloc[end_idx])
        timestamps.append(feature_df.index[end_idx])

    return np.array(X_sequences, dtype=np.float32), np.array(y_values), pd.Index(timestamps)


def create_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    task: str,
) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    if task == "regression":
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    else:
        y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    task: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    losses = []
    preds = []
    targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if is_training:
            optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        if task == "regression":
            preds.append(outputs.detach().cpu().numpy().flatten())
            targets.append(y_batch.detach().cpu().numpy().flatten())
        else:
            preds.append(torch.argmax(outputs, dim=1).detach().cpu().numpy())
            targets.append(y_batch.detach().cpu().numpy())

    return float(np.mean(losses)), np.concatenate(preds), np.concatenate(targets)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    save_path: Path,
) -> dict[str, list[float]]:
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(EPOCHS):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, task)
        val_loss, _, _ = run_epoch(model, val_loader, criterion, None, device, task)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    return history


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    accuracy = float(np.mean(y_true == y_pred))
    f1_scores = []
    weighted_parts = []
    for cls in [0, 1, 2]:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int(np.sum(y_true == cls))
        f1_scores.append(f1)
        weighted_parts.append(f1 * support)
    macro_f1 = float(np.mean(f1_scores))
    weighted_f1 = float(sum(weighted_parts) / len(y_true))
    return {"accuracy": accuracy, "f1_macro": macro_f1, "f1_weighted": weighted_f1}


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def plot_regression_results(index: pd.Index, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    sample_size = min(300, len(y_true))
    axes[0].plot(index[-sample_size:], y_true[-sample_size:], label="actual")
    axes[0].plot(index[-sample_size:], y_pred[-sample_size:], label="predicted")
    axes[0].set_title("PyTorch LSTM regression: actual vs predicted")
    axes[0].legend()

    axes[1].scatter(y_true, y_pred, alpha=0.35)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    axes[1].plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    axes[1].set_xlabel("Actual water level")
    axes[1].set_ylabel("Predicted water level")
    axes[1].set_title("PyTorch LSTM regression scatter")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lstm_regression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Oranges")
    labels = ["safe", "medium", "severe"]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(name)
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{name.replace(' ', '_').lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: dict[str, list[float]], name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="train_loss")
    ax.plot(history["val_loss"], label="val_loss")
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{name.replace(' ', '_').lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    plt.style.use("ggplot")
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    device = get_device()
    print(f"Using device: {device}")

    feature_df, _ = build_feature_dataset(thresholds=THRESHOLDS)
    seq_source = create_sequence_features(feature_df).dropna().copy()

    train_df, val_df, test_df = split_timewise(seq_source)
    scaler_state = fit_standardization(train_df)
    joblib.dump(scaler_state, ARTIFACTS_DIR / "lstm_sequence_scaler.joblib")

    train_scaled = transform_standardization(train_df, scaler_state)
    val_scaled = transform_standardization(val_df, scaler_state)
    test_scaled = transform_standardization(test_df, scaler_state)

    train_y_reg = train_df["Water_Level"]
    val_y_reg = val_df["Water_Level"]
    test_y_reg = test_df["Water_Level"]

    train_y_cls = pd.Series(severity_from_water_level(train_y_reg.values), index=train_df.index)
    val_y_cls = pd.Series(severity_from_water_level(val_y_reg.values), index=val_df.index)
    test_y_cls = pd.Series(severity_from_water_level(test_y_reg.values), index=test_df.index)

    X_train_reg, y_train_reg, _ = make_sequences(train_scaled, train_y_reg, SEQUENCE_LENGTH)
    X_val_reg, y_val_reg, _ = make_sequences(val_scaled, val_y_reg, SEQUENCE_LENGTH)
    X_test_reg, y_test_reg, reg_test_index = make_sequences(test_scaled, test_y_reg, SEQUENCE_LENGTH)

    X_train_cls, y_train_cls, _ = make_sequences(train_scaled, train_y_cls, SEQUENCE_LENGTH)
    X_val_cls, y_val_cls, _ = make_sequences(val_scaled, val_y_cls, SEQUENCE_LENGTH)
    X_test_cls, y_test_cls, _ = make_sequences(test_scaled, test_y_cls, SEQUENCE_LENGTH)

    train_reg_loader = create_loader(X_train_reg, y_train_reg, BATCH_SIZE, True, "regression")
    val_reg_loader = create_loader(X_val_reg, y_val_reg, BATCH_SIZE, False, "regression")
    test_reg_loader = create_loader(X_test_reg, y_test_reg, BATCH_SIZE, False, "regression")

    train_cls_loader = create_loader(X_train_cls, y_train_cls, BATCH_SIZE, True, "classification")
    val_cls_loader = create_loader(X_val_cls, y_val_cls, BATCH_SIZE, False, "classification")
    test_cls_loader = create_loader(X_test_cls, y_test_cls, BATCH_SIZE, False, "classification")

    input_size = X_train_reg.shape[2]

    reg_model = LSTMRegressor(input_size=input_size).to(device)
    reg_criterion = nn.MSELoss()
    reg_optimizer = torch.optim.Adam(reg_model.parameters(), lr=1e-3)
    reg_model_path = ARTIFACTS_DIR / "best_lstm_regression.pt"
    reg_history = train_model(
        reg_model,
        train_reg_loader,
        val_reg_loader,
        reg_criterion,
        reg_optimizer,
        device,
        "regression",
        reg_model_path,
    )
    reg_model.load_state_dict(torch.load(reg_model_path, map_location=device))
    _, reg_pred, reg_true = run_epoch(reg_model, test_reg_loader, reg_criterion, None, device, "regression")
    reg_metrics = regression_metrics(reg_true, reg_pred)
    plot_regression_results(reg_test_index, reg_true, reg_pred)
    plot_history(reg_history, "LSTM Regression Loss")

    pred_severity_from_reg = severity_from_water_level(reg_pred)
    severity_from_reg_metrics = classification_metrics(y_test_cls, pred_severity_from_reg)
    plot_confusion_matrix(confusion(y_test_cls, pred_severity_from_reg), "LSTM Severity From Regression")

    cls_model = LSTMClassifier(input_size=input_size).to(device)
    cls_criterion = nn.CrossEntropyLoss()
    cls_optimizer = torch.optim.Adam(cls_model.parameters(), lr=1e-3)
    cls_model_path = ARTIFACTS_DIR / "best_lstm_classifier.pt"
    cls_history = train_model(
        cls_model,
        train_cls_loader,
        val_cls_loader,
        cls_criterion,
        cls_optimizer,
        device,
        "classification",
        cls_model_path,
    )
    cls_model.load_state_dict(torch.load(cls_model_path, map_location=device))
    _, cls_pred, cls_true = run_epoch(cls_model, test_cls_loader, cls_criterion, None, device, "classification")
    cls_metrics = classification_metrics(cls_true, cls_pred)
    plot_confusion_matrix(confusion(cls_true, cls_pred), "LSTM Classification")
    plot_history(cls_history, "LSTM Classification Loss")

    summary = {
        "framework": "pytorch",
        "device": str(device),
        "sequence_length": SEQUENCE_LENGTH,
        "regression_metrics": reg_metrics,
        "severity_metrics_from_regression": severity_from_reg_metrics,
        "classification_metrics": cls_metrics,
        "saved_models": {
            "lstm_regression": str(reg_model_path.resolve()),
            "lstm_classifier": str(cls_model_path.resolve()),
            "sequence_scaler": str((ARTIFACTS_DIR / "lstm_sequence_scaler.joblib").resolve()),
        },
        "input_features_for_sequence_model": list(train_scaled.columns),
    }
    with open(ARTIFACTS_DIR / "lstm_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with open(ARTIFACTS_DIR / "lstm_history.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "regression_history": reg_history,
                "classification_history": cls_history,
            },
            file,
            indent=2,
        )

    print("PyTorch LSTM regression metrics")
    print(reg_metrics)
    print()
    print("Severity metrics from thresholded regression output")
    print(severity_from_reg_metrics)
    print()
    print("PyTorch LSTM classification metrics")
    print(cls_metrics)
    print()
    print(f"LSTM artifacts saved in: {ARTIFACTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
