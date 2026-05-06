from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score

from preprocessing import TideThresholds, build_feature_dataset, prepare_model_data, scale_splits


ARTIFACTS_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACTS_DIR / "plots"
THRESHOLDS = TideThresholds(safe_threshold=0.5, severe_threshold=1.0)


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


def get_regression_models() -> dict[str, object]:
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }


def get_classification_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting_classifier": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            random_state=42,
        ),
    }


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def severity_from_water_level(
    values: np.ndarray | pd.Series,
    thresholds: TideThresholds = THRESHOLDS,
) -> np.ndarray:
    values = np.asarray(values)
    return np.where(
        values < thresholds.safe_threshold,
        0,
        np.where(values < thresholds.severe_threshold, 1, 2),
    )


def plot_regression_predictions(
    index: pd.Index,
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    sample_size = min(300, len(y_true))
    axes[0].plot(index[-sample_size:], y_true.iloc[-sample_size:], label="actual", linewidth=1.4)
    axes[0].plot(index[-sample_size:], y_pred[-sample_size:], label="predicted", linewidth=1.2)
    axes[0].set_title(f"{model_name} regression: actual vs predicted")
    axes[0].legend()

    axes[1].scatter(y_true, y_pred, alpha=0.35)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    axes[1].plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    axes[1].set_xlabel("Actual water level")
    axes[1].set_ylabel("Predicted water level")
    axes[1].set_title(f"{model_name} regression scatter")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model_name}_regression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: np.ndarray, model_name: str, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} confusion matrix")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model_name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_regression_models(model_data: dict[str, object]) -> tuple[pd.DataFrame, str, dict[str, object]]:
    regression_models = get_regression_models()
    results: list[dict[str, float | str]] = []
    trained_models: dict[str, object] = {}

    X_train = model_data["X_train"]
    X_val = model_data["X_val"]
    X_test = model_data["X_test"]
    y_train = model_data["y_train_reg"]
    y_val = model_data["y_val_reg"]
    y_test = model_data["y_test_reg"]

    X_train_scaled = model_data["X_train_scaled"]
    X_val_scaled = model_data["X_val_scaled"]
    X_test_scaled = model_data["X_test_scaled"]

    scaled_model_names = {"linear_regression", "ridge"}

    for name, model in regression_models.items():
        if name in scaled_model_names:
            fit_X_train, fit_X_val, fit_X_test = X_train_scaled, X_val_scaled, X_test_scaled
        else:
            fit_X_train, fit_X_val, fit_X_test = X_train, X_val, X_test

        model.fit(fit_X_train, y_train)
        val_pred = model.predict(fit_X_val)
        test_pred = model.predict(fit_X_test)

        val_metrics = evaluate_regression(y_val, val_pred)
        test_metrics = evaluate_regression(y_test, test_pred)

        results.append(
            {
                "model": name,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values(["val_rmse", "val_mae"]).reset_index(drop=True)
    best_model_name = str(results_df.iloc[0]["model"])
    best_model = trained_models[best_model_name]

    best_input = X_test_scaled if best_model_name in scaled_model_names else X_test
    best_pred = best_model.predict(best_input)
    plot_regression_predictions(X_test.index, y_test, best_pred, best_model_name)

    joblib.dump(best_model, ARTIFACTS_DIR / "best_regression_model.joblib")
    np.save(ARTIFACTS_DIR / "best_regression_test_predictions.npy", best_pred)

    return results_df, best_model_name, trained_models


def train_classification_models(model_data: dict[str, object]) -> tuple[pd.DataFrame, str, dict[str, object]]:
    classification_models = get_classification_models()
    results: list[dict[str, float | str]] = []
    trained_models: dict[str, object] = {}

    X_train = model_data["X_train"]
    X_val = model_data["X_val"]
    X_test = model_data["X_test"]
    y_train = model_data["y_train_cls"]
    y_val = model_data["y_val_cls"]
    y_test = model_data["y_test_cls"]

    X_train_scaled = model_data["X_train_scaled"]
    X_val_scaled = model_data["X_val_scaled"]
    X_test_scaled = model_data["X_test_scaled"]

    scaled_model_names = {"logistic_regression"}

    for name, model in classification_models.items():
        if name in scaled_model_names:
            fit_X_train, fit_X_val, fit_X_test = X_train_scaled, X_val_scaled, X_test_scaled
        else:
            fit_X_train, fit_X_val, fit_X_test = X_train, X_val, X_test

        model.fit(fit_X_train, y_train)
        val_pred = model.predict(fit_X_val)
        test_pred = model.predict(fit_X_test)

        val_metrics = evaluate_classification(y_val, val_pred)
        test_metrics = evaluate_classification(y_test, test_pred)

        results.append(
            {
                "model": name,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "val_f1_macro": val_metrics["f1_macro"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_weighted": test_metrics["f1_weighted"],
                "test_f1_macro": test_metrics["f1_macro"],
            }
        )
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values(
        ["val_f1_weighted", "val_accuracy"],
        ascending=[False, False],
    ).reset_index(drop=True)
    best_model_name = str(results_df.iloc[0]["model"])
    best_model = trained_models[best_model_name]

    best_input = X_test_scaled if best_model_name in scaled_model_names else X_test
    best_pred = best_model.predict(best_input)
    cm = confusion_matrix(y_test, best_pred)
    plot_confusion(cm, best_model_name, ["safe", "medium", "severe"])

    report = classification_report(
        y_test,
        best_pred,
        target_names=["safe", "medium", "severe"],
        output_dict=True,
        zero_division=0,
    )
    with open(ARTIFACTS_DIR / "best_classification_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    joblib.dump(best_model, ARTIFACTS_DIR / "best_classification_model.joblib")
    np.save(ARTIFACTS_DIR / "best_classification_test_predictions.npy", best_pred)

    return results_df, best_model_name, trained_models


def evaluate_regression_as_severity(
    model_name: str,
    trained_models: dict[str, object],
    model_data: dict[str, object],
) -> dict[str, float]:
    scaled_model_names = {"linear_regression", "ridge"}
    model = trained_models[model_name]
    X_test = model_data["X_test_scaled"] if model_name in scaled_model_names else model_data["X_test"]
    y_test_reg = model_data["y_test_reg"]
    y_test_cls = model_data["y_test_cls"]

    pred_water_level = model.predict(X_test)
    pred_severity = severity_from_water_level(pred_water_level)
    return evaluate_classification(y_test_cls, pred_severity)


def save_feature_manifest(feature_cols: list[str]) -> None:
    feature_manifest = {
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "raw_api_candidates": [
            "timestamp",
            "wind_speed",
            "pressure",
            "air_temp",
            "water_temp",
            "recent historical water levels",
            "recent historical wind speed",
            "recent historical pressure",
        ],
        "derived_in_backend": [
            "calendar features",
            "cyclical encodings",
            "lag features",
            "rolling means and std",
            "difference features",
            "severity thresholds from water level",
        ],
    }
    with open(ARTIFACTS_DIR / "feature_manifest.json", "w", encoding="utf-8") as file:
        json.dump(feature_manifest, file, indent=2)


def main() -> None:
    ensure_dirs()
    plt.style.use("ggplot")

    feature_df, _ = build_feature_dataset(thresholds=THRESHOLDS)
    model_data = prepare_model_data(feature_df)
    scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_splits(
        model_data["X_train"],
        model_data["X_val"],
        model_data["X_test"],
    )

    model_data["X_train_scaled"] = X_train_scaled
    model_data["X_val_scaled"] = X_val_scaled
    model_data["X_test_scaled"] = X_test_scaled

    regression_results, best_reg_name, reg_models = train_regression_models(model_data)
    classification_results, best_cls_name, cls_models = train_classification_models(model_data)

    severity_from_best_reg = evaluate_regression_as_severity(best_reg_name, reg_models, model_data)

    regression_results.to_csv(ARTIFACTS_DIR / "regression_results.csv", index=False)
    classification_results.to_csv(ARTIFACTS_DIR / "classification_results.csv", index=False)
    joblib.dump(scaler, ARTIFACTS_DIR / "feature_scaler.joblib")
    save_feature_manifest(model_data["feature_cols"])

    summary = {
        "best_regression_model": best_reg_name,
        "best_classification_model": best_cls_name,
        "best_regression_validation_rmse": float(regression_results.iloc[0]["val_rmse"]),
        "best_classification_validation_f1_weighted": float(classification_results.iloc[0]["val_f1_weighted"]),
        "severity_metrics_from_best_regression": severity_from_best_reg,
        "severity_thresholds": {
            "safe_threshold": THRESHOLDS.safe_threshold,
            "severe_threshold": THRESHOLDS.severe_threshold,
        },
    }
    with open(ARTIFACTS_DIR / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Regression leaderboard")
    print(regression_results)
    print()
    print("Classification leaderboard")
    print(classification_results)
    print()
    print("Severity metrics if best regression predictions are thresholded")
    print(summary["severity_metrics_from_best_regression"])
    print()
    print(f"Best regression model saved to: {(ARTIFACTS_DIR / 'best_regression_model.joblib').resolve()}")
    print(f"Best classification model saved to: {(ARTIFACTS_DIR / 'best_classification_model.joblib').resolve()}")
    print(f"Scaler saved to: {(ARTIFACTS_DIR / 'feature_scaler.joblib').resolve()}")


if __name__ == "__main__":
    main()
