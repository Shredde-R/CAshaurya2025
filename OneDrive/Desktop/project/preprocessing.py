from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATA_PATH = Path("digha_beach_final_interpolated.csv")
DATE_COLUMN = "Date Time"
DATE_COLUMN_CANDIDATES = ["Date Time", "Date_Time", "Datetime", "datetime", "date_time"]
TARGET_COLUMN = "Water_Level"
EXPECTED_NUMERIC_COLUMNS = ["Water_Level", "Wind_Speed", "Pressure", "Air_Temp", "Water_Temp"]
NUMERIC_COLUMNS = EXPECTED_NUMERIC_COLUMNS
DEFAULT_LAG_HOURS = (1, 2, 3, 6, 12, 24)
DEFAULT_ROLLING_WINDOWS = (3, 6, 12, 24)
FORECAST_HORIZON = 1


@dataclass
class TideThresholds:
    safe_threshold: float = 0.5
    severe_threshold: float = 1.0


@dataclass
class SimpleStandardScaler:
    mean_: pd.Series | None = None
    scale_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "SimpleStandardScaler":
        self.mean_ = X.mean()
        scale = X.std(ddof=0).replace(0, 1.0)
        self.scale_ = scale
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before calling transform.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def load_dataset(csv_path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    date_column = next((col for col in DATE_COLUMN_CANDIDATES if col in df.columns), None)
    if date_column is None:
        raise ValueError(f"No datetime column found. Expected one of: {DATE_COLUMN_CANDIDATES}")
    df = df.rename(columns={date_column: DATE_COLUMN})
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in EXPECTED_NUMERIC_COLUMNS if column in df.columns]


def preprocess_dataset(
    df: pd.DataFrame,
    numeric_cols: Iterable[str] | None = None,
    enforce_hourly: bool = True,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    processed = df.copy()
    if DATE_COLUMN not in processed.columns:
        date_column = next((col for col in DATE_COLUMN_CANDIDATES if col in processed.columns), None)
        if date_column is None:
            raise ValueError(f"No datetime column found. Expected one of: {DATE_COLUMN_CANDIDATES}")
        processed = processed.rename(columns={date_column: DATE_COLUMN})
    processed[DATE_COLUMN] = pd.to_datetime(processed[DATE_COLUMN])
    processed = processed.sort_values(DATE_COLUMN).drop_duplicates(subset=DATE_COLUMN)
    processed = processed.set_index(DATE_COLUMN)
    numeric_cols = list(numeric_cols or get_numeric_columns(processed))

    if not enforce_hourly:
        return processed, pd.DatetimeIndex([])

    full_index = pd.date_range(processed.index.min(), processed.index.max(), freq="h")
    missing_timestamps = full_index.difference(processed.index)

    processed = processed.reindex(full_index)
    processed.index.name = DATE_COLUMN
    processed[list(numeric_cols)] = (
        processed[list(numeric_cols)].interpolate(method="time").ffill().bfill()
    )
    return processed, missing_timestamps


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    featured["hour"] = featured.index.hour
    featured["day_of_week"] = featured.index.dayofweek
    featured["day_of_month"] = featured.index.day
    featured["month"] = featured.index.month
    featured["day_of_year"] = featured.index.dayofyear
    featured["is_weekend"] = (featured["day_of_week"] >= 5).astype(int)

    featured["hour_sin"] = np.sin(2 * np.pi * featured["hour"] / 24)
    featured["hour_cos"] = np.cos(2 * np.pi * featured["hour"] / 24)
    featured["dayofyear_sin"] = np.sin(2 * np.pi * featured["day_of_year"] / 365.25)
    featured["dayofyear_cos"] = np.cos(2 * np.pi * featured["day_of_year"] / 365.25)
    featured["month_sin"] = np.sin(2 * np.pi * featured["month"] / 12)
    featured["month_cos"] = np.cos(2 * np.pi * featured["month"] / 12)
    return featured


def add_lag_features(
    df: pd.DataFrame,
    lag_hours: Iterable[int] = DEFAULT_LAG_HOURS,
) -> pd.DataFrame:
    featured = df.copy()
    for lag in lag_hours:
        if "Water_Level" in featured.columns:
            featured[f"water_level_lag_{lag}"] = featured["Water_Level"].shift(lag)
        if "Wind_Speed" in featured.columns:
            featured[f"wind_speed_lag_{lag}"] = featured["Wind_Speed"].shift(lag)
        if "Pressure" in featured.columns:
            featured[f"pressure_lag_{lag}"] = featured["Pressure"].shift(lag)
    return featured


def add_rolling_features(
    df: pd.DataFrame,
    rolling_windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    featured = df.copy()
    for window in rolling_windows:
        if "Water_Level" in featured.columns:
            featured[f"water_level_roll_mean_{window}"] = (
                featured["Water_Level"].shift(1).rolling(window).mean()
            )
            featured[f"water_level_roll_std_{window}"] = (
                featured["Water_Level"].shift(1).rolling(window).std()
            )
        if "Wind_Speed" in featured.columns:
            featured[f"wind_speed_roll_mean_{window}"] = (
                featured["Wind_Speed"].shift(1).rolling(window).mean()
            )
        if "Pressure" in featured.columns:
            featured[f"pressure_roll_mean_{window}"] = (
                featured["Pressure"].shift(1).rolling(window).mean()
            )
    return featured


def add_change_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    if "Water_Level" in featured.columns:
        featured["water_level_diff_1"] = featured["Water_Level"].diff(1)
        featured["water_level_diff_3"] = featured["Water_Level"].diff(3)
    if "Pressure" in featured.columns:
        featured["pressure_diff_1"] = featured["Pressure"].diff(1)
    if "Wind_Speed" in featured.columns:
        featured["wind_speed_diff_1"] = featured["Wind_Speed"].diff(1)
    if {"Air_Temp", "Water_Temp"}.issubset(featured.columns):
        featured["air_water_temp_gap"] = featured["Air_Temp"] - featured["Water_Temp"]
    return featured


def classify_severity(
    water_level: float,
    thresholds: TideThresholds = TideThresholds(),
) -> str:
    if water_level < thresholds.safe_threshold:
        return "safe"
    if water_level < thresholds.severe_threshold:
        return "medium"
    return "severe"


def add_targets(
    df: pd.DataFrame,
    thresholds: TideThresholds = TideThresholds(),
    forecast_horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    featured = df.copy()
    featured["target_water_level"] = featured[TARGET_COLUMN].shift(-forecast_horizon)
    featured["severity_class"] = featured["target_water_level"].apply(
        lambda value: classify_severity(value, thresholds)
    )
    featured["severity_code"] = featured["severity_class"].map(
        {"safe": 0, "medium": 1, "severe": 2}
    )
    return featured


def build_feature_dataset(
    csv_path: Path | str = DATA_PATH,
    thresholds: TideThresholds = TideThresholds(),
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    raw = load_dataset(csv_path)
    processed, missing_timestamps = preprocess_dataset(raw)
    featured = add_time_features(processed)
    featured = add_lag_features(featured)
    featured = add_rolling_features(featured)
    featured = add_change_features(featured)
    featured = add_targets(featured, thresholds)
    return featured, missing_timestamps


def prepare_model_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, pd.DataFrame | pd.Series | list[str]]:
    model_df = df.dropna().copy()

    feature_cols = [col for col in model_df.columns if col not in ["severity_class", "severity_code", "target_water_level"]]
    X = model_df[feature_cols]
    y_reg = model_df["target_water_level"]
    y_cls = model_df["severity_code"]

    n_rows = len(model_df)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    return {
        "model_df": model_df,
        "feature_cols": feature_cols,
        "X_train": X.iloc[:train_end].copy(),
        "X_val": X.iloc[train_end:val_end].copy(),
        "X_test": X.iloc[val_end:].copy(),
        "y_train_reg": y_reg.iloc[:train_end].copy(),
        "y_val_reg": y_reg.iloc[train_end:val_end].copy(),
        "y_test_reg": y_reg.iloc[val_end:].copy(),
        "y_train_cls": y_cls.iloc[:train_end].copy(),
        "y_val_cls": y_cls.iloc[train_end:val_end].copy(),
        "y_test_cls": y_cls.iloc[val_end:].copy(),
    }


def scale_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[SimpleStandardScaler, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = SimpleStandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        index=X_val.index,
        columns=X_val.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns,
    )
    return scaler, X_train_scaled, X_val_scaled, X_test_scaled


def summarize_dataset(df: pd.DataFrame, missing_timestamps: pd.DatetimeIndex) -> dict[str, object]:
    numeric_columns = get_numeric_columns(df)
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "start": df.index.min(),
        "end": df.index.max(),
        "missing_timestamps_count": len(missing_timestamps),
        "numeric_summary": df[numeric_columns].describe().round(3),
    }


def main() -> None:
    feature_df, missing_timestamps = build_feature_dataset()
    summary = summarize_dataset(feature_df, missing_timestamps)
    model_parts = prepare_model_data(feature_df)
    scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_splits(
        model_parts["X_train"],
        model_parts["X_val"],
        model_parts["X_test"],
    )

    print("Dataset summary")
    print(f"Rows: {summary['rows']}")
    print(f"Start: {summary['start']}")
    print(f"End: {summary['end']}")
    print(f"Missing timestamps inserted: {summary['missing_timestamps_count']}")
    print()
    print("Numeric summary")
    print(summary["numeric_summary"])
    print()
    print("Model split shapes")
    print("Train:", model_parts["X_train"].shape)
    print("Validation:", model_parts["X_val"].shape)
    print("Test:", model_parts["X_test"].shape)
    print()
    print("Scaled train sample")
    print(X_train_scaled.head())
    print()
    print("Severity distribution")
    print(feature_df["severity_class"].value_counts().sort_index())
    _ = scaler, X_val_scaled, X_test_scaled


if __name__ == "__main__":
    main()
