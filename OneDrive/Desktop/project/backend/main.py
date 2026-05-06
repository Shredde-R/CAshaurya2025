from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from lstm import LSTMClassifier, LSTMRegressor, SEQUENCE_LENGTH
from noaa_feature_fetcher import build_lstm_sequence_source
from open_meteo_feature_fetcher import (
    DIGHA_LATITUDE,
    DIGHA_LONGITUDE,
    fetch_open_meteo_digha_features,
    preprocess_open_meteo_features,
)


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts_lstm"
REGRESSION_MODEL_PATH = ARTIFACTS_DIR / "best_lstm_regression.pt"
CLASSIFIER_MODEL_PATH = ARTIFACTS_DIR / "best_lstm_classifier.pt"
SCALER_PATH = ARTIFACTS_DIR / "lstm_sequence_scaler.joblib"
SUMMARY_PATH = ARTIFACTS_DIR / "lstm_summary.json"

SEVERITY_THRESHOLDS = {
    "safe": 0.5,
    "severe": 1.0,
}
SEVERITY_COLORS = {
    "safe": "#2faa71",
    "medium": "#ffb11a",
    "severe": "#e14747",
}
DEFAULT_FEATURE_COLUMNS = [
    "Water_Level",
    "Wind_Speed",
    "Pressure",
    "Air_Temp",
    "Water_Temp",
    "hour_sin",
    "hour_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "day_of_week",
    "month",
]
METRIC_UNIT_MAP = {
    "Wind Speed": "m/s",
    "Air Pressure": "hPa",
    "Air Temp": "C",
    "Water Temp": "C",
}


app = FastAPI(title="Coastal Sentinel API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def classify_severity(value: float) -> str:
    if value < SEVERITY_THRESHOLDS["safe"]:
        return "safe"
    if value < SEVERITY_THRESHOLDS["severe"]:
        return "medium"
    return "severe"


def severity_message(severity: str) -> str:
    messages = {
        "safe": "Water levels remain in a normal range for the next forecast step.",
        "medium": "Minor coastal impacts are possible in the next forecast window.",
        "severe": "Elevated tide conditions may create localized coastal flooding soon.",
    }
    return messages[severity]


def format_timestamp(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC")


def scale_frame(df: pd.DataFrame, scaler_state: dict[str, dict[str, float]]) -> pd.DataFrame:
    mean = pd.Series(scaler_state["mean"])
    std = pd.Series(scaler_state["std"])
    return (df - mean) / std


def fetch_recent_open_meteo_window(
    latitude: float,
    longitude: float,
    lookback_hours: int,
    forecast_hours: int,
) -> pd.DataFrame:
    return fetch_open_meteo_digha_features(
        latitude=latitude,
        longitude=longitude,
        past_hours=max(lookback_hours, 48),
        forecast_hours=max(forecast_hours, 24),
    )


def build_sequence_from_processed(processed_df: pd.DataFrame, scaler_state: dict[str, dict[str, float]]) -> tuple[pd.DataFrame, np.ndarray]:
    sequence_source = build_lstm_sequence_source(processed_df)
    if len(sequence_source) < SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {SEQUENCE_LENGTH} hourly rows to build the LSTM sequence.",
        )

    feature_columns = app.state.feature_columns
    missing_columns = [column for column in feature_columns if column not in sequence_source.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing model feature columns: {missing_columns}")
    sequence_source = sequence_source[feature_columns].copy()
    scaled = scale_frame(sequence_source, scaler_state)
    latest_window = scaled.tail(SEQUENCE_LENGTH).to_numpy(dtype=np.float32)
    return sequence_source, latest_window


def recursive_forecast(
    reg_model: LSTMRegressor,
    base_sequence: pd.DataFrame,
    scaler_state: dict[str, dict[str, float]],
    hours_ahead: int,
    device: torch.device,
) -> list[dict[str, object]]:
    future_records: list[dict[str, object]] = []
    sequence = base_sequence.copy()
    last_index = sequence.index[-1]

    for step in range(hours_ahead):
        feature_columns = app.state.feature_columns
        scaled = scale_frame(sequence[feature_columns], scaler_state).tail(SEQUENCE_LENGTH)
        x = torch.tensor(scaled.to_numpy(dtype=np.float32)[None, :, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            predicted_level = float(reg_model(x).cpu().numpy().ravel()[0])

        next_timestamp = last_index + timedelta(hours=1)
        next_row = sequence.iloc[-1].copy()
        next_row["Water_Level"] = predicted_level
        next_row["day_of_week"] = next_timestamp.dayofweek
        next_row["month"] = next_timestamp.month
        next_row["hour_sin"] = np.sin(2 * np.pi * next_timestamp.hour / 24)
        next_row["hour_cos"] = np.cos(2 * np.pi * next_timestamp.hour / 24)
        next_row["dayofyear_sin"] = np.sin(2 * np.pi * next_timestamp.dayofyear / 365.25)
        next_row["dayofyear_cos"] = np.cos(2 * np.pi * next_timestamp.dayofyear / 365.25)

        sequence.loc[next_timestamp] = next_row
        last_index = next_timestamp

        severity = classify_severity(predicted_level)
        future_records.append(
            {
                "timestamp": next_timestamp.isoformat(),
                "waterLevel": round(predicted_level, 3),
                "severity": severity,
            }
        )

    return future_records


def confidence_from_probability(probability: float) -> str:
    if probability >= 0.8:
        return "High"
    if probability >= 0.6:
        return "Medium"
    return "Low"


def build_forecast_summary(forecast_points: list[dict[str, object]]) -> dict[str, object]:
    if not forecast_points:
        return {
            "bestTime": None,
            "dangerousTime": None,
            "peakTime": None,
            "riskWindows": 0,
        }

    best_time = min(forecast_points, key=lambda item: item["waterLevel"])
    peak_time = max(forecast_points, key=lambda item: item["waterLevel"])
    dangerous = [item for item in forecast_points if item["severity"] == "severe"]
    medium_or_higher = [item for item in forecast_points if item["severity"] != "safe"]

    return {
        "bestTime": best_time,
        "dangerousTime": dangerous[0] if dangerous else peak_time,
        "peakTime": peak_time,
        "riskWindows": len(medium_or_higher),
    }


@app.on_event("startup")
def load_assets() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_state = joblib.load(SCALER_PATH)
    with open(SUMMARY_PATH, "r", encoding="utf-8") as file:
        model_summary = __import__("json").load(file)
    feature_columns = model_summary.get("input_features_for_sequence_model", DEFAULT_FEATURE_COLUMNS)

    reg_model = LSTMRegressor(input_size=len(feature_columns)).to(device)
    reg_model.load_state_dict(torch.load(REGRESSION_MODEL_PATH, map_location=device))
    reg_model.eval()

    cls_model = LSTMClassifier(input_size=len(feature_columns)).to(device)
    cls_model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    cls_model.eval()

    app.state.device = device
    app.state.scaler_state = scaler_state
    app.state.reg_model = reg_model
    app.state.cls_model = cls_model
    app.state.model_summary = model_summary
    app.state.feature_columns = feature_columns


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/dashboard/live")
def get_live_dashboard(
    latitude: float = Query(DIGHA_LATITUDE),
    longitude: float = Query(DIGHA_LONGITUDE),
    lookback_hours: int = Query(72, ge=24, le=240),
    forecast_hours: int = Query(6, ge=1, le=24),
) -> dict[str, object]:
    try:
        raw_df = fetch_recent_open_meteo_window(
            latitude=latitude,
            longitude=longitude,
            lookback_hours=lookback_hours,
            forecast_hours=forecast_hours,
        )
        processed_df = preprocess_open_meteo_features(raw_df)
        sequence_source, latest_window = build_sequence_from_processed(processed_df, app.state.scaler_state)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    latest_processed = processed_df.iloc[-1]
    latest_timestamp = processed_df.index[-1]
    latest_sequence_timestamp = sequence_source.index[-1]

    x = torch.tensor(latest_window[None, :, :], dtype=torch.float32, device=app.state.device)

    with torch.no_grad():
        predicted_level = float(app.state.reg_model(x).cpu().numpy().ravel()[0])
        cls_logits = app.state.cls_model(x)
        cls_probs = torch.softmax(cls_logits, dim=1).cpu().numpy().ravel()

    class_index = int(np.argmax(cls_probs))
    class_labels = ["safe", "medium", "severe"]
    classifier_severity = class_labels[class_index]
    derived_severity = classify_severity(predicted_level)
    confidence_score = float(cls_probs[class_index])

    forecast_points = recursive_forecast(
        reg_model=app.state.reg_model,
        base_sequence=sequence_source.copy(),
        scaler_state=app.state.scaler_state,
        hours_ahead=forecast_hours,
        device=app.state.device,
    )
    forecast_summary = build_forecast_summary(forecast_points)

    observed_tail = processed_df.tail(18).copy()
    metrics = [
        {
            "label": "Wind Speed",
            "value": round(float(latest_processed["Wind_Speed"]), 2),
            "unit": METRIC_UNIT_MAP["Wind Speed"],
            "icon": "wind",
        },
        {
            "label": "Air Pressure",
            "value": round(float(latest_processed["Pressure"]), 1),
            "unit": METRIC_UNIT_MAP["Air Pressure"],
            "icon": "pressure",
        },
        {
            "label": "Air Temp",
            "value": round(float(latest_processed["Air_Temp"]), 1),
            "unit": METRIC_UNIT_MAP["Air Temp"],
            "icon": "air",
        },
        {
            "label": "Water Temp",
            "value": round(float(latest_processed["Water_Temp"]), 1),
            "unit": METRIC_UNIT_MAP["Water Temp"],
            "icon": "water",
        },
    ]

    chart_labels = [ts.strftime("%H:%M") for ts in observed_tail.index[:: max(1, len(observed_tail) // 6)]]
    chart_labels += [pd.to_datetime(point["timestamp"]).strftime("%H:%M") for point in forecast_points]

    return {
        "location": {
            "name": "Digha Beach",
            "region": "West Bengal Coastal Region",
            "stationId": "Open-Meteo",
            "latitude": latitude,
            "longitude": longitude,
            "latestObservationTime": format_timestamp(latest_timestamp),
            "modelWindowEnd": format_timestamp(latest_sequence_timestamp),
            "lookbackHours": lookback_hours,
        },
        "prediction": {
            "severity": derived_severity.title(),
            "severityColor": SEVERITY_COLORS[derived_severity],
            "predictedTideLevel": round(predicted_level, 3),
            "unit": "m",
            "lastUpdated": latest_timestamp.strftime("%H:%M UTC"),
            "summary": severity_message(derived_severity),
            "classifierSeverity": classifier_severity.title(),
            "observedWaterLevel": round(float(latest_processed["Water_Level"]), 3),
        },
        "confidence": {
            "score": round(confidence_score * 100, 1),
            "label": confidence_from_probability(confidence_score),
            "classification": f"{classifier_severity.title()} Tide Risk",
            "note": "Severity is estimated using the trained LSTM classifier, while tide height comes from the LSTM regressor.",
            "probabilities": {
                "safe": round(float(cls_probs[0]), 4),
                "medium": round(float(cls_probs[1]), 4),
                "severe": round(float(cls_probs[2]), 4),
            },
        },
        "metrics": metrics,
        "trend": {
            "observed": [round(float(v), 3) for v in observed_tail["Water_Level"].tolist()],
            "predicted": [round(float(observed_tail["Water_Level"].iloc[-1]), 3)] + [point["waterLevel"] for point in forecast_points],
            "observedTimestamps": [ts.isoformat() for ts in observed_tail.index],
            "predictedTimestamps": [latest_timestamp.isoformat()] + [point["timestamp"] for point in forecast_points],
            "labels": [ts.strftime("%H:%M") for ts in observed_tail.index],
        },
        "futureForecast": forecast_points,
        "forecastSummary": forecast_summary,
        "modelPerformance": {
            "regression": {
                "mae": round(float(app.state.model_summary["regression_metrics"]["mae"]), 3),
                "rmse": round(float(app.state.model_summary["regression_metrics"]["rmse"]), 3),
                "r2": round(float(app.state.model_summary["regression_metrics"]["r2"]), 3),
            },
            "classification": {
                "accuracy": round(float(app.state.model_summary["classification_metrics"]["accuracy"]) * 100, 1),
                "f1Macro": round(float(app.state.model_summary["classification_metrics"]["f1_macro"]), 3),
                "f1Weighted": round(float(app.state.model_summary["classification_metrics"]["f1_weighted"]), 3),
            },
            "severityFromRegression": {
                "accuracy": round(float(app.state.model_summary["severity_metrics_from_regression"]["accuracy"]) * 100, 1),
                "f1Macro": round(float(app.state.model_summary["severity_metrics_from_regression"]["f1_macro"]), 3),
                "f1Weighted": round(float(app.state.model_summary["severity_metrics_from_regression"]["f1_weighted"]), 3),
            },
        },
        "alerts": {
            "severity": derived_severity,
            "headline": f"{derived_severity.title()} severity coastal outlook",
            "message": severity_message(derived_severity),
        },
        "info": {
            "title": "How it works",
            "description": "Live Open-Meteo weather and marine data is transformed into a 24-hour feature sequence and scored by trained PyTorch LSTM models for tide height and severity.",
            "points": [
                f"Open-Meteo weather and marine feed near {latitude:.3f}, {longitude:.3f}",
                "24-hour sequence LSTM regression for tide level",
                "LSTM classification for safe, medium, and severe labels",
            ],
        },
    }
