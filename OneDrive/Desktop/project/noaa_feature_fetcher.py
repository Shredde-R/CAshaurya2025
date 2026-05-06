from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from preprocessing import (
    DATA_PATH,
    NUMERIC_COLUMNS,
    add_change_features,
    add_lag_features,
    add_rolling_features,
    add_time_features,
    get_numeric_columns,
)


NOAA_API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
DEFAULT_STATION = "8531680"
DEFAULT_DATUM = "MLLW"
DEFAULT_TIME_ZONE = "gmt"
DEFAULT_UNITS = "metric"
DEFAULT_INTERVAL = "h"
DEFAULT_OUTPUT_DIR = Path("live_features")
REQUIRED_PRODUCTS = {
    "water_level": "Water_Level",
    "hourly_height": "Water_Level",
    "wind": "Wind_Speed",
    "air_pressure": "Pressure",
    "air_temperature": "Air_Temp",
    "water_temperature": "Water_Temp",
}


def fetch_noaa_product(
    station: str,
    product: str,
    begin_date: str,
    end_date: str,
    datum: str = DEFAULT_DATUM,
    units: str = DEFAULT_UNITS,
    time_zone: str = DEFAULT_TIME_ZONE,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    params = {
        "begin_date": begin_date,
        "end_date": end_date,
        "station": station,
        "product": product,
        "units": units,
        "time_zone": time_zone,
        "format": "json",
        "application": "tide_prediction_pipeline",
    }

    if product in {"water_level", "hourly_height", "predictions"}:
        params["datum"] = datum
    if product in {"wind", "air_pressure", "air_temperature", "water_temperature", "predictions"}:
        params["interval"] = interval

    url = f"{NOAA_API_URL}?{urlencode(params)}"
    with urlopen(url) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if "error" in payload:
        raise ValueError(f"NOAA API error for product={product}: {payload['error'].get('message', payload['error'])}")

    records = payload.get("data") or payload.get("predictions")
    if not records:
        raise ValueError(f"No NOAA data returned for product={product}. URL: {url}")

    frame = pd.DataFrame(records)
    frame["Date Time"] = pd.to_datetime(frame["t"])
    return frame


def parse_product_frame(product: str, frame: pd.DataFrame) -> pd.DataFrame:
    if product == "wind":
        parsed = frame[["Date Time", "s"]].copy()
        parsed["Wind_Speed"] = pd.to_numeric(parsed["s"], errors="coerce")
        return parsed.drop(columns=["s"])

    parsed = frame[["Date Time", "v"]].copy()
    parsed[REQUIRED_PRODUCTS[product]] = pd.to_numeric(parsed["v"], errors="coerce")
    return parsed.drop(columns=["v"])


def fetch_required_features(
    station: str,
    begin_date: str,
    end_date: str,
    datum: str = DEFAULT_DATUM,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    for product in REQUIRED_PRODUCTS:
        raw_frame = fetch_noaa_product(
            station=station,
            product=product,
            begin_date=begin_date,
            end_date=end_date,
            datum=datum,
        )
        parsed_frame = parse_product_frame(product, raw_frame)
        merged = parsed_frame if merged is None else merged.merge(parsed_frame, on="Date Time", how="outer")

    if merged is None:
        raise ValueError("No feature data could be fetched from NOAA.")

    merged = merged.sort_values("Date Time").drop_duplicates(subset="Date Time")
    return merged


def preprocess_live_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    processed = raw_df.copy()
    processed["Date Time"] = pd.to_datetime(processed["Date Time"])
    processed = processed.sort_values("Date Time").set_index("Date Time")

    full_index = pd.date_range(processed.index.min(), processed.index.max(), freq="h")
    processed = processed.reindex(full_index)
    processed.index.name = "Date Time"

    present_numeric = [column for column in NUMERIC_COLUMNS if column in processed.columns]
    processed[present_numeric] = processed[present_numeric].interpolate(method="time").ffill().bfill()
    return processed


def build_classical_feature_frame(processed_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = add_time_features(processed_df)
    feature_df = add_lag_features(feature_df)
    feature_df = add_rolling_features(feature_df)
    feature_df = add_change_features(feature_df)
    return feature_df.dropna().copy()


def build_lstm_sequence_source(processed_df: pd.DataFrame) -> pd.DataFrame:
    seq_df = processed_df.copy()
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
    return seq_df[numeric_columns + time_columns].dropna()


def save_outputs(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    classical_df: pd.DataFrame,
    lstm_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(exist_ok=True)
    raw_df.to_csv(output_dir / "noaa_raw_features.csv", index=False)
    processed_df.reset_index().to_csv(output_dir / "noaa_processed_hourly.csv", index=False)
    classical_df.reset_index().to_csv(output_dir / "noaa_classical_features.csv", index=False)
    lstm_df.reset_index().to_csv(output_dir / "noaa_lstm_sequence_source.csv", index=False)


def print_summary(
    station: str,
    processed_df: pd.DataFrame,
    classical_df: pd.DataFrame,
    lstm_df: pd.DataFrame,
) -> None:
    print(f"Station: {station}")
    print(f"Processed hourly rows: {len(processed_df)}")
    print(f"Date range: {processed_df.index.min()} to {processed_df.index.max()}")
    print()
    print("Available columns")
    print(list(processed_df.columns))
    print()
    print("Latest processed row")
    print(processed_df.tail(1))
    print()
    print(f"Classical feature rows after lag/rolling preprocessing: {len(classical_df)}")
    if not classical_df.empty:
        print("Latest classical inference row")
        print(classical_df.tail(1).T.head(20))
    print()
    print(f"LSTM sequence source rows: {len(lstm_df)}")
    print(f"LSTM ready once you have at least 24 rows: {len(lstm_df) >= 24}")


def build_recent_window_from_csv(
    csv_path: Path | str = DATA_PATH,
    hours: int = 72,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date Time"] = pd.to_datetime(df["Date Time"])
    df = df.sort_values("Date Time")
    return df.tail(hours).copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NOAA CO-OPS features and preprocess them for the tide prediction models."
    )
    parser.add_argument("--station", default=DEFAULT_STATION, help="NOAA station id. Default: 8531680 (Sandy Hook)")
    parser.add_argument("--begin-date", required=True, help="Begin date in YYYYMMDD format")
    parser.add_argument("--end-date", required=True, help="End date in YYYYMMDD format")
    parser.add_argument("--datum", default=DEFAULT_DATUM, help="Water level datum for NOAA water products")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save fetched features")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_df = fetch_required_features(
        station=args.station,
        begin_date=args.begin_date,
        end_date=args.end_date,
        datum=args.datum,
    )
    processed_df = preprocess_live_features(raw_df)
    classical_df = build_classical_feature_frame(processed_df)
    lstm_df = build_lstm_sequence_source(processed_df)

    save_outputs(raw_df, processed_df, classical_df, lstm_df, Path(args.output_dir))
    print_summary(args.station, processed_df, classical_df, lstm_df)


if __name__ == "__main__":
    main()
