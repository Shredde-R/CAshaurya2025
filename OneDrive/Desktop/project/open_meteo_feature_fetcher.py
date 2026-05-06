from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from preprocessing import DATE_COLUMN


DIGHA_LATITUDE = 21.627
DIGHA_LONGITUDE = 87.508
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"


def fetch_json(url: str, params: dict[str, object]) -> dict[str, object]:
    request_url = f"{url}?{urlencode(params)}"
    with urlopen(request_url) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("error"):
        raise ValueError(payload.get("reason", f"Open-Meteo error for {request_url}"))
    return payload


def hourly_payload_to_frame(payload: dict[str, object], rename_map: dict[str, str]) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError("Open-Meteo response does not contain hourly time-series data.")

    frame = pd.DataFrame({DATE_COLUMN: pd.to_datetime(hourly["time"])})
    for source, target in rename_map.items():
        if source in hourly:
            frame[target] = pd.to_numeric(hourly[source], errors="coerce")
    return frame


def fetch_open_meteo_digha_features(
    latitude: float = DIGHA_LATITUDE,
    longitude: float = DIGHA_LONGITUDE,
    past_hours: int = 72,
    forecast_hours: int = 24,
    timezone: str = "auto",
) -> pd.DataFrame:
    weather_payload = fetch_json(
        WEATHER_API_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,pressure_msl,wind_speed_10m",
            "wind_speed_unit": "ms",
            "past_hours": past_hours,
            "forecast_hours": forecast_hours,
            "timezone": timezone,
        },
    )
    weather_frame = hourly_payload_to_frame(
        weather_payload,
        {
            "temperature_2m": "Air_Temp",
            "pressure_msl": "Pressure",
            "wind_speed_10m": "Wind_Speed",
        },
    )

    marine_payload = fetch_json(
        MARINE_API_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "sea_level_height_msl,sea_surface_temperature",
            "past_hours": past_hours,
            "forecast_hours": forecast_hours,
            "timezone": timezone,
        },
    )
    marine_frame = hourly_payload_to_frame(
        marine_payload,
        {
            "sea_level_height_msl": "Water_Level",
            "sea_surface_temperature": "Water_Temp",
        },
    )

    merged = weather_frame.merge(marine_frame, on=DATE_COLUMN, how="outer")
    merged = merged.sort_values(DATE_COLUMN).drop_duplicates(subset=DATE_COLUMN)
    return merged


def preprocess_open_meteo_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    processed = raw_df.copy()
    processed[DATE_COLUMN] = pd.to_datetime(processed[DATE_COLUMN])
    processed = processed.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
    full_index = pd.date_range(processed.index.min(), processed.index.max(), freq="h")
    processed = processed.reindex(full_index)
    processed.index.name = DATE_COLUMN
    processed = processed.interpolate(method="time").ffill().bfill()
    return processed
