from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAFFIC_THRESHOLDS = {"LOW": 10, "MEDIUM": 20}
CATEGORICAL_COLUMNS = ["traffic_status", "time_of_day", "weather_condition", "day_of_week"]
NUMERIC_BASE_COLUMNS = [
    "time_sec",
    "vehicles_passed",
    "vehicles_per_minute",
    "unique_vehicles",
    "emergency_detected",
    "emergency_lane",
    "lane_count",
]


@dataclass
class ForecastModelBundle:
    model_path: Path
    intercept: float
    coefficients: dict[str, float]
    categorical_values: dict[str, list[str]]
    sample_count: int
    metrics: dict[str, float]
    trained_at: str


def traffic_status_from_vpm(value: float) -> str:
    if value >= TRAFFIC_THRESHOLDS["MEDIUM"]:
        return "HIGH"
    if value >= TRAFFIC_THRESHOLDS["LOW"]:
        return "MEDIUM"
    return "LOW"


def _lane_vehicle_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith("lane_") and column.endswith("_vehicles"))


def _categorical_values(frame: pd.DataFrame) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for column in CATEGORICAL_COLUMNS:
        if column not in frame.columns:
            values[column] = ["UNKNOWN"]
            continue
        cleaned = frame[column].fillna("UNKNOWN").astype(str).str.upper()
        unique_values = sorted({value if value else "UNKNOWN" for value in cleaned})
        values[column] = unique_values or ["UNKNOWN"]
    return values


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    coerced = frame.copy()
    for column in columns:
        if column not in coerced.columns:
            coerced[column] = 0
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce").fillna(0.0)
    return coerced


def _build_feature_names(lane_columns: list[str], categorical_values: dict[str, list[str]]) -> list[str]:
    feature_names = NUMERIC_BASE_COLUMNS + lane_columns
    for column in CATEGORICAL_COLUMNS:
        for category in categorical_values.get(column, ["UNKNOWN"]):
            feature_names.append(f"{column}={category}")
    return feature_names


def _row_to_feature_map(
    row: dict[str, object],
    lane_columns: list[str],
    categorical_values: dict[str, list[str]],
) -> dict[str, float]:
    feature_map: dict[str, float] = {}
    for column in NUMERIC_BASE_COLUMNS + lane_columns:
        feature_map[column] = float(row.get(column, 0) or 0)
    for column in CATEGORICAL_COLUMNS:
        value = str(row.get(column, "UNKNOWN") or "UNKNOWN").upper()
        for category in categorical_values.get(column, ["UNKNOWN"]):
            feature_map[f"{column}={category}"] = 1.0 if value == category else 0.0
    return feature_map


def prepare_training_frame(training_path: Path) -> pd.DataFrame:
    if not training_path.exists() or training_path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(training_path)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["next_vehicles_per_minute"] = pd.to_numeric(frame["next_vehicles_per_minute"], errors="coerce")
    frame = frame.dropna(subset=["next_vehicles_per_minute"]).reset_index(drop=True)
    if frame.empty:
        return frame
    lane_columns = _lane_vehicle_columns(frame)
    frame = _coerce_numeric(frame, NUMERIC_BASE_COLUMNS + lane_columns)
    for column in CATEGORICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = "UNKNOWN"
        frame[column] = frame[column].fillna("UNKNOWN").astype(str).str.upper()
    return frame


def train_forecast_model(
    training_path: str = "history/forecast_training_data.csv",
    model_output_path: str = "models/traffic_forecast_model.json",
    alpha: float = 1.0,
) -> ForecastModelBundle:
    training_file = Path(training_path)
    frame = prepare_training_frame(training_file)
    if len(frame) < 2:
        raise ValueError("At least 2 labeled samples are required to train the forecasting model.")

    lane_columns = _lane_vehicle_columns(frame)
    categorical_values = _categorical_values(frame)
    feature_names = _build_feature_names(lane_columns, categorical_values)

    feature_rows = []
    for _, series in frame.iterrows():
        feature_map = _row_to_feature_map(series.to_dict(), lane_columns, categorical_values)
        feature_rows.append([feature_map[name] for name in feature_names])

    x = np.asarray(feature_rows, dtype=float)
    y = frame["next_vehicles_per_minute"].to_numpy(dtype=float)

    x_design = np.column_stack([np.ones(len(x)), x])
    regularizer = np.eye(x_design.shape[1], dtype=float)
    regularizer[0, 0] = 0.0
    weights = np.linalg.pinv(x_design.T @ x_design + (alpha * regularizer)) @ x_design.T @ y
    predictions = x_design @ weights

    errors = predictions - y
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - (np.sum(errors ** 2) / denominator)) if denominator > 0 else 0.0

    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sample_count": int(len(frame)),
        "alpha": alpha,
        "feature_names": feature_names,
        "intercept": float(weights[0]),
        "coefficients": {name: float(value) for name, value in zip(feature_names, weights[1:])},
        "categorical_values": categorical_values,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
    }
    model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return load_forecast_model(model_path)


def load_forecast_model(model_path: str | Path = "models/traffic_forecast_model.json") -> ForecastModelBundle | None:
    resolved = Path(model_path)
    if not resolved.exists() or resolved.stat().st_size == 0:
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return ForecastModelBundle(
        model_path=resolved,
        intercept=float(payload.get("intercept", 0.0)),
        coefficients={key: float(value) for key, value in payload.get("coefficients", {}).items()},
        categorical_values={key: [str(item) for item in value] for key, value in payload.get("categorical_values", {}).items()},
        sample_count=int(payload.get("sample_count", 0)),
        metrics={key: float(value) for key, value in payload.get("metrics", {}).items()},
        trained_at=str(payload.get("trained_at", "")),
    )


def predict_next_vehicles_per_minute(
    bundle: ForecastModelBundle,
    state: dict[str, object],
) -> float:
    lane_columns = sorted(column for column in bundle.coefficients if column.startswith("lane_") and column.endswith("_vehicles"))
    lane_columns = [column for column in lane_columns if "=" not in column]
    feature_map = _row_to_feature_map(state, lane_columns, bundle.categorical_values)
    prediction = bundle.intercept
    for name, coefficient in bundle.coefficients.items():
        prediction += coefficient * feature_map.get(name, 0.0)
    return max(0.0, float(prediction))


def forecast_future_steps(
    current_data: pd.DataFrame,
    metadata: dict[str, object] | None = None,
    bundle: ForecastModelBundle | None = None,
    steps: int = 3,
) -> tuple[pd.DataFrame, str]:
    if current_data.empty:
        return pd.DataFrame(columns=["Time(sec)", "Vehicles Per Minute", "Type", "Traffic Status"]), "No data"

    history = current_data[["Time(sec)", "Vehicles Per Minute", "Traffic Status"]].copy()
    history["Type"] = "Observed"
    if bundle is None:
        return history, "No model"

    metadata = metadata or {}
    if len(current_data) <= 1:
        base_interval = 5
    else:
        positive_intervals = current_data["Time(sec)"].diff().dropna()
        positive_intervals = positive_intervals[positive_intervals > 0]
        base_interval = int(positive_intervals.median()) if not positive_intervals.empty else 5
        base_interval = max(base_interval, 1)

    latest_row = current_data.iloc[-1]
    lane_columns = sorted(column for column in bundle.coefficients if column.startswith("lane_") and column.endswith("_vehicles") and "=" not in column)
    state: dict[str, object] = {
        "time_sec": int(latest_row["Time(sec)"]),
        "vehicles_passed": int(latest_row["Vehicles Passed"]),
        "vehicles_per_minute": int(latest_row["Vehicles Per Minute"]),
        "unique_vehicles": int(latest_row["Unique Vehicles"]),
        "traffic_status": str(latest_row["Traffic Status"]).upper(),
        "emergency_detected": int(latest_row.get("Emergency Detected", 0)),
        "emergency_lane": int(latest_row.get("Emergency Lane", 0)),
        "lane_count": len(parse_lane_counts(latest_row.get("Lane Vehicle Counts", "[]"))),
        "time_of_day": str(metadata.get("time_of_day", "UNKNOWN")).upper(),
        "weather_condition": str(metadata.get("weather_condition", "UNKNOWN")).upper(),
        "day_of_week": str(metadata.get("day_of_week", "UNKNOWN")).upper(),
    }
    lane_counts = parse_lane_counts(latest_row.get("Lane Vehicle Counts", "[]"))
    for index, column in enumerate(lane_columns):
        state[column] = lane_counts[index] if index < len(lane_counts) else 0

    forecast_rows = []
    for step in range(1, steps + 1):
        state["time_sec"] = int(state["time_sec"]) + base_interval
        next_vpm = predict_next_vehicles_per_minute(bundle, state)
        state["vehicles_per_minute"] = float(next_vpm)
        state["traffic_status"] = traffic_status_from_vpm(next_vpm)
        forecast_rows.append(
            {
                "Time(sec)": int(latest_row["Time(sec)"]) + (step * base_interval),
                "Vehicles Per Minute": int(round(next_vpm)),
                "Traffic Status": state["traffic_status"],
                "Type": "Forecast",
            }
        )

    return pd.concat([history, pd.DataFrame(forecast_rows)], ignore_index=True), "ML baseline"


def parse_lane_counts(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [int(item) for item in parsed]
        except json.JSONDecodeError:
            return []
    return []
