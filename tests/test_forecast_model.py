from __future__ import annotations

import json
import unittest
from pathlib import Path
import shutil

import pandas as pd

from forecast_model import (
    ForecastModelBundle,
    forecast_future_steps,
    load_forecast_model,
    parse_lane_counts,
    prepare_training_frame,
    traffic_status_from_vpm,
    train_forecast_model,
)


class ForecastModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests/.tmp/forecast_model")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_parse_lane_counts_handles_json_string(self) -> None:
        self.assertEqual(parse_lane_counts("[1, 2, 3]"), [1, 2, 3])
        self.assertEqual(parse_lane_counts("not-json"), [])

    def test_traffic_status_from_vpm_thresholds(self) -> None:
        self.assertEqual(traffic_status_from_vpm(5), "LOW")
        self.assertEqual(traffic_status_from_vpm(10), "MEDIUM")
        self.assertEqual(traffic_status_from_vpm(20), "HIGH")

    def test_prepare_training_frame_drops_unlabeled_rows(self) -> None:
        training_csv = self.temp_dir / "training.csv"
        training_csv.write_text(
            "\n".join(
                [
                    "time_sec,vehicles_passed,vehicles_per_minute,unique_vehicles,emergency_detected,emergency_lane,lane_count,next_vehicles_per_minute,traffic_status,time_of_day,weather_condition,day_of_week,lane_1_vehicles",
                    "0,0,12,3,0,0,3,15,MEDIUM,MORNING,CLEAR,MONDAY,4",
                    "5,1,15,4,0,0,3,,HIGH,EVENING,RAIN,TUESDAY,5",
                ]
            ),
            encoding="utf-8",
        )

        prepared = prepare_training_frame(training_csv)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(int(prepared.iloc[0]["next_vehicles_per_minute"]), 15)

    def test_train_and_load_forecast_model(self) -> None:
        training_csv = self.temp_dir / "training.csv"
        training_csv.write_text(
            "\n".join(
                [
                    "time_sec,vehicles_passed,vehicles_per_minute,unique_vehicles,emergency_detected,emergency_lane,lane_count,next_vehicles_per_minute,traffic_status,time_of_day,weather_condition,day_of_week,lane_1_vehicles,lane_2_vehicles,lane_3_vehicles",
                    "0,0,8,3,0,0,3,10,LOW,MORNING,CLEAR,MONDAY,2,3,1",
                    "5,1,10,4,0,0,3,12,MEDIUM,MORNING,CLEAR,MONDAY,3,4,2",
                    "10,2,12,5,0,0,3,14,MEDIUM,EVENING,RAIN,TUESDAY,4,4,3",
                ]
            ),
            encoding="utf-8",
        )
        model_path = self.temp_dir / "model.json"

        bundle = train_forecast_model(
            training_path=str(training_csv),
            model_output_path=str(model_path),
            alpha=0.5,
        )

        self.assertTrue(model_path.exists())
        self.assertEqual(bundle.sample_count, 3)
        loaded = load_forecast_model(model_path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.model_path, model_path)
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        self.assertIn("coefficients", payload)

    def test_forecast_future_steps_returns_observed_and_forecast_rows(self) -> None:
        bundle = ForecastModelBundle(
            model_path=self.temp_dir / "model.json",
            intercept=2.0,
            coefficients={
                "time_sec": 0.0,
                "vehicles_passed": 0.0,
                "vehicles_per_minute": 0.5,
                "unique_vehicles": 0.0,
                "emergency_detected": 0.0,
                "emergency_lane": 0.0,
                "lane_count": 0.0,
                "lane_1_vehicles": 0.2,
                "lane_2_vehicles": 0.1,
                "lane_3_vehicles": 0.1,
                "traffic_status=LOW": 1.0,
                "time_of_day=MORNING": 0.0,
                "weather_condition=CLEAR": 0.0,
                "day_of_week=MONDAY": 0.0,
            },
            categorical_values={
                "traffic_status": ["LOW"],
                "time_of_day": ["MORNING"],
                "weather_condition": ["CLEAR"],
                "day_of_week": ["MONDAY"],
            },
            sample_count=4,
            metrics={"mae": 1.0, "rmse": 1.2, "r2": 0.4},
            trained_at="2026-05-03T10:00:00",
        )
        current_data = pd.DataFrame(
            [
                {
                    "Time(sec)": 0,
                    "Vehicles Passed": 0,
                    "Vehicles Per Minute": 8,
                    "Unique Vehicles": 3,
                    "Traffic Status": "LOW",
                    "Emergency Detected": 0,
                    "Emergency Lane": 0,
                    "Lane Vehicle Counts": "[2, 1, 1]",
                },
                {
                    "Time(sec)": 5,
                    "Vehicles Passed": 1,
                    "Vehicles Per Minute": 10,
                    "Unique Vehicles": 4,
                    "Traffic Status": "MEDIUM",
                    "Emergency Detected": 0,
                    "Emergency Lane": 0,
                    "Lane Vehicle Counts": "[3, 2, 1]",
                },
            ]
        )

        forecast_df, source = forecast_future_steps(
            current_data=current_data,
            metadata={"time_of_day": "MORNING", "weather_condition": "CLEAR", "day_of_week": "MONDAY"},
            bundle=bundle,
            steps=2,
        )

        self.assertEqual(source, "ML baseline")
        self.assertEqual(len(forecast_df[forecast_df["Type"] == "Forecast"]), 2)
        self.assertGreaterEqual(int(forecast_df.iloc[-1]["Vehicles Per Minute"]), 0)


if __name__ == "__main__":
    unittest.main()
