from __future__ import annotations

import unittest
from pathlib import Path
import shutil

from traffic_pipeline import (
    TrafficAnalysisConfig,
    _append_to_history,
    _merge_fieldnames,
    _normalize_text,
    _resolve_lane_boundaries,
    _safe_day_of_week,
)


class TrafficPipelineHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests/.tmp/traffic_pipeline")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_resolve_lane_boundaries_defaults_even_split(self) -> None:
        self.assertEqual(_resolve_lane_boundaries(900, 3, None), [0, 300, 600, 900])

    def test_resolve_lane_boundaries_normalizes_unsorted_inputs(self) -> None:
        boundaries = _resolve_lane_boundaries(1000, 3, [0.8, 0.2])
        self.assertEqual(boundaries[0], 0)
        self.assertEqual(boundaries[-1], 1000)
        self.assertLess(boundaries[1], boundaries[2])

    def test_merge_fieldnames_preserves_order(self) -> None:
        self.assertEqual(_merge_fieldnames(["a", "b"], ["b", "c"]), ["a", "b", "c"])

    def test_text_and_day_helpers(self) -> None:
        self.assertEqual(_normalize_text("  hi  "), "hi")
        self.assertEqual(_normalize_text("", fallback="UNKNOWN"), "UNKNOWN")
        self.assertEqual(_safe_day_of_week("2026-05-03"), "SUNDAY")
        self.assertEqual(_safe_day_of_week("bad-date"), "UNKNOWN")

    def test_append_to_history_creates_history_and_training_rows(self) -> None:
        source_csv = self.temp_dir / "run.csv"
        source_csv.write_text(
            "\n".join(
                [
                    "Time(sec),Vehicles Passed,Vehicles Per Minute,Unique Vehicles,Traffic Status,Lane Vehicle Counts,Lane Congestion,Emergency Detected,Emergency Lane,Emergency Count",
                    '0,0,8,3,LOW,"[2, 1, 0]","[""LOW"", ""LOW"", ""LOW""]",0,0,0',
                    '5,1,12,4,MEDIUM,"[3, 2, 1]","[""MEDIUM"", ""LOW"", ""LOW""]",0,0,0',
                ]
            ),
            encoding="utf-8",
        )
        config = TrafficAnalysisConfig(
            video_path="videos/traffic.mp4",
            history_dir=str(self.temp_dir / "history"),
            run_label="test_run",
            location_name="Main Junction",
            intersection_id="INT-7",
            capture_date="2026-05-03",
            time_of_day="MORNING",
            weather_condition="CLEAR",
        )

        history_path, training_path = _append_to_history(
            csv_path=source_csv,
            config=config,
            lane_boundaries=[0, 300, 600, 900],
            emergency_supported=False,
        )

        history_text = Path(history_path).read_text(encoding="utf-8")
        training_text = Path(training_path).read_text(encoding="utf-8")
        self.assertIn("run_label", history_text)
        self.assertIn("test_run", history_text)
        self.assertIn("next_vehicles_per_minute", training_text)


if __name__ == "__main__":
    unittest.main()
