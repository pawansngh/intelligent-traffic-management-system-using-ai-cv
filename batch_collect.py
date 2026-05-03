from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from traffic_pipeline import TrafficAnalysisConfig, analyze_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-ingest traffic videos into the historical forecasting dataset."
    )
    parser.add_argument(
        "--manifest",
        default="dataset_manifest.csv",
        help="CSV manifest describing traffic videos and their collection metadata.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory for per-run CSV and annotated video artifacts.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path to the YOLO model file.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame across all runs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for quick batch tests.",
    )
    parser.add_argument(
        "--skip-video-output",
        action="store_true",
        help="Skip annotated MP4 generation to ingest datasets faster.",
    )
    return parser


def parse_lane_boundaries(value: str) -> list[float] | None:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid lane boundaries JSON: {cleaned}") from exc
    if not isinstance(parsed, list):
        raise ValueError("Lane boundaries must be a JSON list.")
    return [float(item) for item in parsed]


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest)
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    if not rows:
        raise RuntimeError(f"No rows found in manifest: {manifest_path}")

    for index, row in enumerate(rows, start=1):
        video_path = Path(str(row.get("video_path", "")).strip())
        if not video_path.exists():
            print(f"[{index}/{len(rows)}] Skipping missing video: {video_path}")
            continue

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(video_path).stem
        csv_output = outputs_dir / f"{run_id}_{stem}_data.csv"
        video_output = None if args.skip_video_output else outputs_dir / f"{run_id}_{stem}_annotated.mp4"

        summary = analyze_video(
            TrafficAnalysisConfig(
                video_path=str(video_path),
                model_path=args.model,
                output_csv=str(csv_output),
                output_video=str(video_output) if video_output else None,
                confidence=args.confidence,
                frame_skip=max(1, args.frame_skip),
                max_frames=args.max_frames,
                lane_count=max(1, int(row.get("lane_count", 3) or 3)),
                lane_boundaries=parse_lane_boundaries(row.get("lane_boundaries", "")),
                line_y_ratio=float(row.get("line_y_ratio", 0.55) or 0.55),
                log_interval_sec=float(row.get("log_interval_sec", 5.0) or 5.0),
                run_label=str(row.get("run_label", "")).strip(),
                location_name=str(row.get("location_name", "")).strip(),
                intersection_id=str(row.get("intersection_id", "")).strip(),
                capture_date=str(row.get("capture_date", "")).strip(),
                time_of_day=str(row.get("time_of_day", "UNKNOWN")).strip() or "UNKNOWN",
                weather_condition=str(row.get("weather_condition", "UNKNOWN")).strip() or "UNKNOWN",
                notes=str(row.get("notes", "")).strip(),
            )
        )
        print(f"[{index}/{len(rows)}] Completed {video_path.name}")
        print(
            f"  run_label={summary['run_label']} time_of_day={summary['time_of_day']} "
            f"vehicles={summary['vehicles_passed']} history={summary['history_path']}"
        )


if __name__ == "__main__":
    main()
