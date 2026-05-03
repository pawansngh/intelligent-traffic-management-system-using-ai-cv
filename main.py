from __future__ import annotations

import argparse

from traffic_pipeline import TrafficAnalysisConfig, analyze_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Intelligent Traffic Management System Using AI and CV."
    )
    parser.add_argument(
        "--video",
        default="videos/traffic.mp4",
        help="Path to the input video.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path to the YOLO model file.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/traffic_data.csv",
        help="Where to save the generated traffic log CSV.",
    )
    parser.add_argument(
        "--output-video",
        default="outputs/traffic_annotated.mp4",
        help="Optional path for an annotated output video.",
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
        help="Process every Nth frame to trade accuracy for speed.",
    )
    parser.add_argument(
        "--line-y-ratio",
        type=float,
        default=0.55,
        help="Vertical position of the counting line as a ratio of frame height.",
    )
    parser.add_argument(
        "--log-interval-sec",
        type=float,
        default=5.0,
        help="How often to write a row to the CSV.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show the annotated frames in a live OpenCV window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for processed frames, useful for quick tests.",
    )
    parser.add_argument(
        "--lane-count",
        type=int,
        default=3,
        help="Number of lane regions to monitor across the frame width.",
    )
    parser.add_argument(
        "--lane-boundaries",
        nargs="*",
        type=float,
        default=None,
        help="Optional normalized lane split positions such as 0.30 0.62 for 3 lanes.",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Short label for this run, such as morning_peak_day1.",
    )
    parser.add_argument(
        "--location-name",
        default="",
        help="Human-readable road or area name for the recording location.",
    )
    parser.add_argument(
        "--intersection-id",
        default="",
        help="Intersection or camera identifier for dataset grouping.",
    )
    parser.add_argument(
        "--capture-date",
        default="",
        help="Recording date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--time-of-day",
        default="UNKNOWN",
        help="Context label such as MORNING, AFTERNOON, EVENING, NIGHT, or PEAK.",
    )
    parser.add_argument(
        "--weather",
        default="UNKNOWN",
        help="Weather condition label such as CLEAR, CLOUDY, RAIN, or FOG.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional notes about camera angle, event conditions, or traffic context.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = analyze_video(
        TrafficAnalysisConfig(
            video_path=args.video,
            model_path=args.model,
            output_csv=args.output_csv,
            output_video=args.output_video,
            confidence=args.confidence,
            frame_skip=max(1, args.frame_skip),
            line_y_ratio=args.line_y_ratio,
            log_interval_sec=max(1.0, args.log_interval_sec),
            display=args.display,
            max_frames=args.max_frames,
            lane_count=max(1, args.lane_count),
            lane_boundaries=args.lane_boundaries,
            run_label=args.run_label,
            location_name=args.location_name,
            intersection_id=args.intersection_id,
            capture_date=args.capture_date,
            time_of_day=args.time_of_day,
            weather_condition=args.weather,
            notes=args.notes,
        )
    )

    print("Intelligent Traffic Management System Using AI and CV")
    print("Traffic analysis complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
