from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Callable

import cv2
import supervision as sv
from ultralytics import YOLO


DEFAULT_VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}


@dataclass
class TrafficAnalysisConfig:
    video_path: str = "videos/traffic.mp4"
    model_path: str = "yolov8n.pt"
    output_csv: str = "traffic_data.csv"
    output_video: str | None = None
    confidence: float = 0.35
    frame_skip: int = 1
    line_y_ratio: float = 0.55
    log_interval_sec: float = 5.0
    display: bool = False
    max_frames: int | None = None


ProgressCallback = Callable[[dict[str, int | float | str]], None]


def _traffic_status(vehicles_per_minute: int) -> tuple[str, tuple[int, int, int]]:
    if vehicles_per_minute < 10:
        return "LOW", (0, 200, 0)
    if vehicles_per_minute < 20:
        return "MEDIUM", (0, 215, 255)
    return "HIGH", (0, 0, 255)


def analyze_video(
    config: TrafficAnalysisConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int | float | str]:
    video_path = Path(config.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model_path = Path(config.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_total_frames = (
        min(total_frames, config.max_frames)
        if config.max_frames is not None and total_frames
        else (config.max_frames or total_frames)
    )

    line_y = int(height * config.line_y_ratio)
    line_zone = sv.LineZone(
        start=sv.Point(0, line_y),
        end=sv.Point(width, line_y),
    )
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2)

    writer = None
    if config.output_video:
        video_output_path = Path(config.output_video)
        video_output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(video_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    csv_path = Path(config.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    unique_vehicle_ids: set[int] = set()
    processed_frames = 0
    logged_rows = 0
    next_log_time = 0.0
    last_status = "UNKNOWN"
    last_flow_rate = 0
    last_elapsed_seconds = 0.0
    last_vehicles_passed = 0
    last_logged_second = -1

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "Time(sec)",
                "Vehicles Passed",
                "Vehicles Per Minute",
                "Unique Vehicles",
                "Traffic Status",
            ]
        )

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if config.frame_skip > 1 and (frame_number - 1) % config.frame_skip != 0:
                continue

            if config.max_frames is not None and processed_frames >= config.max_frames:
                break
            processed_frames += 1

            result = model.predict(frame, conf=config.confidence, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            if len(detections) > 0:
                mask = [
                    model.names[int(class_id)] in DEFAULT_VEHICLE_CLASSES
                    for class_id in detections.class_id
                ]
                detections = detections[mask]
            detections = tracker.update_with_detections(detections)

            if len(detections) > 0 and detections.tracker_id is not None:
                for tracker_id in detections.tracker_id:
                    if tracker_id is not None:
                        unique_vehicle_ids.add(int(tracker_id))

            line_zone.trigger(detections)

            elapsed_seconds = max(frame_number / fps, 1e-6)
            vehicles_passed = int(line_zone.in_count + line_zone.out_count)
            vehicles_per_minute = int((vehicles_passed / elapsed_seconds) * 60)
            status, color = _traffic_status(vehicles_per_minute)
            last_elapsed_seconds = elapsed_seconds
            last_vehicles_passed = vehicles_passed
            last_status = status
            last_flow_rate = vehicles_per_minute

            while elapsed_seconds >= next_log_time:
                csv_writer.writerow(
                    [
                        int(elapsed_seconds),
                        vehicles_passed,
                        vehicles_per_minute,
                        len(unique_vehicle_ids),
                        status,
                    ]
                )
                logged_rows += 1
                last_logged_second = int(elapsed_seconds)
                next_log_time += config.log_interval_sec

            annotated_frame = result.plot()
            annotated_frame = line_annotator.annotate(annotated_frame, line_zone)
            cv2.putText(
                annotated_frame,
                f"Vehicles Passed: {vehicles_passed}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Unique Vehicles: {len(unique_vehicle_ids)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Traffic Status: {status}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

            if writer is not None:
                writer.write(annotated_frame)

            if config.display:
                cv2.imshow("Traffic Monitor", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if progress_callback is not None and processed_frames % 5 == 0:
                progress_callback(
                    {
                        "processed_frames": processed_frames,
                        "total_frames": target_total_frames,
                        "vehicles_passed": vehicles_passed,
                        "unique_vehicles": len(unique_vehicle_ids),
                        "vehicles_per_minute": vehicles_per_minute,
                        "traffic_status": status,
                    }
                )

        final_second = int(last_elapsed_seconds)
        if last_elapsed_seconds and final_second != last_logged_second:
            csv_writer.writerow(
                [
                    final_second,
                    last_vehicles_passed,
                    last_flow_rate,
                    len(unique_vehicle_ids),
                    last_status,
                ]
            )
            logged_rows += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    duration_seconds = int(frame_number / fps) if frame_number else 0
    return {
        "video_path": str(video_path),
        "output_csv": str(csv_path),
        "output_video": str(config.output_video) if config.output_video else "",
        "processed_frames": processed_frames,
        "total_frames": target_total_frames,
        "duration_seconds": duration_seconds,
        "vehicles_passed": int(line_zone.in_count + line_zone.out_count),
        "unique_vehicles": len(unique_vehicle_ids),
        "vehicles_per_minute": last_flow_rate,
        "traffic_status": last_status,
        "rows_logged": logged_rows,
    }
