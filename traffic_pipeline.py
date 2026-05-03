from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
from datetime import datetime
from typing import Callable

import cv2
import supervision as sv
from ultralytics import YOLO


DEFAULT_VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
DEFAULT_EMERGENCY_KEYWORDS = (
    "ambulance",
    "fire truck",
    "fire engine",
    "police car",
    "police vehicle",
    "emergency vehicle",
)


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
    lane_count: int = 3
    lane_boundaries: list[float] | None = None
    emergency_keywords: tuple[str, ...] = DEFAULT_EMERGENCY_KEYWORDS
    history_dir: str = "history"
    run_label: str = ""
    location_name: str = ""
    intersection_id: str = ""
    capture_date: str = ""
    time_of_day: str = "UNKNOWN"
    weather_condition: str = "UNKNOWN"
    notes: str = ""


ProgressCallback = Callable[[dict[str, int | float | str]], None]


def _traffic_status(vehicles_per_minute: int) -> tuple[str, tuple[int, int, int]]:
    if vehicles_per_minute < 10:
        return "LOW", (0, 200, 0)
    if vehicles_per_minute < 20:
        return "MEDIUM", (0, 215, 255)
    return "HIGH", (0, 0, 255)


def _lane_status(vehicle_count: int) -> str:
    if vehicle_count <= 2:
        return "LOW"
    if vehicle_count <= 5:
        return "MEDIUM"
    return "HIGH"


def _resolve_emergency_classes(model_names: dict[int, str], keywords: tuple[str, ...]) -> set[int]:
    keyword_set = tuple(keyword.lower() for keyword in keywords)
    emergency_class_ids: set[int] = set()
    for class_id, class_name in model_names.items():
        lower_name = class_name.lower()
        if any(keyword in lower_name for keyword in keyword_set):
            emergency_class_ids.add(int(class_id))
    return emergency_class_ids


def _resolve_lane_boundaries(width: int, lane_count: int, boundaries: list[float] | None) -> list[int]:
    if not boundaries:
        return [int(width * index / lane_count) for index in range(lane_count + 1)]

    normalized = sorted(
        max(0.05, min(0.95, float(boundary)))
        for boundary in boundaries
    )
    if len(normalized) != lane_count - 1:
        return [int(width * index / lane_count) for index in range(lane_count + 1)]

    boundary_pixels = [0]
    boundary_pixels.extend(int(width * boundary) for boundary in normalized)
    boundary_pixels.append(width)
    for index in range(1, len(boundary_pixels)):
        boundary_pixels[index] = max(boundary_pixels[index], boundary_pixels[index - 1] + 1)
    boundary_pixels[-1] = width
    return boundary_pixels


def _normalize_text(value: str, fallback: str = "") -> str:
    cleaned = str(value or "").strip()
    return cleaned if cleaned else fallback


def _safe_day_of_week(capture_date: str) -> str:
    try:
        return datetime.strptime(capture_date, "%Y-%m-%d").strftime("%A").upper()
    except ValueError:
        return "UNKNOWN"


def _read_csv_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return [], []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _merge_fieldnames(existing_fields: list[str], new_fields: list[str]) -> list[str]:
    merged = list(existing_fields)
    for field in new_fields:
        if field not in merged:
            merged.append(field)
    return merged


def _append_to_history(
    csv_path: Path,
    config: TrafficAnalysisConfig,
    lane_boundaries: list[int],
    emergency_supported: bool,
) -> tuple[str, str]:
    history_dir = Path(config.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)

    runs_path = history_dir / "traffic_history.csv"
    training_path = history_dir / "forecast_training_data.csv"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    with csv_path.open("r", newline="", encoding="utf-8") as source_file:
        reader = csv.DictReader(source_file)
        rows = list(reader)

    if not rows:
        return str(runs_path), str(training_path)

    base_fields = [
        "run_id",
        "run_timestamp",
        "run_label",
        "source_video",
        "location_name",
        "intersection_id",
        "capture_date",
        "day_of_week",
        "time_of_day",
        "weather_condition",
        "notes",
        "model_path",
        "lane_count",
        "lane_boundaries",
        "emergency_supported",
    ]
    run_timestamp = datetime.now().isoformat(timespec="seconds")
    capture_date = _normalize_text(config.capture_date, fallback=run_timestamp[:10])
    metadata_fields = {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "run_label": _normalize_text(config.run_label, fallback=Path(config.video_path).stem),
        "source_video": config.video_path,
        "location_name": _normalize_text(config.location_name, fallback="UNKNOWN"),
        "intersection_id": _normalize_text(config.intersection_id, fallback="UNKNOWN"),
        "capture_date": capture_date,
        "day_of_week": _safe_day_of_week(capture_date),
        "time_of_day": _normalize_text(config.time_of_day, fallback="UNKNOWN").upper(),
        "weather_condition": _normalize_text(config.weather_condition, fallback="UNKNOWN").upper(),
        "notes": _normalize_text(config.notes),
        "model_path": config.model_path,
        "lane_count": config.lane_count,
        "lane_boundaries": json.dumps(lane_boundaries),
        "emergency_supported": int(emergency_supported),
    }
    history_fields = base_fields + list(rows[0].keys())
    existing_history_rows, existing_history_fields = _read_csv_rows(runs_path)
    final_history_fields = _merge_fieldnames(existing_history_fields, history_fields)
    history_rows_to_write = existing_history_rows + [{**metadata_fields, **row} for row in rows]
    with runs_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=final_history_fields)
        writer.writeheader()
        writer.writerows(history_rows_to_write)

    training_rows: list[dict[str, str | int | float]] = []
    for index, row in enumerate(rows):
        current_time = int(float(row["Time(sec)"]))
        current_vpm = int(float(row["Vehicles Per Minute"]))
        lane_counts = json.loads(row.get("Lane Vehicle Counts", "[]"))
        lane_congestion = json.loads(row.get("Lane Congestion", "[]"))
        next_vpm = ""
        next_status = ""
        if index + 1 < len(rows):
            next_vpm = int(float(rows[index + 1]["Vehicles Per Minute"]))
            next_status = rows[index + 1]["Traffic Status"]

        feature_row: dict[str, str | int | float] = {
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "run_label": metadata_fields["run_label"],
            "source_video": config.video_path,
            "location_name": metadata_fields["location_name"],
            "intersection_id": metadata_fields["intersection_id"],
            "capture_date": capture_date,
            "day_of_week": metadata_fields["day_of_week"],
            "time_of_day": metadata_fields["time_of_day"],
            "weather_condition": metadata_fields["weather_condition"],
            "time_sec": current_time,
            "vehicles_passed": int(float(row["Vehicles Passed"])),
            "vehicles_per_minute": current_vpm,
            "unique_vehicles": int(float(row["Unique Vehicles"])),
            "traffic_status": row["Traffic Status"],
            "emergency_detected": int(float(row.get("Emergency Detected", 0))),
            "emergency_lane": int(float(row.get("Emergency Lane", 0))),
            "lane_count": config.lane_count,
            "next_vehicles_per_minute": next_vpm,
            "next_traffic_status": next_status,
        }
        for lane_index in range(config.lane_count):
            feature_row[f"lane_{lane_index + 1}_vehicles"] = lane_counts[lane_index] if lane_index < len(lane_counts) else 0
            feature_row[f"lane_{lane_index + 1}_status"] = lane_congestion[lane_index] if lane_index < len(lane_congestion) else "UNKNOWN"
        training_rows.append(feature_row)

    if training_rows:
        training_fields = list(training_rows[0].keys())
        existing_training_rows, existing_training_fields = _read_csv_rows(training_path)
        final_training_fields = _merge_fieldnames(existing_training_fields, training_fields)
        with training_path.open("w", newline="", encoding="utf-8") as training_file:
            writer = csv.DictWriter(training_file, fieldnames=final_training_fields)
            writer.writeheader()
            writer.writerows(existing_training_rows + training_rows)

    return str(runs_path), str(training_path)


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
    emergency_class_ids = _resolve_emergency_classes(model.names, config.emergency_keywords)
    emergency_supported = bool(emergency_class_ids)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    lane_count = max(1, config.lane_count)
    lane_boundaries = _resolve_lane_boundaries(width, lane_count, config.lane_boundaries)
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
    lane_unique_vehicle_ids: list[set[int]] = [set() for _ in range(lane_count)]
    processed_frames = 0
    logged_rows = 0
    next_log_time = 0.0
    last_status = "UNKNOWN"
    last_flow_rate = 0
    last_elapsed_seconds = 0.0
    last_vehicles_passed = 0
    last_logged_second = -1
    last_lane_counts = [0 for _ in range(lane_count)]
    last_lane_statuses = ["LOW" for _ in range(lane_count)]
    last_emergency_detected = False
    last_emergency_lane = -1
    last_emergency_count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "Time(sec)",
                "Vehicles Passed",
                "Vehicles Per Minute",
                "Unique Vehicles",
                "Traffic Status",
                "Lane Vehicle Counts",
                "Lane Congestion",
                "Emergency Detected",
                "Emergency Lane",
                "Emergency Count",
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
            emergency_lane = -1
            emergency_count = 0
            emergency_detected = False

            if emergency_supported and result.boxes is not None:
                for xyxy, cls_tensor in zip(result.boxes.xyxy, result.boxes.cls):
                    class_id = int(cls_tensor.item())
                    if class_id not in emergency_class_ids:
                        continue
                    center_x = int((float(xyxy[0]) + float(xyxy[2])) / 2)
                    emergency_lane = lane_count - 1
                    for candidate_index in range(lane_count):
                        if lane_boundaries[candidate_index] <= center_x < lane_boundaries[candidate_index + 1]:
                            emergency_lane = candidate_index
                            break
                    emergency_count += 1
                emergency_detected = emergency_count > 0
                last_emergency_detected = emergency_detected
                last_emergency_lane = emergency_lane
                last_emergency_count = emergency_count

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

            lane_counts = [0 for _ in range(lane_count)]
            if len(detections) > 0:
                for index, xyxy in enumerate(detections.xyxy):
                    center_x = int((xyxy[0] + xyxy[2]) / 2)
                    lane_index = lane_count - 1
                    for candidate_index in range(lane_count):
                        if lane_boundaries[candidate_index] <= center_x < lane_boundaries[candidate_index + 1]:
                            lane_index = candidate_index
                            break
                    lane_counts[lane_index] += 1
                    if detections.tracker_id is not None and index < len(detections.tracker_id):
                        tracker_id = detections.tracker_id[index]
                        if tracker_id is not None:
                            lane_unique_vehicle_ids[lane_index].add(int(tracker_id))

            lane_statuses = [_lane_status(count) for count in lane_counts]
            last_lane_counts = lane_counts[:]
            last_lane_statuses = lane_statuses[:]

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
                        json.dumps(lane_counts),
                        json.dumps(lane_statuses),
                        int(emergency_detected),
                        emergency_lane + 1 if emergency_lane >= 0 else 0,
                        emergency_count,
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
            emergency_text = "Emergency Priority: Not Supported"
            emergency_color = (60, 60, 60)
            if emergency_supported:
                if emergency_detected and emergency_lane >= 0:
                    emergency_text = f"Emergency Priority: Lane {emergency_lane + 1}"
                    emergency_color = (0, 0, 255)
                else:
                    emergency_text = "Emergency Priority: Clear"
                    emergency_color = (20, 120, 20)
            cv2.putText(
                annotated_frame,
                emergency_text,
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                emergency_color,
                2,
            )
            for lane_index in range(1, lane_count):
                x_position = lane_boundaries[lane_index]
                cv2.line(annotated_frame, (x_position, 0), (x_position, height), (20, 20, 20), 2)

            for lane_index, lane_status in enumerate(lane_statuses):
                lane_start = lane_boundaries[lane_index]
                lane_end = lane_boundaries[lane_index + 1]
                lane_x = lane_start + 12
                text_y = height - 20 - (18 if lane_index % 2 else 0)
                cv2.putText(
                    annotated_frame,
                    f"Lane {lane_index + 1}: {lane_counts[lane_index]} ({lane_status})",
                    (lane_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (15, 15, 15),
                    2,
                )
                lane_center = int((lane_start + lane_end) / 2)
                cv2.putText(
                    annotated_frame,
                    f"L{lane_index + 1}",
                    (lane_center - 18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (20, 20, 20),
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
                        "lane_vehicle_counts": lane_counts,
                        "lane_congestion": lane_statuses,
                        "emergency_detected": emergency_detected,
                        "emergency_lane": emergency_lane + 1 if emergency_lane >= 0 else 0,
                        "emergency_supported": emergency_supported,
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
                    json.dumps(last_lane_counts),
                    json.dumps(last_lane_statuses),
                    int(last_emergency_detected),
                    last_emergency_lane + 1 if last_emergency_lane >= 0 else 0,
                    last_emergency_count,
                ]
            )
            logged_rows += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    duration_seconds = int(frame_number / fps) if frame_number else 0
    history_path, training_data_path = _append_to_history(
        csv_path=csv_path,
        config=config,
        lane_boundaries=lane_boundaries,
        emergency_supported=emergency_supported,
    )
    return {
        "video_path": str(video_path),
        "output_csv": str(csv_path),
        "output_video": str(config.output_video) if config.output_video else "",
        "run_label": config.run_label or video_path.stem,
        "location_name": config.location_name or "UNKNOWN",
        "intersection_id": config.intersection_id or "UNKNOWN",
        "capture_date": config.capture_date or datetime.now().strftime("%Y-%m-%d"),
        "time_of_day": (config.time_of_day or "UNKNOWN").upper(),
        "weather_condition": (config.weather_condition or "UNKNOWN").upper(),
        "processed_frames": processed_frames,
        "total_frames": target_total_frames,
        "duration_seconds": duration_seconds,
        "vehicles_passed": int(line_zone.in_count + line_zone.out_count),
        "unique_vehicles": len(unique_vehicle_ids),
        "vehicles_per_minute": last_flow_rate,
        "traffic_status": last_status,
        "lane_vehicle_counts": last_lane_counts,
        "lane_congestion": last_lane_statuses,
        "lane_unique_vehicles": [len(ids) for ids in lane_unique_vehicle_ids],
        "lane_boundaries": lane_boundaries,
        "emergency_supported": emergency_supported,
        "emergency_detected": last_emergency_detected,
        "emergency_lane": last_emergency_lane + 1 if last_emergency_lane >= 0 else 0,
        "emergency_count": last_emergency_count,
        "history_path": history_path,
        "training_data_path": training_data_path,
        "rows_logged": logged_rows,
    }
