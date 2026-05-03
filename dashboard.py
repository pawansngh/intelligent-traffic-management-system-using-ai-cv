from datetime import datetime
from pathlib import Path
import json

import altair as alt
import cv2
import pandas as pd
from pandas.errors import EmptyDataError
import streamlit as st

from forecast_model import forecast_future_steps, load_forecast_model
from traffic_pipeline import TrafficAnalysisConfig, analyze_video


st.set_page_config(
    page_title="Intelligent Traffic Management System Using AI and CV",
    layout="wide",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');

    :root {
        --bg-top: #f7f4ea;
        --bg-bottom: #ffffff;
        --panel: rgba(255, 255, 255, 0.78);
        --panel-border: rgba(44, 62, 80, 0.10);
        --ink: #18222f;
        --muted: #536170;
        --accent: #ff6b35;
        --accent-soft: rgba(255, 107, 53, 0.14);
        --accent-cool: #1f7a8c;
        --success: #2a9d8f;
        --warning: #e9c46a;
        --danger: #e63946;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 107, 53, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(31, 122, 140, 0.14), transparent 25%),
            linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 58%);
        color: var(--ink);
    }

    html, body, [class*="css"]  {
        font-family: "DM Sans", sans-serif;
    }

    h1, h2, h3 {
        font-family: "Space Grotesk", sans-serif;
        color: var(--ink);
        letter-spacing: -0.02em;
    }

    .hero {
        background: linear-gradient(135deg, rgba(24, 34, 47, 0.95), rgba(31, 122, 140, 0.90));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 28px;
        padding: 2.2rem;
        color: white;
        box-shadow: 0 24px 60px rgba(24, 34, 47, 0.18);
        margin-bottom: 1.25rem;
    }

    .hero h1 {
        color: white;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        margin: 0;
        font-size: 1.05rem;
        color: rgba(255, 255, 255, 0.82);
        max-width: 48rem;
    }

    .pill-row {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }

    .pill {
        border-radius: 999px;
        padding: 0.45rem 0.85rem;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.14);
        font-size: 0.9rem;
    }

    .panel {
        background: var(--panel);
        backdrop-filter: blur(10px);
        border: 1px solid var(--panel-border);
        border-radius: 24px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 18px 40px rgba(44, 62, 80, 0.08);
    }

    .insight-card {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(44, 62, 80, 0.08);
        border-radius: 20px;
        padding: 1rem;
        min-height: 8.5rem;
    }

    .insight-card h3 {
        font-size: 1rem;
        margin-bottom: 0.35rem;
    }

    .insight-card p {
        margin: 0;
        color: var(--muted);
        line-height: 1.45;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.74);
        border: 1px solid rgba(44, 62, 80, 0.08);
        padding: 1rem;
        border-radius: 20px;
        box-shadow: 0 12px 30px rgba(44, 62, 80, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-family: "Space Grotesk", sans-serif;
    }

    [data-testid="stMetricValue"] {
        color: #111111;
        font-family: "Space Grotesk", sans-serif;
        font-weight: 700;
    }

    div[data-testid="stDownloadButton"] button {
        background: #18222f;
        color: #ffffff;
        border: 1px solid rgba(24, 34, 47, 0.15);
        border-radius: 14px;
        font-weight: 700;
    }

    div[data-testid="stDownloadButton"] button:hover {
        background: #111827;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def resolve_default_csv() -> Path:
    preferred_paths = [
        Path("outputs/traffic_data.csv"),
        Path("traffic_data.csv"),
    ]
    for path in preferred_paths:
        if path.exists():
            return path
    return preferred_paths[0]


status_palette = {"LOW": "#2a9d8f", "MEDIUM": "#e9c46a", "HIGH": "#b57609", "UNKNOWN": "#225bbf"}


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    if "Vehicles Per Minute" not in data.columns:
        data["Vehicles Per Minute"] = 0
    if "Unique Vehicles" not in data.columns:
        data["Unique Vehicles"] = data["Vehicles Passed"]
    if "Traffic Status" not in data.columns:
        data["Traffic Status"] = "UNKNOWN"
    if "Lane Vehicle Counts" not in data.columns:
        data["Lane Vehicle Counts"] = ["[]"] * len(data)
    if "Lane Congestion" not in data.columns:
        data["Lane Congestion"] = ["[]"] * len(data)
    if "Emergency Detected" not in data.columns:
        data["Emergency Detected"] = 0
    if "Emergency Lane" not in data.columns:
        data["Emergency Lane"] = 0
    if "Emergency Count" not in data.columns:
        data["Emergency Count"] = 0

    data["Time(sec)"] = pd.to_numeric(data["Time(sec)"], errors="coerce").fillna(0).astype(int)
    data["Vehicles Passed"] = pd.to_numeric(
        data["Vehicles Passed"], errors="coerce"
    ).fillna(0).astype(int)
    data["Vehicles Per Minute"] = pd.to_numeric(
        data["Vehicles Per Minute"], errors="coerce"
    ).fillna(0).astype(int)
    data["Unique Vehicles"] = pd.to_numeric(
        data["Unique Vehicles"], errors="coerce"
    ).fillna(0).astype(int)
    data["Emergency Detected"] = pd.to_numeric(
        data["Emergency Detected"], errors="coerce"
    ).fillna(0).astype(int)
    data["Emergency Lane"] = pd.to_numeric(
        data["Emergency Lane"], errors="coerce"
    ).fillna(0).astype(int)
    data["Emergency Count"] = pd.to_numeric(
        data["Emergency Count"], errors="coerce"
    ).fillna(0).astype(int)
    data["Traffic Status"] = data["Traffic Status"].astype(str).str.upper()
    return data


def parse_json_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def normalize_lane_boundaries(boundaries: list[float], lane_count: int) -> list[float]:
    cleaned = sorted(max(0.05, min(0.95, float(boundary))) for boundary in boundaries)
    unique_boundaries: list[float] = []
    for boundary in cleaned:
        if not unique_boundaries or abs(boundary - unique_boundaries[-1]) >= 0.03:
            unique_boundaries.append(boundary)
    while len(unique_boundaries) < lane_count - 1:
        fallback = len(unique_boundaries) + 1
        unique_boundaries.append(round(fallback / lane_count, 2))
        unique_boundaries = sorted(unique_boundaries)
    return unique_boundaries[: max(0, lane_count - 1)]


def build_signal_plan(lane_df: pd.DataFrame) -> pd.DataFrame:
    if lane_df.empty:
        return pd.DataFrame(columns=["Lane", "Vehicles", "Congestion", "Recommended Green (sec)", "Priority"])

    minimum_green = 15
    total_cycle = 120
    priority_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}
    vehicle_total = max(int(lane_df["Vehicles"].sum()), 1)
    available_green = total_cycle - (minimum_green * len(lane_df))

    plan_rows = []
    for _, row in lane_df.iterrows():
        vehicle_share = int(row["Vehicles"]) / vehicle_total
        recommended_green = minimum_green + round(available_green * vehicle_share)
        plan_rows.append(
            {
                "Lane": row["Lane"],
                "Vehicles": int(row["Vehicles"]),
                "Congestion": row["Congestion"],
                "Recommended Green (sec)": recommended_green,
                "Priority": priority_order.get(str(row["Congestion"]).upper(), 0),
            }
        )

    plan_df = pd.DataFrame(plan_rows).sort_values(
        by=["Priority", "Vehicles"],
        ascending=[False, False],
    )
    return plan_df.drop(columns=["Priority"])


def heuristic_forecast_traffic(data: pd.DataFrame, steps: int = 3) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["Time(sec)", "Vehicles Per Minute", "Type", "Traffic Status"])

    history = data[["Time(sec)", "Vehicles Per Minute", "Traffic Status"]].copy()
    history["Type"] = "Observed"

    if len(data) <= 1:
        base_interval = 5
    else:
        positive_intervals = data["Time(sec)"].diff().dropna()
        positive_intervals = positive_intervals[positive_intervals > 0]
        base_interval = int(positive_intervals.median()) if not positive_intervals.empty else 5
        base_interval = max(base_interval, 1)

    recent_values = data["Vehicles Per Minute"].tail(min(4, len(data))).tolist()
    last_time = int(data["Time(sec)"].iloc[-1])
    current_estimate = float(data["Vehicles Per Minute"].iloc[-1])
    smoothed_average = sum(recent_values) / len(recent_values)

    if len(recent_values) >= 2:
        trend_changes = [
            recent_values[index] - recent_values[index - 1]
            for index in range(1, len(recent_values))
        ]
        trend = sum(trend_changes) / len(trend_changes)
    else:
        trend = 0.0

    forecast_rows = []
    for step in range(1, steps + 1):
        current_estimate = max(
            0.0,
            (0.65 * current_estimate) + (0.35 * smoothed_average) + trend,
        )
        if current_estimate >= 20:
            predicted_status = "HIGH"
        elif current_estimate >= 10:
            predicted_status = "MEDIUM"
        else:
            predicted_status = "LOW"

        forecast_rows.append(
            {
                "Time(sec)": last_time + (step * base_interval),
                "Vehicles Per Minute": int(round(current_estimate)),
                "Traffic Status": predicted_status,
                "Type": "Forecast",
            }
        )

    forecast_df = pd.DataFrame(forecast_rows)
    return pd.concat([history, forecast_df], ignore_index=True)


def forecast_traffic(
    data: pd.DataFrame,
    summary: dict[str, object] | None = None,
    model_path: str = "models/traffic_forecast_model.json",
    steps: int = 3,
) -> tuple[pd.DataFrame, str, object | None]:
    bundle = load_forecast_model(model_path)
    if bundle is not None:
        capture_day = "UNKNOWN"
        if summary and summary.get("capture_date"):
            parsed_date = pd.to_datetime(summary.get("capture_date"), errors="coerce")
            if not pd.isna(parsed_date):
                capture_day = str(parsed_date.day_name()).upper()
        metadata = {
            "time_of_day": summary.get("time_of_day", "UNKNOWN") if summary else "UNKNOWN",
            "weather_condition": summary.get("weather_condition", "UNKNOWN") if summary else "UNKNOWN",
            "day_of_week": capture_day,
        }
        forecast_df, source_label = forecast_future_steps(data, metadata=metadata, bundle=bundle, steps=steps)
        return forecast_df, source_label, bundle

    return heuristic_forecast_traffic(data, steps=steps), "Heuristic fallback", None


def load_historical_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    history_path = Path("history/traffic_history.csv")
    training_path = Path("history/forecast_training_data.csv")

    if history_path.exists() and history_path.stat().st_size > 0:
        history_df = pd.read_csv(history_path)
    else:
        history_df = pd.DataFrame()

    if training_path.exists() and training_path.stat().st_size > 0:
        training_df = pd.read_csv(training_path)
    else:
        training_df = pd.DataFrame()

    for frame in (history_df, training_df):
        if frame.empty:
            continue
        for column in ("run_label", "location_name", "intersection_id", "capture_date", "day_of_week", "time_of_day", "weather_condition"):
            if column not in frame.columns:
                frame[column] = "UNKNOWN"

    return history_df, training_df


def default_video_path() -> str:
    preferred_paths = [Path("videos/traffic.mp4")]
    for path in preferred_paths:
        if path.exists():
            return str(path)
    return ""


def latest_outputs() -> tuple[Path | None, Path | None]:
    output_dir = Path("outputs")
    if not output_dir.exists():
        return None, None

    csv_files = sorted(
        [path for path in output_dir.glob("*_data.csv") if path.stat().st_size > 0],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    csv_fallbacks = sorted(
        [path for path in output_dir.glob("*.csv") if path.stat().st_size > 0],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    video_files = sorted(
        [path for path in output_dir.glob("*.mp4") if path.stat().st_size > 1024],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    csv_path = csv_files[0] if csv_files else (csv_fallbacks[0] if csv_fallbacks else None)
    video_path = video_files[0] if video_files else None
    return csv_path, video_path


def load_active_dataset() -> tuple[pd.DataFrame | None, Path | None, Path | None]:
    active_csv = st.session_state.get("active_csv_path")
    active_video = st.session_state.get("active_video_path")

    csv_path = Path(active_csv) if active_csv else None
    video_path = Path(active_video) if active_video else None

    if csv_path is None or not csv_path.exists():
        latest_csv, latest_video = latest_outputs()
        csv_path = latest_csv or resolve_default_csv()
        if video_path is None:
            video_path = latest_video

    if csv_path is None or not csv_path.exists():
        return None, None, None

    try:
        data = pd.read_csv(csv_path)
    except EmptyDataError:
        return None, None, video_path
    if data.empty:
        return None, csv_path, video_path

    return normalize_data(data), csv_path, video_path


def save_uploaded_video(uploaded_file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
    destination = uploads_dir / f"{timestamp}_{safe_name}"
    with destination.open("wb") as output_file:
        output_file.write(uploaded_file.getbuffer())
    return destination


def sample_video_frames(video_path: Path, max_frames: int = 3) -> list[tuple[str, object]]:
    frames: list[tuple[str, object]] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        positions = [0]
    else:
        positions = sorted({int(total_frames * ratio) for ratio in (0.2, 0.5, 0.8)})

    for index, position in enumerate(positions[:max_frames], start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((f"Preview {index}", frame_rgb))

    cap.release()
    return frames


st.session_state.setdefault("analysis_summary", None)

st.sidebar.markdown("### Full-Stack Controls")
uploaded_video = st.sidebar.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])
use_sample_video = st.sidebar.checkbox("Use bundled sample video", value=uploaded_video is None)
st.sidebar.markdown("### Collection Metadata")
run_label = st.sidebar.text_input("Run label", value="")
location_name = st.sidebar.text_input("Location name", value="")
intersection_id = st.sidebar.text_input("Intersection ID", value="")
capture_date = st.sidebar.date_input("Capture date", value=datetime.now()).strftime("%Y-%m-%d")
time_of_day = st.sidebar.selectbox(
    "Time of day",
    options=["MORNING", "AFTERNOON", "EVENING", "NIGHT", "PEAK", "OFF_PEAK", "UNKNOWN"],
    index=0,
)
weather_condition = st.sidebar.selectbox(
    "Weather",
    options=["CLEAR", "CLOUDY", "RAIN", "FOG", "UNKNOWN"],
    index=0,
)
notes = st.sidebar.text_area("Collection notes", value="", height=80)
confidence = st.sidebar.slider("Detection confidence", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
frame_skip = st.sidebar.slider("Frame skip", min_value=1, max_value=5, value=1, step=1)
lane_count = st.sidebar.slider("Lane count", min_value=1, max_value=6, value=3, step=1)
default_lane_boundaries = [round(index / lane_count, 2) for index in range(1, lane_count)]
lane_boundary_inputs: list[float] = []
for boundary_index in range(max(0, lane_count - 1)):
    lane_boundary_inputs.append(
        st.sidebar.slider(
            f"Lane split {boundary_index + 1}",
            min_value=0.05,
            max_value=0.95,
            value=float(default_lane_boundaries[boundary_index]),
            step=0.01,
        )
    )
lane_boundaries = normalize_lane_boundaries(lane_boundary_inputs, lane_count)
if lane_boundaries != lane_boundary_inputs:
    st.sidebar.caption(f"Adjusted lane splits to keep them ordered: {lane_boundaries}")
line_y_ratio = st.sidebar.slider("Count line position", min_value=0.20, max_value=0.85, value=0.55, step=0.05)
log_interval = st.sidebar.slider("Log interval (sec)", min_value=1, max_value=10, value=5, step=1)
max_frames = st.sidebar.number_input("Max frames for quick test", min_value=0, value=0, step=30)

st.sidebar.markdown("### Project Snapshot")
st.sidebar.write("Frontend: Streamlit web app")
st.sidebar.write("Backend: YOLOv8 + ByteTrack pipeline")
st.sidebar.write("Storage: uploaded videos, CSV metrics, annotated MP4, historical forecast dataset")
st.sidebar.write("Experience: upload -> analyze -> review")

run_analysis = st.sidebar.button("Run Analysis", type="primary", width="stretch")

if run_analysis:
    source_video_path: Path | None = None
    if uploaded_video is not None:
        source_video_path = save_uploaded_video(uploaded_video)
    elif use_sample_video and default_video_path():
        source_video_path = Path(default_video_path())

    if source_video_path is None:
        st.sidebar.error("Upload a video or enable the bundled sample video first.")
    else:
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_output = output_dir / f"{run_id}_data.csv"
        video_output = output_dir / f"{run_id}_annotated.mp4"

        progress_bar = st.sidebar.progress(0, text="Preparing traffic analysis...")
        status_box = st.sidebar.empty()

        def on_progress(update: dict[str, int | float | str]) -> None:
            total_frames = int(update.get("total_frames", 0) or 0)
            processed_frames = int(update.get("processed_frames", 0) or 0)
            ratio = min(processed_frames / total_frames, 1.0) if total_frames else 0.0
            progress_bar.progress(
                ratio,
                text=f"Processing frame {processed_frames}/{total_frames or '?'}",
            )
            status_box.caption(
                " | ".join(
                    [
                        f"Passed: {update.get('vehicles_passed', 0)}",
                        f"Unique: {update.get('unique_vehicles', 0)}",
                        f"Flow: {update.get('vehicles_per_minute', 0)}/min",
                        f"Status: {update.get('traffic_status', 'UNKNOWN')}",
                    ]
                )
            )

        try:
            summary = analyze_video(
                TrafficAnalysisConfig(
                    video_path=str(source_video_path),
                    model_path="yolov8n.pt",
                    output_csv=str(csv_output),
                    output_video=str(video_output),
                    confidence=confidence,
                    frame_skip=frame_skip,
                    lane_count=lane_count,
                    lane_boundaries=lane_boundaries,
                    line_y_ratio=line_y_ratio,
                    log_interval_sec=float(log_interval),
                    max_frames=int(max_frames) if int(max_frames) > 0 else None,
                    run_label=run_label,
                    location_name=location_name,
                    intersection_id=intersection_id,
                    capture_date=capture_date,
                    time_of_day=time_of_day,
                    weather_condition=weather_condition,
                    notes=notes,
                ),
                progress_callback=on_progress,
            )
            progress_bar.progress(1.0, text="Analysis complete")
            st.session_state["analysis_summary"] = summary
            st.session_state["active_csv_path"] = str(csv_output)
            st.session_state["active_video_path"] = str(video_output)
            st.sidebar.success("Traffic analysis finished. The dashboard below is now showing this run.")
        except Exception as exc:
            progress_bar.empty()
            status_box.empty()
            st.sidebar.error(f"Analysis failed: {exc}")


data, csv_path, output_video_path = load_active_dataset()

st.sidebar.markdown("### Active Assets")
st.sidebar.caption(f"CSV: `{csv_path}`" if csv_path else "CSV: none yet")
st.sidebar.caption(f"Video: `{output_video_path}`" if output_video_path else "Video: none yet")

st.markdown(
    f"""
    <section class="hero">
        <h1>Intelligent Traffic Management System Using AI and CV</h1>
        <p>
            A full-stack traffic intelligence platform powered by artificial intelligence and computer vision.
            The long-term goal is to connect live CCTV feeds at intersections, measure congestion in real time, support adaptive signal timing, forecast traffic patterns, and prioritize emergency vehicles.
        </p>
        <div class="pill-row">
            <span class="pill">YOLOv8 Detection</span>
            <span class="pill">ByteTrack Tracking</span>
            <span class="pill">Adaptive Signal Vision</span>
            <span class="pill">Emergency Priority Ready</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

if data is None:
    st.warning("No traffic analysis data is available yet. Upload a video and click `Run Analysis`, or use the bundled sample video.")
    st.stop()

latest = data.iloc[-1]
peak_row = data.loc[data["Vehicles Per Minute"].idxmax()]
duration_seconds = int(data["Time(sec)"].max())
average_flow = int(data["Vehicles Per Minute"].mean())
summary = st.session_state.get("analysis_summary")
latest_lane_counts = parse_json_list(latest["Lane Vehicle Counts"])
latest_lane_statuses = parse_json_list(latest["Lane Congestion"])
lane_rows = []
for lane_index, vehicle_count in enumerate(latest_lane_counts, start=1):
    lane_rows.append(
        {
            "Lane": f"Lane {lane_index}",
            "Vehicles": int(vehicle_count),
            "Congestion": latest_lane_statuses[lane_index - 1] if lane_index - 1 < len(latest_lane_statuses) else "UNKNOWN",
        }
    )
lane_df = pd.DataFrame(lane_rows)
signal_plan_df = build_signal_plan(lane_df)
forecast_df, forecast_source, forecast_model_bundle = forecast_traffic(data, summary=summary, steps=3)
emergency_supported = bool(summary["emergency_supported"]) if summary and "emergency_supported" in summary else False
emergency_detected = bool(latest["Emergency Detected"])
emergency_lane = int(latest["Emergency Lane"])
emergency_count = int(latest["Emergency Count"])
if summary:
    st.info(
        "Latest run: "
        f"{summary['vehicles_passed']} vehicles passed, "
        f"{summary['unique_vehicles']} unique vehicles, "
        f"ending at {summary['traffic_status']} traffic."
    )
    st.caption(
        "Run context: "
        f"`{summary.get('run_label', 'UNKNOWN')}` | "
        f"`{summary.get('location_name', 'UNKNOWN')}` | "
        f"`{summary.get('intersection_id', 'UNKNOWN')}` | "
        f"`{summary.get('capture_date', 'UNKNOWN')}` | "
        f"`{summary.get('time_of_day', 'UNKNOWN')}` | "
        f"`{summary.get('weather_condition', 'UNKNOWN')}`"
    )
    if "history_path" in summary and "training_data_path" in summary:
        st.caption(
            f'Historical logs updated at `{summary["history_path"]}` and training-ready data updated at `{summary["training_data_path"]}`.'
        )

history_df, training_df = load_historical_data()

st.subheader("Emergency Priority Status")
if emergency_supported:
    if emergency_detected and emergency_lane > 0:
        st.error(
            f"Emergency vehicle priority active. Lane {emergency_lane} should receive immediate clearance. "
            f"Detected emergency count: {emergency_count}."
        )
    else:
        st.success("No emergency vehicle detected in the latest analyzed frame window.")
else:
    st.warning(
        "The current YOLOv8n model does not contain ambulance, police, or fire-truck classes. "
        "Load a fine-tuned emergency-aware model to activate automatic priority detection."
    )

st.subheader("Mission Scope")
scope_columns = st.columns(3)
scope_columns[0].markdown(
    """
    <div class="panel">
        <h3>Live CCTV Monitoring</h3>
        <p>The target deployment is a web app connected to CCTV feeds at intersections so traffic can be monitored continuously instead of from offline video only.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
scope_columns[1].markdown(
    """
    <div class="panel">
        <h3>Adaptive Signal Control</h3>
        <p>Instead of fixed timers, the system is intended to recommend or drive signal duration using observed congestion and lane-level traffic pressure.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
scope_columns[2].markdown(
    """
    <div class="panel">
        <h3>Prediction And Priority</h3>
        <p>The future roadmap includes traffic forecasting and emergency vehicle detection so the junction can react before congestion worsens.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(4)
metric_columns[0].metric("Vehicles Passed", int(latest["Vehicles Passed"]))
metric_columns[1].metric("Vehicles / Minute", int(latest["Vehicles Per Minute"]))
metric_columns[2].metric("Unique Vehicles", int(latest["Unique Vehicles"]))
metric_columns[3].metric("Traffic Status", str(latest["Traffic Status"]))

if not lane_df.empty:
    st.subheader("Lane-Wise Congestion")
    lane_metric_columns = st.columns(len(lane_df))
    for index, row in lane_df.iterrows():
        lane_metric_columns[index].metric(row["Lane"], int(row["Vehicles"]), row["Congestion"])
    st.caption(f"Active lane split ratios: {lane_boundaries}")

if not signal_plan_df.empty:
    st.subheader("Adaptive Signal Recommendation")
    recommendation_columns = st.columns(len(signal_plan_df))
    for index, (_, row) in enumerate(signal_plan_df.iterrows()):
        metric_value = f'{int(row["Recommended Green (sec)"])} sec'
        if emergency_supported and emergency_detected and row["Lane"] == f"Lane {emergency_lane}":
            metric_value = "PRIORITY"
        recommendation_columns[index].metric(
            row["Lane"],
            metric_value,
            row["Congestion"],
        )
    top_lane = signal_plan_df.iloc[0]
    if emergency_supported and emergency_detected and emergency_lane > 0:
        st.caption(
            f"Emergency override active: Lane {emergency_lane} should be cleared first before normal adaptive timing resumes."
        )
    else:
        st.caption(
            f'Priority lane right now: {top_lane["Lane"]} with {int(top_lane["Vehicles"])} vehicles. '
            f'Recommended green time: {int(top_lane["Recommended Green (sec)"])} seconds.'
        )

if not forecast_df.empty:
    forecast_rows = forecast_df[forecast_df["Type"] == "Forecast"].reset_index(drop=True)
    if not forecast_rows.empty:
        st.subheader("Traffic Forecast")
        forecast_columns = st.columns(len(forecast_rows))
        for index, row in forecast_rows.iterrows():
            forecast_columns[index].metric(
                f'T+{int(row["Time(sec)"] - int(latest["Time(sec)"]))} sec',
                f'{int(row["Vehicles Per Minute"])} / min',
                row["Traffic Status"],
            )
        if forecast_model_bundle is not None:
            st.caption(
                f"Forecast source: {forecast_source}. "
                f"Model trained on {forecast_model_bundle.sample_count} supervised samples with "
                f"training RMSE {forecast_model_bundle.metrics.get('rmse', 0.0):.2f}."
            )
        else:
            st.caption(
                "Forecast source: Heuristic fallback. Train `models/traffic_forecast_model.json` to replace this with the learned baseline model."
            )

if not history_df.empty or not training_df.empty:
    st.subheader("Historical Data Readiness")
    history_columns = st.columns(3)
    history_columns[0].metric("Stored History Rows", int(len(history_df)) if not history_df.empty else 0)
    history_columns[1].metric("Training Samples", int(len(training_df)) if not training_df.empty else 0)
    history_columns[2].metric("Recorded Runs", int(history_df["run_id"].nunique()) if not history_df.empty and "run_id" in history_df.columns else 0)
    st.caption(
        "Each analysis run is appended to persistent history files so future forecasting models can be trained on accumulated traffic behavior."
    )
    if not history_df.empty:
        coverage_columns = st.columns(3)
        coverage_columns[0].metric("Unique Videos", int(history_df["source_video"].nunique()))
        coverage_columns[1].metric("Time-of-Day Buckets", int(history_df["time_of_day"].nunique()))
        coverage_columns[2].metric("Intersections", int(history_df["intersection_id"].replace("", "UNKNOWN").nunique()))

        run_history = (
            history_df.drop_duplicates(subset=["run_id"])
            .copy()
        )
        time_of_day_order = ["MORNING", "AFTERNOON", "EVENING", "NIGHT", "PEAK", "OFF_PEAK", "UNKNOWN"]
        run_history["time_of_day"] = pd.Categorical(run_history["time_of_day"], categories=time_of_day_order, ordered=True)
        time_of_day_counts = (
            run_history.groupby("time_of_day", observed=False)
            .size()
            .reset_index(name="Runs")
        )
        source_counts = (
            run_history.groupby("source_video")
            .size()
            .reset_index(name="Runs")
            .sort_values("Runs", ascending=False)
            .head(8)
        )

        coverage_left, coverage_right = st.columns(2)
        coverage_left.altair_chart(
            alt.Chart(time_of_day_counts)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#1f7a8c")
            .encode(
                x=alt.X("time_of_day:N", title="Time of Day", sort=time_of_day_order),
                y=alt.Y("Runs:Q", title="Recorded Runs"),
                tooltip=["time_of_day", "Runs"],
            )
            .properties(height=260, title="Dataset Coverage by Time of Day"),
            width="stretch",
        )
        coverage_right.altair_chart(
            alt.Chart(source_counts)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#ff6b35")
            .encode(
                x=alt.X("Runs:Q", title="Recorded Runs"),
                y=alt.Y("source_video:N", title="Video Source", sort="-x"),
                tooltip=["source_video", "Runs"],
            )
            .properties(height=260, title="Most-Used Traffic Videos"),
            width="stretch",
        )

    if forecast_model_bundle is not None:
        st.subheader("Forecast Model Status")
        model_columns = st.columns(4)
        model_columns[0].metric("Model Samples", forecast_model_bundle.sample_count)
        model_columns[1].metric("Training MAE", f"{forecast_model_bundle.metrics.get('mae', 0.0):.2f}")
        model_columns[2].metric("Training RMSE", f"{forecast_model_bundle.metrics.get('rmse', 0.0):.2f}")
        model_columns[3].metric("Training R2", f"{forecast_model_bundle.metrics.get('r2', 0.0):.2f}")
        st.caption(
            f"Loaded trained model from `{forecast_model_bundle.model_path}` created at `{forecast_model_bundle.trained_at}`."
        )

insight_columns = st.columns(3)
insight_columns[0].markdown(
    f"""
    <div class="insight-card">
        <h3>Peak Traffic Load</h3>
        <p>The busiest sampled moment reached <strong>{int(peak_row["Vehicles Per Minute"])}</strong> vehicles per minute at second <strong>{int(peak_row["Time(sec)"])}</strong>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
insight_columns[1].markdown(
    f"""
    <div class="insight-card">
        <h3>Monitoring Window</h3>
        <p>The current analysis spans <strong>{duration_seconds}</strong> seconds and tracks <strong>{int(latest["Unique Vehicles"])}</strong> unique vehicles across the scene.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
insight_columns[2].markdown(
    f"""
    <div class="insight-card">
        <h3>Traffic Condition Signal</h3>
        <p>The average observed flow is <strong>{average_flow}</strong> vehicles per minute, ending in a <strong>{latest["Traffic Status"]}</strong> traffic state.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

chart_columns = st.columns([1.4, 1])
with chart_columns[0]:
    st.subheader("Traffic Flow Timeline")
    passed_flow_data = data.melt(
        id_vars=["Time(sec)"],
        value_vars=["Vehicles Passed"],
        var_name="Metric",
        value_name="Value",
    )
    passed_chart = (
        alt.Chart(passed_flow_data)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Time(sec):Q", title="Time (seconds)"),
            y=alt.Y("Value:Q", title="Vehicles"),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(
                    domain=["Vehicles Passed"],
                    range=["#ff6b35"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["Time(sec)", "Metric", "Value"],
        )
        .properties(height=360)
    )
    traffic_forecast_chart = (
        alt.Chart(forecast_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Time(sec):Q", title="Time (seconds)"),
            y=alt.Y("Vehicles Per Minute:Q", title="Vehicles / Minute"),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["Observed", "Forecast"],
                    range=["#1f7a8c", "#e76f51"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            strokeDash=alt.StrokeDash(
                "Type:N",
                scale=alt.Scale(
                    domain=["Observed", "Forecast"],
                    range=[[1, 0], [6, 4]],
                ),
            ),
            tooltip=["Time(sec)", "Vehicles Per Minute", "Type", "Traffic Status"],
        )
        .properties(height=360)
    )
    st.altair_chart((passed_chart + traffic_forecast_chart).resolve_scale(y="independent"), width="stretch")

with chart_columns[1]:
    st.subheader("Traffic Density Mix")
    if not lane_df.empty:
        lane_chart = (
            alt.Chart(lane_df)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("Lane:N", title=None),
                y=alt.Y("Vehicles:Q", title="Vehicles"),
                color=alt.Color(
                    "Congestion:N",
                    scale=alt.Scale(
                        domain=list(status_palette.keys()),
                        range=list(status_palette.values()),
                    ),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=["Lane", "Vehicles", "Congestion"],
            )
            .properties(height=360)
        )
        st.altair_chart(lane_chart, width="stretch")
    else:
        status_counts = (
            data["Traffic Status"]
            .value_counts()
            .rename_axis("Traffic Status")
            .reset_index(name="Count")
        )
        status_chart = (
            alt.Chart(status_counts)
            .mark_arc(innerRadius=55, outerRadius=120)
            .encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color(
                    "Traffic Status:N",
                    scale=alt.Scale(
                        domain=list(status_palette.keys()),
                        range=list(status_palette.values()),
                    ),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=["Traffic Status", "Count"],
            )
            .properties(height=360)
        )
        st.altair_chart(status_chart, width="stretch")

media_columns = st.columns([1.15, 1])
with media_columns[0]:
    st.subheader("Recent Log Entries")
    st.dataframe(data.tail(15), width="stretch")

with media_columns[1]:
    st.subheader("Annotated Footage")
    if output_video_path is not None:
        video_bytes = output_video_path.read_bytes()
        st.video(video_bytes)
        st.caption(f"Loaded annotated output from `{output_video_path}`.")
        st.download_button(
            "Download annotated video",
            data=video_bytes,
            file_name=output_video_path.name,
            mime="video/mp4",
            width="stretch",
        )
        st.caption(
            "If the embedded player does not render in your browser, use the download button or review the sampled frames below."
        )

        preview_frames = sample_video_frames(output_video_path)
        if preview_frames:
            preview_columns = st.columns(len(preview_frames))
            for column, (label, frame) in zip(preview_columns, preview_frames):
                column.image(frame, caption=label, width="stretch")
    else:
        st.info("Run an analysis from the sidebar to generate an annotated MP4 preview.")

if not signal_plan_df.empty:
    st.subheader("Signal Plan Table")
    st.dataframe(signal_plan_df, width="stretch")

if not forecast_df.empty:
    st.subheader("Forecast Table")
    st.dataframe(
        forecast_df[forecast_df["Type"] == "Forecast"][["Time(sec)", "Vehicles Per Minute", "Traffic Status"]],
        width="stretch",
    )

if not training_df.empty:
    st.subheader("Training Dataset Preview")
    training_preview_columns = [
        column
        for column in [
            "run_id",
            "time_sec",
            "vehicles_per_minute",
            "traffic_status",
            "next_vehicles_per_minute",
            "next_traffic_status",
        ]
        if column in training_df.columns
    ]
    st.dataframe(training_df[training_preview_columns].tail(15), width="stretch")

st.subheader("System Architecture")
architecture_columns = st.columns(3)
architecture_columns[0].markdown(
    """
    <div class="panel">
        <h3>1. Frontend</h3>
        <p>Streamlit handles video upload, tuning controls, progress feedback, and presentation-ready results in the browser.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
architecture_columns[1].markdown(
    """
    <div class="panel">
        <h3>2. Backend</h3>
        <p>YOLOv8 and ByteTrack power the current backend pipeline for detection, tracking, counting, and traffic status estimation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
architecture_columns[2].markdown(
    """
    <div class="panel">
        <h3>3. Data Layer</h3>
        <p>Each run produces reusable CSV and MP4 artifacts today, and this layer can later expand into CCTV ingestion, forecasting datasets, and signal-control outputs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
