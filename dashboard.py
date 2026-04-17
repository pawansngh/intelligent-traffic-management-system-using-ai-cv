from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from traffic_pipeline import TrafficAnalysisConfig, analyze_video


st.set_page_config(page_title="Traffic Monitoring Dashboard", layout="wide")
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


status_palette = {"LOW": "#2a9d8f", "MEDIUM": "#e9c46a", "HIGH": "#e63946", "UNKNOWN": "#8d99ae"}


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    if "Vehicles Per Minute" not in data.columns:
        data["Vehicles Per Minute"] = 0
    if "Unique Vehicles" not in data.columns:
        data["Unique Vehicles"] = data["Vehicles Passed"]
    if "Traffic Status" not in data.columns:
        data["Traffic Status"] = "UNKNOWN"

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
    data["Traffic Status"] = data["Traffic Status"].astype(str).str.upper()
    return data


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

    csv_files = sorted(output_dir.glob("*_data.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    csv_fallbacks = sorted(output_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    video_files = sorted(output_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
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

    data = pd.read_csv(csv_path)
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


st.session_state.setdefault("analysis_summary", None)

st.sidebar.markdown("### Full-Stack Controls")
uploaded_video = st.sidebar.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])
use_sample_video = st.sidebar.checkbox("Use bundled sample video", value=uploaded_video is None)
confidence = st.sidebar.slider("Detection confidence", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
frame_skip = st.sidebar.slider("Frame skip", min_value=1, max_value=5, value=1, step=1)
line_y_ratio = st.sidebar.slider("Count line position", min_value=0.20, max_value=0.85, value=0.55, step=0.05)
log_interval = st.sidebar.slider("Log interval (sec)", min_value=1, max_value=10, value=5, step=1)
max_frames = st.sidebar.number_input("Max frames for quick test", min_value=0, value=0, step=30)

st.sidebar.markdown("### Project Snapshot")
st.sidebar.write("Frontend: Streamlit web app")
st.sidebar.write("Backend: YOLOv8 + ByteTrack pipeline")
st.sidebar.write("Storage: uploaded videos, CSV metrics, annotated MP4")
st.sidebar.write("Experience: upload -> analyze -> review")

run_analysis = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

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
                    line_y_ratio=line_y_ratio,
                    log_interval_sec=float(log_interval),
                    max_frames=int(max_frames) if int(max_frames) > 0 else None,
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
        <h1>Traffic Intelligence Studio</h1>
        <p>
            A full-stack style computer-vision app for traffic detection, tracking, and analytics.
            Upload a video, run the backend pipeline, and review stakeholder-friendly results in one browser workflow.
        </p>
        <div class="pill-row">
            <span class="pill">YOLOv8 Detection</span>
            <span class="pill">ByteTrack Tracking</span>
            <span class="pill">Upload And Analyze</span>
            <span class="pill">Streamlit Frontend + Python Backend</span>
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
if summary:
    st.info(
        "Latest run: "
        f"{summary['vehicles_passed']} vehicles passed, "
        f"{summary['unique_vehicles']} unique vehicles, "
        f"ending at {summary['traffic_status']} traffic."
    )

metric_columns = st.columns(4)
metric_columns[0].metric("Vehicles Passed", int(latest["Vehicles Passed"]))
metric_columns[1].metric("Vehicles / Minute", int(latest["Vehicles Per Minute"]))
metric_columns[2].metric("Unique Vehicles", int(latest["Unique Vehicles"]))
metric_columns[3].metric("Traffic Status", str(latest["Traffic Status"]))

insight_columns = st.columns(3)
insight_columns[0].markdown(
    f"""
    <div class="insight-card">
        <h3>Peak Throughput</h3>
        <p>The busiest sampled moment reached <strong>{int(peak_row["Vehicles Per Minute"])}</strong> vehicles per minute at second <strong>{int(peak_row["Time(sec)"])}</strong>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
insight_columns[1].markdown(
    f"""
    <div class="insight-card">
        <h3>Coverage Window</h3>
        <p>The current analysis spans <strong>{duration_seconds}</strong> seconds and tracks <strong>{int(latest["Unique Vehicles"])}</strong> unique vehicles across the scene.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
insight_columns[2].markdown(
    f"""
    <div class="insight-card">
        <h3>Operational Signal</h3>
        <p>The average observed flow is <strong>{average_flow}</strong> vehicles per minute, ending in a <strong>{latest["Traffic Status"]}</strong> traffic state.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

chart_columns = st.columns([1.4, 1])
with chart_columns[0]:
    st.subheader("Traffic Flow Timeline")
    flow_data = data.melt(
        id_vars=["Time(sec)"],
        value_vars=["Vehicles Passed", "Vehicles Per Minute"],
        var_name="Metric",
        value_name="Value",
    )
    flow_chart = (
        alt.Chart(flow_data)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Time(sec):Q", title="Time (seconds)"),
            y=alt.Y("Value:Q", title="Vehicles"),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(
                    domain=["Vehicles Passed", "Vehicles Per Minute"],
                    range=["#ff6b35", "#1f7a8c"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["Time(sec)", "Metric", "Value"],
        )
        .properties(height=360)
    )
    st.altair_chart(flow_chart, width="stretch")

with chart_columns[1]:
    st.subheader("Traffic State Mix")
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
        st.video(str(output_video_path))
        st.caption(f"Loaded annotated output from `{output_video_path}`.")
    else:
        st.info("Run an analysis from the sidebar to generate an annotated MP4 preview.")

st.subheader("System Design")
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
        <p>YOLOv8 and ByteTrack power the backend pipeline for detection, tracking, counting, and traffic status estimation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
architecture_columns[2].markdown(
    """
    <div class="panel">
        <h3>3. Data Layer</h3>
        <p>Each run produces reusable CSV and MP4 artifacts so metrics, visuals, and demos all come from the same execution.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
