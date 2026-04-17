# Traffic Intelligence Studio

Traffic Intelligence Studio is a portfolio-ready full-stack style computer vision app that analyzes roadway footage with YOLOv8, tracks vehicles with ByteTrack, estimates flow intensity, and presents the results in a polished Streamlit interface.

## Highlights

- Vehicle detection for cars, buses, trucks, and motorcycles
- Multi-object tracking with persistent IDs
- Line-cross counting for traffic throughput estimation
- CSV logging for downstream analytics
- Annotated video export for demos and presentations
- Streamlit app with upload, analysis, progress, metrics, insights, charts, and video preview

## Project Structure

- `main.py`: command-line entry point
- `traffic_pipeline.py`: reusable analysis pipeline
- `dashboard.py`: frontend web app with integrated analysis
- `plot_graph.py`: quick matplotlib inspection
- `videos/traffic.mp4`: sample source footage
- `yolov8n.pt`: YOLO model weights
- `outputs/`: generated CSV and annotated video
- `uploads/`: user-uploaded source videos created by the web app

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Analyzer

The default run now saves portfolio-friendly artifacts into `outputs/`.

```powershell
.\.venv\Scripts\python.exe main.py
```

Optional examples:

```powershell
.\.venv\Scripts\python.exe main.py --display
.\.venv\Scripts\python.exe main.py --max-frames 180
.\.venv\Scripts\python.exe main.py --video videos/traffic.mp4 --output-csv outputs/custom_run.csv --output-video outputs/custom_run.mp4
```

## Launch the Web App

```powershell
.\.venv\Scripts\streamlit.exe run dashboard.py
```

Inside the app, you can:

1. Upload a video or use the bundled sample video
2. Adjust detection settings from the sidebar
3. Run analysis directly in the browser
4. Review metrics, charts, and annotated footage in one place

The app automatically looks for:

1. `outputs/traffic_data.csv`
2. `traffic_data.csv`

## Portfolio Talking Points

- Demonstrates an end-to-end AI workflow: upload, detection, tracking, analytics, and visualization
- Produces both machine-readable metrics and stakeholder-friendly outputs
- Designed to be extensible for congestion alerts, signal optimization, and smart-city monitoring

## Future Extensions

- Real-time webcam or RTSP stream support
- Lane-wise analytics and direction-aware counting
- Alerting for congestion spikes
- Historical trend storage in a database
