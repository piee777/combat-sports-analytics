# Combat Sports Video Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-6.20%2B-orange)](https://gradio.app)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-green)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

 Upload a fight video (or paste a URL), and get real-time pose tracking, strike detection, round-by-round breakdowns, and interactive analytics — all through a browser interface.

---

## Features

- **2-Person Pose Tracking** — MediaPipe PoseLandmarker identifies both fighters' skeletons frame by frame
- **Live Processing Preview** — see annotated frames in real time as the video processes
- **Strike Detection & Classification** — automatically detects and classifies Jab, Cross, and Hook strikes
- **Interactive Dashboards** — 12+ Plotly charts covering velocity, guard angles, strike zones, round analysis, and more
- **Fighter Scorecard** — 9-dimension scoring with overall winner determination
- **Strike Combinations** — detects 2–3 strike sequences within 1.5-second windows
- **URL Download** — paste a YouTube or direct video URL and the app downloads it via yt-dlp
- **Export Capabilities** — download frame-level CSV, JSON summary, and the annotated MP4 video
- **Customizable** — tune detection confidence, strike velocity threshold, round duration, and annotation toggles

---

## Quick Start

### Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) installed and on your PATH

### Installation

```bash
# Clone the repository
git clone https://github.com/piee777/combat-sports-analytics.git
cd combat-sports-analytics

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

### Usage

1. **Upload a video** or paste a YouTube/UFC URL and click "Download from URL"
2. Optionally set **fighter names**, **detection confidence**, and **annotation toggles**
3. Click **"Analyze Video"** — watch the live preview as frames are processed
4. Explore results across 6 tabs:
   - **Summary** — annotated video, KPI cards, fight summary
   - **Analytics Dashboard** — velocity, guard angles, radar comparison, punch breakdown
   - **Strike Breakdown** — strike timeline, zone heatmap, activity heatmap
   - **Round Analysis** — strikes per round, scorecard
   - **Combinations & Log** — detected combinations and full strike event log
   - **Data & Export** — frame-level data, CSV/JSON/MP4 downloads

---

## Configuration

Key settings in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_COMPLEXITY` | `0` | 0 = lite, 1 = full, 2 = heavy (CPU performance trade-off) |
| `MIN_DETECTION_CONFIDENCE` | `0.5` | Minimum confidence to accept a pose detection |
| `MIN_TRACKING_CONFIDENCE` | `0.5` | Minimum confidence to track a pose between frames |
| `PROCESS_MAX_WIDTH` | `480` | Downscale frames wider than this before detection |
| `SKIP_FRAMES` | `2` | Process every Nth frame for pose detection |
| `VELOCITY_THRESHOLD` | `0.012` | Normalized wrist velocity to trigger strike detection |
| `ARM_EXTENSION_THRESHOLD` | `140°` | Minimum elbow extension to count as a strike |
| `PUNCH_COOLDOWN` | `6` | Frame cooldown between consecutive strike detections |

All parameters except these are adjustable from the Gradio UI.

---

## Project Structure

```
combat-sports-analytics/
├── app.py                 # Gradio UI, video orchestration, export
├── analytics.py           # FightAnalytics — data collection, strike detection, Plotly charts
├── cv_pipeline.py         # FighterPoseProcessor — MediaPipe pose tracking, frame rendering
├── config.py              # Constants, thresholds, landmark indices
├── pose_landmarker.task   # MediaPipe PoseLandmarker model file
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

### Data Flow

```
Video Input ──► OpenCV frame reading
                     │
                     ▼
         FighterPoseProcessor (MediaPipe)
         ┌────────────────────────────┐
         │ Pose detection             │
         │ Fighter assignment (L/R)   │
         │ Skeleton / bbox rendering  │
         │ Wrist speed & guard angle  │
         └────────────┬───────────────┘
                      │
         FightAnalytics
         ┌────────────────────────────┐
         │ Strike detection           │
         │ Classification (Jab/Cross) │
         │ Angle analysis             │
         │ Per-round stats            │
         └────────────┬───────────────┘
                      │
         Gradio UI (6 tabs)
         ┌────────────────────────────┐
         │ Summary & KPIs             │
         │ Analytics charts           │
         │ Strike breakdown           │
         │ Round analysis             │
         │ Combinations & log         │
         │ CSV / JSON / video export  │
         └────────────────────────────┘
```

---

## Export Formats

| Format | Content |
|---|---|
| **CSV** | Per-frame metrics: wrist speed, guard angle, strike events, joint angles for both fighters |
| **JSON** | Fight summary: total strikes, landed/missed counts, max speed, strike type breakdown |
| **MP4** | Full annotated video with skeleton overlays, bounding boxes, and real-time metric text |

---

## Dependencies

- [Gradio](https://gradio.app) — web UI framework
- [MediaPipe](https://mediapipe.dev) — pose estimation (PoseLandmarker)
- [OpenCV](https://opencv.org) — video I/O and rendering
- [Plotly](https://plotly.com/python/) — interactive charts
- [Pandas](https://pandas.pydata.org) — data management
- [NumPy](https://numpy.org) — numerical operations
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — video URL downloading
- [Pillow](https://python-pillow.org) — image handling

---

## Performance Notes

- **CPU only**: MediaPipe runs on CPU by default. For real-time use on lower-end machines, set `MODEL_COMPLEXITY = 0` and `SKIP_FRAMES = 2` (default).
- **Processed videos** are saved at original resolution. The downscale in `PROCESS_MAX_WIDTH` only affects the pose detection stage.
- **Long videos** (10+ minutes) may take several minutes to process depending on hardware.

---

## License

[MIT](LICENSE)
