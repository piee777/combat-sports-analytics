"""
Combat Sports Video Analytics - Enhanced Gradio Application

Entry point:  python app.py
Opens on http://localhost:7860

Features:
  - Upload video or paste URL (YouTube, UFC, etc.)
  - Real-time MediaPipe 2-person pose tracking
  - Strike detection & classification (Jab/Cross/Hook)
  - Round-by-round analysis
  - Strike combination detection
  - Fighter scorecard
  - Elbow/knee angle analysis
  - Activity heatmap
  - Annotation toggles (skeleton, bbox, metrics)
  - KPI summary cards
  - Strike event log
  - Interactive Plotly charts
  - CSV/JSON/Video export
"""

import json
import os
import subprocess
import tempfile

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    FIGHTER_A_NAME,
    FIGHTER_B_NAME,
)
from cv_pipeline import FighterPoseProcessor
from analytics import FightAnalytics


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.main-title { text-align: center; margin-bottom: 0.5em; }
.main-title h1 { font-size: 2em; font-weight: 700; color: #f5a623 !important; }
.main-title p { color: #94a3b8; font-size: 1.05em; }

.kpi-row { display: flex; gap: 12px; margin: 12px 0; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 140px; background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155; border-radius: 12px; padding: 16px 18px;
    text-align: center; transition: transform 0.15s;
}
.kpi-card:hover { transform: translateY(-2px); border-color: #f5a623; }
.kpi-label { font-size: 0.78em; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-value { font-size: 1.6em; font-weight: 700; color: #f5a623; margin: 4px 0; }
.kpi-sub { font-size: 0.82em; color: #64748b; }
"""


# ---------------------------------------------------------------------------
# URL Download
# ---------------------------------------------------------------------------

def download_video_from_url(url):
    """Download video from URL using yt-dlp. Returns (path, title, duration)."""
    try:
        import yt_dlp
    except ImportError:
        raise gr.Error("yt-dlp is not installed. Run: pip install yt-dlp")

    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "downloaded_video.mp4")

    ydl_opts = {
        "outtmpl": output_path,
        "format": "best[height<=1080][ext=mp4]/best[height<=1080]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Unknown")
        duration = info.get("duration", 0)

    if not os.path.exists(output_path):
        for f in os.listdir(tmp_dir):
            if f.endswith((".mp4", ".mkv", ".webm")):
                output_path = os.path.join(tmp_dir, f)
                break

    return output_path, title, duration


def handle_url_download(url):
    """Download a video from URL and return path + status."""
    if not url or not url.strip():
        raise gr.Error("Please enter a URL.")

    try:
        path, title, duration = download_video_from_url(url.strip())
        status = f"Downloaded: **{title}** ({duration}s)"
        return path, status
    except Exception as e:
        raise gr.Error(f"Download failed: {e}")


# ---------------------------------------------------------------------------
# Core Processing
# ---------------------------------------------------------------------------

def process_video(
    video_path,
    fighter_a_name,
    fighter_b_name,
    detection_conf,
    tracking_conf,
    velocity_thresh,
    show_skeleton,
    show_bbox,
    show_metrics_flag,
    round_duration_sec,
    progress=gr.Progress(),
):
    """Full video processing pipeline with annotation controls."""
    if video_path is None:
        raise gr.Error("Please upload a video or provide a URL first.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Failed to open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    tmp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
    tmp_raw.close()
    tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_output.close()

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(tmp_raw.name, fourcc, fps, (width, height))

    processor = FighterPoseProcessor(
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf,
    )
    analytics = FightAnalytics(fps=fps)

    import config
    config.VELOCITY_THRESHOLD = velocity_thresh / 100.0

    # Update fighter names in config
    config.FIGHTER_A_NAME = fighter_a_name
    config.FIGHTER_B_NAME = fighter_b_name

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, metrics = processor.process_frame(
                frame, timestamp_ms=int(frame_count * 1000 / fps)
            )

            # Apply annotation toggles
            if not (show_skeleton and show_bbox and show_metrics_flag):
                annotated = _apply_annotation_toggles(
                    frame.copy(), processor, metrics,
                    show_skeleton, show_bbox, show_metrics_flag,
                    fighter_a_name, fighter_b_name,
                )

            writer.write(annotated)
            analytics.record_frame(frame_count, metrics)
            frame_count += 1

            # Yield real-time preview frame
            preview = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            yield (preview,) + (gr.skip(),) * 23

            if total_frames > 0:
                progress(frame_count / total_frames, desc=f"Frame {frame_count}/{total_frames}")
            elif frame_count % 30 == 0:
                progress(0.5, desc=f"Processed {frame_count} frames...")
    except Exception as e:
        raise gr.Error(f"Processing error: {e}")
    finally:
        cap.release()
        writer.release()
        processor.close()

    # Convert to browser-compatible format
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_raw.name,
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-pix_fmt", "yuv420p", tmp_output.name],
            check=True, capture_output=True, timeout=300,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        try:
            os.replace(tmp_raw.name, tmp_output.name)
        except OSError:
            pass
    finally:
        try:
            if os.path.exists(tmp_raw.name):
                os.unlink(tmp_raw.name)
        except OSError:
            pass

    analytics.build_dataframe()
    s = analytics.summary
    fa, fb = fighter_a_name, fighter_b_name

    # ---- KPI Summary Cards (HTML) ----
    kpi_html = f"""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-label">Duration</div>
        <div class="kpi-value">{s['fight_duration_sec']:.1f}s</div>
        <div class="kpi-sub">fight time</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">{fa}</div>
        <div class="kpi-value">{s['fighter_a_total_strikes']}</div>
        <div class="kpi-sub">{s['fighter_a_landed']} landed / {s['fighter_a_missed']} missed</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">{fb}</div>
        <div class="kpi-value">{s['fighter_b_total_strikes']}</div>
        <div class="kpi-sub">{s['fighter_b_landed']} landed / {s['fighter_b_missed']} missed</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Max Speed</div>
        <div class="kpi-value">{s['max_speed']:.2f}</div>
        <div class="kpi-sub">m/s peak strike</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Total Strikes</div>
        <div class="kpi-value">{s['fighter_a_total_strikes'] + s['fighter_b_total_strikes']}</div>
        <div class="kpi-sub">combined thrown</div>
    </div>
</div>
"""

    # ---- Summary Markdown ----
    acc_a = round(s['fighter_a_landed'] / max(s['fighter_a_total_strikes'], 1) * 100, 1)
    acc_b = round(s['fighter_b_landed'] / max(s['fighter_b_total_strikes'], 1) * 100, 1)
    avg_spd_a = 0.0
    avg_spd_b = 0.0
    if analytics.df is not None:
        spd_a_vals = analytics.df["fighter_a_wrist_speed"].dropna()
        spd_b_vals = analytics.df["fighter_b_wrist_speed"].dropna()
        avg_spd_a = float(spd_a_vals.mean()) if len(spd_a_vals) > 0 else 0.0
        avg_spd_b = float(spd_b_vals.mean()) if len(spd_b_vals) > 0 else 0.0

    summary_md = f"""
## Detailed Summary

| Metric | {fa} | {fb} |
|--------|------|------|
| **Total Strikes** | {s['fighter_a_total_strikes']} | {s['fighter_b_total_strikes']} |
| **Landed** | {s['fighter_a_landed']} | {s['fighter_b_landed']} |
| **Missed** | {s['fighter_a_missed']} | {s['fighter_b_missed']} |
| **Accuracy** | {acc_a}% | {acc_b}% |
| **Jabs** | {s['fighter_a_strike_types']['Jab']} | {s['fighter_b_strike_types']['Jab']} |
| **Crosses** | {s['fighter_a_strike_types']['Cross']} | {s['fighter_b_strike_types']['Cross']} |
| **Hooks** | {s['fighter_a_strike_types']['Hook']} | {s['fighter_b_strike_types']['Hook']} |
| **Avg Speed** | {avg_spd_a:.2f} m/s | {avg_spd_b:.2f} m/s |
| **Max Speed** | {s['max_speed']:.2f} m/s | - |

- **Fight Duration:** {s['fight_duration_sec']:.1f}s
"""

    # ---- Charts ----
    bar_chart = analytics.render_punches_bar()
    gauge_chart = analytics.render_max_speed_gauge()
    radar_chart = analytics.render_radar_comparison()
    pie_chart = analytics.render_strike_types_pie()
    velocity_chart = analytics.render_velocity_timeline()
    guard_chart = analytics.render_guard_angle_timeline()
    strike_timeline = analytics.render_strike_timeline()
    cumulative_chart = analytics.render_cumulative_strikes()
    heatmap_chart = analytics.render_body_heatmap()
    elbow_chart = analytics.render_elbow_angle_timeline()
    knee_chart = analytics.render_knee_angle_timeline()
    activity_heatmap = analytics.render_activity_heatmap()
    round_chart_fig = analytics.render_round_chart(round_duration_sec)

    # ---- Strike Log ----
    strike_log = analytics.get_strike_log()
    strike_log_df = pd.DataFrame(strike_log) if strike_log else pd.DataFrame()

    # ---- Combinations ----
    combos = analytics.get_combinations()
    combos_df = pd.DataFrame(combos) if combos else pd.DataFrame()

    # ---- Scorecard ----
    scorecard = analytics.get_scorecard()
    scorecard_md = _format_scorecard(scorecard, fa, fb)

    # ---- Export Data ----
    display_cols = [
        "frame", "timestamp_sec",
        "fighter_a_detected", "fighter_a_wrist_speed",
        "fighter_a_guard_angle", "fighter_a_strike_detected",
        "fighter_b_detected", "fighter_b_wrist_speed",
        "fighter_b_guard_angle", "fighter_b_strike_detected",
    ]
    available = [c for c in display_cols if c in analytics.df.columns]
    display_df = analytics.df[available].copy()
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]

    csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    json_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
    analytics.df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(s, f, indent=2)

    yield (
        None,  # clear live preview
        tmp_output.name,
        kpi_html,
        summary_md,
        bar_chart,
        gauge_chart,
        radar_chart,
        pie_chart,
        velocity_chart,
        guard_chart,
        strike_timeline,
        cumulative_chart,
        heatmap_chart,
        elbow_chart,
        knee_chart,
        activity_heatmap,
        round_chart_fig,
        scorecard_md,
        strike_log_df,
        combos_df,
        display_df,
        csv_path,
        json_path,
        tmp_output.name,
    )


# ---------------------------------------------------------------------------
# Annotation Toggle Helper
# ---------------------------------------------------------------------------

def _apply_annotation_toggles(frame, processor, metrics,
                               show_skeleton, show_bbox, show_metrics,
                               fa_name, fb_name):
    """Re-draw only the requested annotation layers on the raw frame."""
    result = frame.copy()

    fighter_configs = [
        ("fighter_a", (255, 100, 50), fa_name),
        ("fighter_b", (50, 50, 255), fb_name),
    ]

    for fighter_key, color, name in fighter_configs:
        fm = metrics.get(fighter_key, {})
        px = fm.get("landmarks_px")
        if px is None:
            continue

        if show_skeleton:
            processor._draw_skeleton_on_frame(result, px, color)
        if show_bbox:
            processor._draw_bounding_box(result, px, color, name)
        if show_metrics:
            ga = fm.get("guard_angle") or 0.0
            sp = fm.get("wrist_speed") or 0.0
            y_off = 25 if fighter_key == "fighter_a" else 75
            processor._overlay_metrics(result, name, color, y_off, ga, sp)

    return result


# ---------------------------------------------------------------------------
# Scorecard Formatter
# ---------------------------------------------------------------------------

def _format_scorecard(card, fa, fb):
    """Format the scorecard as a markdown table."""
    if not card or "_totals" not in card:
        return "## Fighter Scorecard\n\n*Insufficient data for scoring.*"

    lines = [
        "## Fighter Scorecard",
        f"### Winner: **{card.get('_winner', 'Draw')}**",
        "",
        f"| Criterion | {fa} | {fb} | Winner |",
        "|-----------|------|------|--------|",
    ]
    for metric, scores in card.items():
        if metric.startswith("_"):
            continue
        winner = scores.get("winner", "Tie")
        lines.append(f"| {metric} | {scores[fa]} | {scores[fb]} | {winner} |")

    totals = card["_totals"]
    lines.append(
        f"| **TOTAL** | **{totals[fa]}** | **{totals[fb]}** "
        f"| **{card.get('_winner', 'Draw')}** |"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Combat Sports Analytics") as demo:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    gr.HTML("""
    <div class="main-title">
        <h1>Combat Sports Video Analytics</h1>
        <p>AI-powered fight analysis with real-time pose tracking, strike detection, and interactive charts.</p>
    </div>
    """)

    # ---- Input Section ----
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Video Input")
            video_input = gr.Video(label="Upload Fight Video", height=200)
            gr.Markdown("**Or paste a URL:**")
            url_input = gr.Textbox(
                label="Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                info="YouTube, UFC Fight Pass, Dailymotion, or direct video URL",
            )
            download_btn = gr.Button("Download from URL", variant="secondary", size="sm")
            download_status = gr.Markdown(visible=True)
            url_video_path = gr.State(value=None)

        with gr.Column(scale=1):
            gr.Markdown("### Fighter Names")
            fa_name_input = gr.Textbox(value=FIGHTER_A_NAME, label="Fighter A (Blue)")
            fb_name_input = gr.Textbox(value=FIGHTER_B_NAME, label="Fighter B (Red)")

            gr.Markdown("### Detection Settings")
            det_conf = gr.Slider(0.1, 1.0, value=MIN_DETECTION_CONFIDENCE, step=0.05, label="Detection Confidence")
            trk_conf = gr.Slider(0.1, 1.0, value=MIN_TRACKING_CONFIDENCE, step=0.05, label="Tracking Confidence")
            vel_thresh = gr.Slider(0.5, 5.0, value=1.2, step=0.1, label="Strike Velocity Threshold")

        with gr.Column(scale=1):
            gr.Markdown("### Annotation Options")
            show_skeleton = gr.Checkbox(label="Show Skeleton", value=True)
            show_bbox = gr.Checkbox(label="Show Bounding Boxes", value=True)
            show_metrics = gr.Checkbox(label="Show Metric Overlays", value=True)

            gr.Markdown("### Round Settings")
            round_duration = gr.Slider(
                60, 600, value=180, step=30,
                label="Round Duration (seconds)",
                info="180s = 3 min (standard MMA round)",
            )

    process_btn = gr.Button("Analyze Video", variant="primary", size="lg")

    # ---- Output Tabs ----
    with gr.Tabs():
        with gr.Tab("Summary"):
            output_video = gr.Video(label="Annotated Video", height=400)
            live_preview = gr.Image(label="Live Processing Preview", height=400)
            kpi_html = gr.HTML()
            summary_md = gr.Markdown()

        with gr.Tab("Analytics Dashboard"):
            with gr.Row():
                bar_chart = gr.Plot(label="Punches Thrown vs Landed")
                gauge_chart = gr.Plot(label="Max Strike Speed")
            with gr.Row():
                radar_chart = gr.Plot(label="Fighter Comparison")
                pie_chart = gr.Plot(label="Strike Type Breakdown")
            with gr.Row():
                velocity_chart = gr.Plot(label="Velocity Over Time")
                guard_chart = gr.Plot(label="Guard Angle Over Time")
            with gr.Row():
                elbow_chart = gr.Plot(label="Elbow Angle Analysis")
                knee_chart = gr.Plot(label="Knee Angle Analysis")

        with gr.Tab("Strike Breakdown"):
            with gr.Row():
                strike_timeline = gr.Plot(label="Strike Event Timeline")
                cumulative_chart = gr.Plot(label="Cumulative Strikes")
            heatmap_chart = gr.Plot(label="Strike Zone Heatmap")
            activity_heatmap = gr.Plot(label="Fighter Activity Heatmap")

        with gr.Tab("Round Analysis"):
            round_chart = gr.Plot(label="Strikes Per Round")
            scorecard_md = gr.Markdown()

        with gr.Tab("Combinations & Log"):
            gr.Markdown("### Detected Strike Combinations")
            gr.Markdown("*Combinations: 2+ strikes within 1.5 seconds of each other.*")
            combos_table = gr.Dataframe(
                label="Strike Combinations", interactive=False, wrap=True,
            )
            gr.Markdown("### Complete Strike Event Log")
            strike_log_table = gr.Dataframe(
                label="All Strikes", interactive=False, wrap=True,
            )

        with gr.Tab("Data & Export"):
            dataframe = gr.Dataframe(label="Frame-Level Data", interactive=False)
            with gr.Row():
                csv_file = gr.File(label="Download CSV")
                json_file = gr.File(label="Download JSON Summary")
            video_download = gr.File(label="Download Processed Video")

    # ---- Event Wiring ----

    def resolve_video_input(video_from_upload, url_path):
        if video_from_upload is not None:
            return video_from_upload
        if url_path is not None and os.path.exists(str(url_path)):
            return url_path
        return None

    download_btn.click(
        fn=handle_url_download,
        inputs=[url_input],
        outputs=[url_video_path, download_status],
    )

    def run_analysis(video_upload, url_path, fa, fb, det, trk, vel,
                     skel, bbox, metrics_flag, round_dur):
        video = resolve_video_input(video_upload, url_path)
        if video is None:
            raise gr.Error("Please upload a video or download from a URL first.")

        yield from process_video(
            video, fa, fb, det, trk, vel,
            skel, bbox, metrics_flag, round_dur,
        )

    process_btn.click(
        fn=run_analysis,
        inputs=[
            video_input, url_video_path,
            fa_name_input, fb_name_input,
            det_conf, trk_conf, vel_thresh,
            show_skeleton, show_bbox, show_metrics,
            round_duration,
        ],
        outputs=[
            live_preview,
            output_video, kpi_html, summary_md,
            bar_chart, gauge_chart, radar_chart, pie_chart,
            velocity_chart, guard_chart,
            strike_timeline, cumulative_chart, heatmap_chart,
            elbow_chart, knee_chart, activity_heatmap,
            round_chart, scorecard_md,
            strike_log_table, combos_table,
            dataframe, csv_file, json_file, video_download,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
