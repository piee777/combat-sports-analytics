"""
Combat Sports Video Analytics - Main Streamlit Application

Entry point:  streamlit run app.py

Dark-themed dashboard with:
  - Sidebar: session management, video upload, analysis settings
  - Main panel (top): processed video player + summary stats
  - Main panel (bottom): interactive Plotly analytics charts
"""

import os
import sys
import tempfile
import time

import cv2
import numpy as np
import streamlit as st

from config import (
    MODEL_COMPLEXITY,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    FIGHTER_A_NAME,
    FIGHTER_B_NAME,
)
from cv_pipeline import FighterPoseProcessor
from analytics import FightAnalytics

# ---------------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Combat Sports Analytics",
    page_icon="boxing_glove",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for dark-themed polish
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark background accent */
    .block-container { padding-top: 1rem; }
    .stMetric { background: #1e1e2e; border-radius: 8px; padding: 12px; }
    /* Sidebar styling */
    [data-testid="stSidebar"] { background-color: #11111b; }
    /* Header accent */
    h1 { color: #f5a623 !important; }
    h3 { color: #cdd6f4 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def _init_state():
    """Create all session-state keys on first load."""
    defaults = {
        "uploaded_path": None,
        "output_path": None,
        "processing_done": False,
        "analytics": None,
        "fighter_a_name": FIGHTER_A_NAME,
        "fighter_b_name": FIGHTER_B_NAME,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    """Build the sidebar UI and return configuration values."""
    with st.sidebar:
        st.header("Session Management")
        st.session_state.fighter_a_name = st.text_input(
            "Fighter A Name", value=st.session_state.fighter_a_name)
        st.session_state.fighter_b_name = st.text_input(
            "Fighter B Name", value=st.session_state.fighter_b_name)

        if st.button("Reset Session"):
            for k in ("uploaded_path", "output_path", "processing_done",
                       "analytics"):
                st.session_state[k] = None
            st.session_state.processing_done = False
            st.rerun()

        st.divider()

        st.header("Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a fight video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload an MMA or Boxing video for analysis.",
        )

        st.divider()

        st.header("Analysis Settings")
        detection_conf = st.slider(
            "Detection Confidence", 0.1, 1.0,
            MIN_DETECTION_CONFIDENCE, 0.05,
        )
        tracking_conf = st.slider(
            "Tracking Confidence", 0.1, 1.0,
            MIN_TRACKING_CONFIDENCE, 0.05,
        )
        model_complexity = st.selectbox(
            "Model Complexity",
            options=[0, 1, 2],
            index=1,
            help="Higher = more accurate but slower.",
        )
        velocity_thresh = st.slider(
            "Strike Velocity Threshold", 0.5, 5.0, 1.2, 0.1,
        )

    return (uploaded_file, detection_conf, tracking_conf,
            model_complexity, velocity_thresh)


# ---------------------------------------------------------------------------
# Video Processing
# ---------------------------------------------------------------------------
def process_video(uploaded_file, detection_conf, tracking_conf,
                  model_complexity, velocity_thresh):
    """
    Full video processing pipeline: read → detect → annotate → write.
    Returns the path to the output video and the FightAnalytics object.
    """
    # Save uploaded file to a temporary location
    tmp_input = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4", dir=tempfile.gettempdir())
    tmp_input.write(uploaded_file.read())
    tmp_input.close()

    cap = cv2.VideoCapture(tmp_input.name)
    if not cap.isOpened():
        st.error("Failed to open the uploaded video. Please try another file.")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer
    tmp_output = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4", dir=tempfile.gettempdir())
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_output.name, fourcc, fps, (width, height))

    # Initialise pipeline and analytics
    processor = FighterPoseProcessor(
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf,
    )
    analytics = FightAnalytics(fps=fps)

    # Override threshold from sidebar
    import config
    config.VELOCITY_THRESHOLD = velocity_thresh / 100.0  # normalise

    # Progress UI
    progress_bar = st.progress(0, text="Starting analysis...")
    status_text = st.empty()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, metrics = processor.process_frame(
                frame, timestamp_ms=int(frame_count * 1000 / fps)
            )
            writer.write(annotated)
            analytics.record_frame(frame_count, metrics)
            frame_count += 1

            # Update progress every N frames for speed
            if frame_count % 5 == 0 or frame_count == total_frames:
                pct = frame_count / total_frames if total_frames else 0
                progress_bar.progress(
                    min(pct, 1.0),
                    text=f"Processing frame {frame_count}/{total_frames} "
                         f"({pct*100:.0f}%)",
                )
                status_text.caption(
                    f"Elapsed: {frame_count/fps:.1f}s / "
                    f"{total_frames/fps:.1f}s video"
                )
    except Exception as e:
        st.error(f"Processing error: {e}")
    finally:
        cap.release()
        writer.release()
        processor.close()

    progress_bar.progress(1.0, text="Analysis complete!")

    # Build the analytics DataFrame
    analytics.build_dataframe()

    return tmp_output.name, analytics


# ---------------------------------------------------------------------------
# Summary Metric Cards
# ---------------------------------------------------------------------------
def render_summary_cards(analytics, fa_name, fb_name):
    """Render top-level KPI cards above the charts."""
    s = analytics.summary
    dur = s["fight_duration_sec"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fight Duration", f"{dur:.1f}s")
    c2.metric(f"{fa_name} — Total Strikes", s["fighter_a_total_strikes"],
              f"{s['fighter_a_landed']} landed")
    c3.metric(f"{fb_name} — Total Strikes", s["fighter_b_total_strikes"],
              f"{s['fighter_b_landed']} landed")
    c4.metric("Max Strike Speed", f"{s['max_speed']:.2f} m/s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.markdown("# Combat Sports Video Analytics")
    st.markdown(
        "Upload a fight video, process it with real-time pose tracking, "
        "and explore comprehensive fight analytics."
    )

    # Sidebar
    (uploaded_file, detection_conf, tracking_conf,
     model_complexity, velocity_thresh) = render_sidebar()

    # ---- Upload & Process ----
    if uploaded_file is not None and not st.session_state.processing_done:
        st.info("Video uploaded. Click below to start analysis.")

    if uploaded_file is not None and st.button(
        "Process Video", type="primary", use_container_width=True
    ):
        with st.spinner("Initializing pose detection model..."):
            output_path, analytics = process_video(
                uploaded_file, detection_conf, tracking_conf,
                model_complexity, velocity_thresh,
            )
        if output_path and analytics:
            st.session_state.uploaded_path = output_path
            st.session_state.output_path = output_path
            st.session_state.analytics = analytics
            st.session_state.processing_done = True
            st.success("Analysis complete! Scroll down to view results.")
            st.rerun()

    # ---- Display Results ----
    if st.session_state.processing_done and st.session_state.analytics:
        analytics = st.session_state.analytics
        fa = st.session_state.fighter_a_name
        fb = st.session_state.fighter_b_name

        st.divider()
        st.header("Processed Video")
        if st.session_state.output_path and os.path.exists(
            st.session_state.output_path
        ):
            st.video(st.session_state.output_path)

        st.divider()
        st.header("Summary Statistics")
        render_summary_cards(analytics, fa, fb)

        st.divider()
        st.header("Full Fight Statistics")

        # Row 1: Bar chart + Gauge
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(
                analytics.render_punches_bar(),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                analytics.render_max_speed_gauge(),
                use_container_width=True,
            )

        # Row 2: Pie charts
        st.plotly_chart(
            analytics.render_strike_types_pie(),
            use_container_width=True,
        )

        # Row 3: Timeline charts
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(
                analytics.render_velocity_timeline(),
                use_container_width=True,
            )
        with col4:
            st.plotly_chart(
                analytics.render_guard_angle_timeline(),
                use_container_width=True,
            )

        # Raw data expander
        with st.expander("View Raw Frame-Level Data"):
            if analytics.df is not None:
                st.dataframe(
                    analytics.df[[
                        "frame", "timestamp_sec",
                        "fighter_a_detected", "fighter_a_wrist_speed",
                        "fighter_a_guard_angle", "fighter_a_strike_detected",
                        "fighter_b_detected", "fighter_b_wrist_speed",
                        "fighter_b_guard_angle", "fighter_b_strike_detected",
                    ]],
                    use_container_width=True,
                )
    else:
        st.info(
            "Upload a fight video in the sidebar to begin analysis."
        )


if __name__ == "__main__":
    main()
