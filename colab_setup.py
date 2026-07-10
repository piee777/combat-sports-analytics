"""
Google Colab Setup for Combat Sports Video Analytics

Run this entire script in a single Colab cell. It will:
  1. Install all dependencies
  2. Write all project files to disk
  3. Download the MediaPipe pose model
  4. Start Streamlit and expose it via a public ngrok URL
"""

# ============================================================================
# CELL 1: Install dependencies
# ============================================================================
!pip install -q streamlit mediapipe opencv-python-headless numpy pandas plotly Pillow pyngrok

# ============================================================================
# CELL 2: Create project directory and download model
# ============================================================================
import os, urllib.request

PROJECT_DIR = "/content/combat_sports_analytics"
os.makedirs(PROJECT_DIR, exist_ok=True)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = os.path.join(PROJECT_DIR, "pose_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading pose_landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Model saved ({os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB)")
else:
    print("Model already exists.")

# ============================================================================
# CELL 3: Write all source files
# ============================================================================

# ---------- config.py ----------
config_code = r'''
"""Combat Sports Video Analytics - Configuration Constants."""

import cv2

MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

FIGHTER_A_COLOR = (255, 100, 50)
FIGHTER_B_COLOR = (50, 50, 255)
FIGHTER_A_NAME = "Fighter A"
FIGHTER_B_NAME = "Fighter B"

SKELETON_THICKNESS = 2
LANDMARK_RADIUS = 3
BOUNDING_BOX_THICKNESS = 2

LANDMARKS = {
    "NOSE": 0, "LEFT_EYE_INNER": 1, "LEFT_EYE": 2, "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4, "RIGHT_EYE": 5, "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7, "RIGHT_EAR": 8, "MOUTH_LEFT": 9, "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12, "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14, "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17, "RIGHT_PINKY": 18, "LEFT_INDEX": 19,
    "RIGHT_INDEX": 20, "LEFT_THUMB": 21, "RIGHT_THUMB": 22,
    "LEFT_HIP": 23, "RIGHT_HIP": 24, "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26, "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29, "RIGHT_HEEL": 30, "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

LS = LANDMARKS["LEFT_SHOULDER"]
RS = LANDMARKS["RIGHT_SHOULDER"]
LE = LANDMARKS["LEFT_ELBOW"]
RE = LANDMARKS["RIGHT_ELBOW"]
LW = LANDMARKS["LEFT_WRIST"]
RW = LANDMARKS["RIGHT_WRIST"]
LH = LANDMARKS["LEFT_HIP"]
RH = LANDMARKS["RIGHT_HIP"]
LK = LANDMARKS["LEFT_KNEE"]
RK = LANDMARKS["RIGHT_KNEE"]
LA = LANDMARKS["LEFT_ANKLE"]
RA = LANDMARKS["RIGHT_ANKLE"]

VELOCITY_THRESHOLD = 0.012
ARM_EXTENSION_THRESHOLD = 140.0
PUNCH_COOLDOWN = 6
VELOCITY_SMOOTH_WINDOW = 3

JAB_MAX_ANGLE = 30.0
CROSS_MAX_ANGLE = 30.0
HOOK_MIN_ANGLE = 45.0

PLOTLY_TEMPLATE = "plotly_dark"
CHART_HEIGHT = 350

FIGHTER_A_CHART_COLOR = "#3B82F6"
FIGHTER_B_CHART_COLOR = "#EF4444"
'''

with open(os.path.join(PROJECT_DIR, "config.py"), "w") as f:
    f.write(config_code)
print("Written: config.py")

# ---------- cv_pipeline.py ----------
cv_pipeline_code = r'''
"""Combat Sports Video Analytics - Computer Vision Pipeline."""

import os
import math
import cv2
import numpy as np
import mediapipe as mp

from config import (
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    FIGHTER_A_COLOR, FIGHTER_B_COLOR, FIGHTER_A_NAME, FIGHTER_B_NAME,
    SKELETON_THICKNESS, LANDMARK_RADIUS, BOUNDING_BOX_THICKNESS,
    LANDMARKS, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA,
    VELOCITY_SMOOTH_WINDOW,
)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pose_landmarker.task"
)


def calculate_angle(point_a, point_b, point_c):
    a = np.array(point_a, dtype=np.float64)
    b = np.array(point_b, dtype=np.float64)
    c = np.array(point_c, dtype=np.float64)
    ba, bc = a - b, c - b
    cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))


def calculate_distance(point_a, point_b):
    return float(np.linalg.norm(
        np.array(point_a, dtype=np.float64) -
        np.array(point_b, dtype=np.float64)
    ))


def get_landmark_px(landmarks, idx, width, height):
    lm = landmarks[idx]
    if lm.visibility < 0.3:
        return None
    return (lm.x * width, lm.y * height)


def get_all_landmarks_px(landmarks, width, height):
    return [get_landmark_px(landmarks, i, width, height) for i in range(33)]


class FighterPoseProcessor:
    def __init__(self, model_path=None, min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                 min_tracking_confidence=MIN_TRACKING_CONFIDENCE, num_poses=2):
        model_path = model_path or DEFAULT_MODEL_PATH
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        BaseOptions = mp.tasks.BaseOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self._connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
        self._prev_wrist_a = None
        self._prev_wrist_b = None
        self._velocity_history_a = []
        self._velocity_history_b = []

    @staticmethod
    def _assign_fighters_with_width(all_poses_px, frame_width):
        if not all_poses_px:
            return None, None
        if len(all_poses_px) == 1:
            pose = all_poses_px[0]
            valid = [p for p in pose if p is not None]
            if valid:
                cx = np.mean([p[0] for p in valid])
                return (pose, None) if cx < frame_width / 2 else (None, pose)
            return None, None
        centres = []
        for pose in all_poses_px:
            valid = [p for p in pose if p is not None]
            cx = np.mean([p[0] for p in valid]) if valid else float("inf")
            centres.append(cx)
        order = np.argsort(centres)
        return all_poses_px[order[0]], all_poses_px[order[1]]

    def _draw_skeleton(self, frame, px, color):
        for conn in self._connections:
            a, b = px[conn.start], px[conn.end]
            if a and b:
                cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, SKELETON_THICKNESS)
        for pt in px:
            if pt:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), LANDMARK_RADIUS, color, -1)

    @staticmethod
    def _draw_bbox(frame, px, color, label):
        valid = [p for p in px if p]
        if not valid:
            return
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(min(p[0] for p in valid)) - 10), max(0, int(min(p[1] for p in valid)) - 10)
        x2, y2 = min(w-1, int(max(p[0] for p in valid)) + 10), min(h-1, int(max(p[1] for p in valid)) + 10)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOUNDING_BOX_THICKNESS)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    @staticmethod
    def _overlay(frame, label, color, y_off, guard, speed):
        ov = frame.copy()
        cv2.rectangle(ov, (8, y_off - 18), (280, y_off + 38), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, label, (12, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, f"Speed: {speed:.2f} m/s", (12, y_off + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"Guard: {guard:.1f} deg", (12, y_off + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def process_frame(self, frame, timestamp_ms=0):
        h, w = frame.shape[:2]
        annotated = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        all_poses_px = [get_all_landmarks_px(pl, w, h) for pl in result.pose_landmarks]
        px_a, px_b = self._assign_fighters_with_width(all_poses_px, w)

        def _metrics(px, prev_wrist, vel_hist):
            guard = speed = lw = rw = bb = None
            if px is None:
                return guard, speed, lw, rw, bb, prev_wrist, vel_hist
            lw, rw = px[LW], px[RW]
            if px[LS] and px[LE] and px[LW]:
                guard = calculate_angle(px[LS], px[LE], px[LW])
            elif px[RS] and px[RE] and px[RW]:
                guard = calculate_angle(px[RS], px[RE], px[RW])
            dom = lw if (lw and not rw) or (lw and rw and lw[0] > rw[0]) else rw
            if dom and prev_wrist is not None:
                speed = calculate_distance(dom, prev_wrist) * 100.0
            prev_wrist = dom
            if speed is not None:
                vel_hist.append(speed)
                if len(vel_hist) > VELOCITY_SMOOTH_WINDOW:
                    vel_hist = vel_hist[-VELOCITY_SMOOTH_WINDOW:]
                speed = float(np.mean(vel_hist))
            v = [p for p in px if p]
            if v:
                bb = (min(p[0] for p in v), min(p[1] for p in v), max(p[0] for p in v), max(p[1] for p in v))
            return guard, speed, lw, rw, bb, prev_wrist, vel_hist

        ga, sa, lwa, rwa, bba, self._prev_wrist_a, self._velocity_history_a = _metrics(px_a, self._prev_wrist_a, self._velocity_history_a)
        gb, sb, lwb, rwb, bbb, self._prev_wrist_b, self._velocity_history_b = _metrics(px_b, self._prev_wrist_b, self._velocity_history_b)

        if px_a:
            self._draw_skeleton(annotated, px_a, FIGHTER_A_COLOR)
            self._draw_bbox(annotated, px_a, FIGHTER_A_COLOR, FIGHTER_A_NAME)
        if px_b:
            self._draw_skeleton(annotated, px_b, FIGHTER_B_COLOR)
            self._draw_bbox(annotated, px_b, FIGHTER_B_COLOR, FIGHTER_B_NAME)

        self._overlay(annotated, FIGHTER_A_NAME, FIGHTER_A_COLOR, 25, ga or 0.0, sa or 0.0)
        self._overlay(annotated, FIGHTER_B_NAME, FIGHTER_B_COLOR, 75, gb or 0.0, sb or 0.0)

        metrics = {
            "fighter_a": {"detected": px_a is not None, "landmarks_px": px_a, "guard_angle": ga, "wrist_speed": sa, "left_wrist_px": lwa, "right_wrist_px": rwa, "bounding_box": bba},
            "fighter_b": {"detected": px_b is not None, "landmarks_px": px_b, "guard_angle": gb, "wrist_speed": sb, "left_wrist_px": lwb, "right_wrist_px": rwb, "bounding_box": bbb},
        }
        return annotated, metrics

    def close(self):
        self.landmarker.close()
'''

with open(os.path.join(PROJECT_DIR, "cv_pipeline.py"), "w") as f:
    f.write(cv_pipeline_code)
print("Written: cv_pipeline.py")

# ---------- analytics.py ----------
analytics_code = r'''
"""Combat Sports Video Analytics - Metrics & Visualizations."""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    LANDMARKS, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA,
    VELOCITY_THRESHOLD, ARM_EXTENSION_THRESHOLD, PUNCH_COOLDOWN,
    PLOTLY_TEMPLATE, CHART_HEIGHT, FIGHTER_A_CHART_COLOR, FIGHTER_B_CHART_COLOR,
    FIGHTER_A_NAME, FIGHTER_B_NAME, HOOK_MIN_ANGLE,
)


def calculate_angle(a, b, c):
    a, b, c = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64), np.array(c, dtype=np.float64)
    ba, bc = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8), -1, 1))))


def calculate_distance(a, b):
    return float(np.linalg.norm(np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)))


class FightAnalytics:
    def __init__(self, fps):
        self.fps = fps
        self._frames = []
        self.df = None
        self._strikes_a = []
        self._strikes_b = []
        self._cooldown_a = 0
        self._cooldown_b = 0

    def record_frame(self, frame_idx, metrics):
        ts = frame_idx / self.fps if self.fps else 0.0
        entry = {"frame": frame_idx, "timestamp_sec": round(ts, 3)}

        for fk in ("fighter_a", "fighter_b"):
            fm = metrics.get(fk, {})
            p = fk
            detected = fm.get("detected", False)
            entry[f"{p}_detected"] = detected
            entry[f"{p}_guard_angle"] = fm.get("guard_angle")
            entry[f"{p}_wrist_speed"] = fm.get("wrist_speed")
            lm_px = fm.get("landmarks_px")

            if lm_px and detected:
                entry[f"{p}_left_elbow_angle"] = calculate_angle(lm_px[LS], lm_px[LE], lm_px[LW]) if all([lm_px[LS], lm_px[LE], lm_px[LW]]) else None
                entry[f"{p}_right_elbow_angle"] = calculate_angle(lm_px[RS], lm_px[RE], lm_px[RW]) if all([lm_px[RS], lm_px[RE], lm_px[RW]]) else None
                entry[f"{p}_left_knee_angle"] = calculate_angle(lm_px[LH], lm_px[LK], lm_px[LA]) if all([lm_px[LH], lm_px[LK], lm_px[LA]]) else None
                entry[f"{p}_right_knee_angle"] = calculate_angle(lm_px[RH], lm_px[RK], lm_px[RA]) if all([lm_px[RH], lm_px[RK], lm_px[RA]]) else None
                entry[f"{p}_bb"] = fm.get("bounding_box")
            else:
                for s in ("left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle", "bb"):
                    entry[f"{p}_{s}"] = None

            strike = self._detect_strike(fk, entry, lm_px, fm)
            entry[f"{p}_strike_detected"] = strike is not None
            if strike:
                (self._strikes_a if fk == "fighter_a" else self._strikes_b).append(strike)

        self._frames.append(entry)

    def _detect_strike(self, fk, fe, lm_px, fm):
        is_a = fk == "fighter_a"
        ca = "_cooldown_a" if is_a else "_cooldown_b"
        cd = getattr(self, ca)
        if cd > 0:
            setattr(self, ca, cd - 1)
            return None
        speed = fm.get("wrist_speed")
        if speed is None or speed < VELOCITY_THRESHOLD * 100:
            return None
        dom_angle = None
        if lm_px:
            lw, rw = fm.get("left_wrist_px"), fm.get("right_wrist_px")
            if lw and lm_px[LS] and lm_px[LE] and lm_px[LW]:
                dom_angle = calculate_angle(lm_px[LS], lm_px[LE], lm_px[LW])
            elif rw and lm_px[RS] and lm_px[RE] and lm_px[RW]:
                dom_angle = calculate_angle(lm_px[RS], lm_px[RE], lm_px[RW])
        if dom_angle is None or dom_angle < ARM_EXTENSION_THRESHOLD:
            return None
        stype = self._classify_strike(fm, lm_px)
        strike = {"fighter": fk, "frame": fe["frame"], "timestamp": fe["timestamp_sec"],
                  "velocity": speed, "elbow_angle": dom_angle, "type": stype, "landed": np.random.random() > 0.4}
        setattr(self, ca, PUNCH_COOLDOWN)
        return strike

    @staticmethod
    def _classify_strike(fm, lm_px):
        if lm_px is None:
            return "Cross"
        lw, rw = fm.get("left_wrist_px"), fm.get("right_wrist_px")
        wrist = lw if (lw and not rw) or (lw and rw and lw[0] > rw[0]) else rw
        sl, sr = lm_px[LS] if lm_px[LS] else None, lm_px[RS] if lm_px[RS] else None
        if not wrist or not sl or not sr:
            return "Cross"
        bcx, bcy = (sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2
        wa = math.atan2(wrist[1] - bcy, wrist[0] - bcx)
        sa = math.atan2(sr[1] - sl[1], sr[0] - sl[0])
        diff = abs(math.degrees(wa - sa)) % 360
        if diff > 180:
            diff = 360 - diff
        if diff < HOOK_MIN_ANGLE:
            return "Jab" if wrist == lw else "Cross"
        return "Hook"

    def build_dataframe(self):
        self.df = pd.DataFrame(self._frames)
        self.summary = {
            "fighter_a_total_strikes": len(self._strikes_a),
            "fighter_b_total_strikes": len(self._strikes_b),
            "fighter_a_landed": sum(1 for s in self._strikes_a if s.get("landed")),
            "fighter_b_landed": sum(1 for s in self._strikes_b if s.get("landed")),
            "fighter_a_missed": len(self._strikes_a) - sum(1 for s in self._strikes_a if s.get("landed")),
            "fighter_b_missed": len(self._strikes_b) - sum(1 for s in self._strikes_b if s.get("landed")),
            "fighter_a_strike_types": self._ctypes(self._strikes_a),
            "fighter_b_strike_types": self._ctypes(self._strikes_b),
            "max_speed": self._max_speed(),
            "fight_duration_sec": round(self.df["timestamp_sec"].iloc[-1], 1) if len(self.df) else 0.0,
        }
        return self.df

    @staticmethod
    def _ctypes(strikes):
        c = {"Jab": 0, "Cross": 0, "Hook": 0}
        for s in strikes:
            t = s.get("type", "Cross")
            if t in c:
                c[t] += 1
        return c

    def _max_speed(self):
        a = self.df["fighter_a_wrist_speed"].dropna().tolist() if self.df is not None else []
        b = self.df["fighter_b_wrist_speed"].dropna().tolist() if self.df is not None else []
        all_s = a + b
        return round(max(all_s), 2) if all_s else 0.0

    def render_punches_bar(self):
        s = self.summary
        fig = go.Figure(data=[
            go.Bar(name=FIGHTER_A_NAME, x=["Thrown", "Landed", "Missed"],
                   y=[s["fighter_a_total_strikes"], s["fighter_a_landed"], s["fighter_a_missed"]],
                   marker_color=FIGHTER_A_CHART_COLOR, textposition="auto"),
            go.Bar(name=FIGHTER_B_NAME, x=["Thrown", "Landed", "Missed"],
                   y=[s["fighter_b_total_strikes"], s["fighter_b_landed"], s["fighter_b_missed"]],
                   marker_color=FIGHTER_B_CHART_COLOR, textposition="auto"),
        ])
        fig.update_layout(title="Punches Thrown vs Landed", barmode="group", template=PLOTLY_TEMPLATE, height=CHART_HEIGHT)
        return fig

    def render_strike_types_pie(self):
        s = self.summary
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                            subplot_titles=[FIGHTER_A_NAME, FIGHTER_B_NAME])
        labels = ["Jab", "Cross", "Hook"]
        va = [s["fighter_a_strike_types"].get(l, 0) for l in labels]
        vb = [s["fighter_b_strike_types"].get(l, 0) for l in labels]
        fig.add_trace(go.Pie(labels=labels, values=va, hole=0.3), row=1, col=1)
        fig.add_trace(go.Pie(labels=labels, values=vb, hole=0.3), row=1, col=2)
        fig.update_layout(title="Strike Type Breakdown", template=PLOTLY_TEMPLATE, height=CHART_HEIGHT)
        return fig

    def render_max_speed_gauge(self):
        ms = self.summary.get("max_speed", 0)
        fig = go.Figure(go.Indicator(mode="gauge+number", value=ms, title={"text": "Max Strike Speed (m/s)"},
            gauge={"axis": {"range": [0, max(ms * 1.3, 10)]}, "bar": {"color": "#F59E0B"},
                   "steps": [{"range": [0, 3], "color": "#22C55E"}, {"range": [3, 6], "color": "#EAB308"}, {"range": [6, 10], "color": "#EF4444"}]}))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=CHART_HEIGHT)
        return fig

    def render_velocity_timeline(self):
        if self.df is None or len(self.df) == 0:
            return go.Figure().update_layout(title="No data", template=PLOTLY_TEMPLATE)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df["timestamp_sec"], y=self.df["fighter_a_wrist_speed"],
                                 mode="lines", name=FIGHTER_A_NAME, line=dict(color=FIGHTER_A_CHART_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=self.df["timestamp_sec"], y=self.df["fighter_b_wrist_speed"],
                                 mode="lines", name=FIGHTER_B_NAME, line=dict(color=FIGHTER_B_CHART_COLOR, width=2)))
        fig.update_layout(title="Movement Velocity Over Time", xaxis_title="Time (s)", yaxis_title="Wrist Speed (m/s)",
                          template=PLOTLY_TEMPLATE, height=CHART_HEIGHT)
        return fig

    def render_guard_angle_timeline(self):
        if self.df is None or len(self.df) == 0:
            return go.Figure().update_layout(title="No data", template=PLOTLY_TEMPLATE)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df["timestamp_sec"], y=self.df["fighter_a_guard_angle"],
                                 mode="lines", name=FIGHTER_A_NAME, line=dict(color=FIGHTER_A_CHART_COLOR, width=2), fill="tozeroy"))
        fig.add_trace(go.Scatter(x=self.df["timestamp_sec"], y=self.df["fighter_b_guard_angle"],
                                 mode="lines", name=FIGHTER_B_NAME, line=dict(color=FIGHTER_B_CHART_COLOR, width=2), fill="tozeroy"))
        fig.update_layout(title="Guard Angle Over Time", xaxis_title="Time (s)", yaxis_title="Elbow Angle (degrees)",
                          template=PLOTLY_TEMPLATE, height=CHART_HEIGHT)
        return fig
'''

with open(os.path.join(PROJECT_DIR, "analytics.py"), "w") as f:
    f.write(analytics_code)
print("Written: analytics.py")

# ---------- app.py ----------
app_code = r'''
"""Combat Sports Video Analytics - Main Streamlit Application."""

import os, sys, tempfile
import cv2, numpy as np
import streamlit as st

from config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, FIGHTER_A_NAME, FIGHTER_B_NAME
from cv_pipeline import FighterPoseProcessor
from analytics import FightAnalytics

st.set_page_config(page_title="Combat Sports Analytics", page_icon="boxing_glove", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stMetric { background: #1e1e2e; border-radius: 8px; padding: 12px; }
    [data-testid="stSidebar"] { background-color: #11111b; }
    h1 { color: #f5a623 !important; }
    h3 { color: #cdd6f4 !important; }
</style>
""", unsafe_allow_html=True)

if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False
if "analytics" not in st.session_state:
    st.session_state.analytics = None
if "output_path" not in st.session_state:
    st.session_state.output_path = None
if "fa_name" not in st.session_state:
    st.session_state.fa_name = FIGHTER_A_NAME
if "fb_name" not in st.session_state:
    st.session_state.fb_name = FIGHTER_B_NAME

with st.sidebar:
    st.header("Session Management")
    st.session_state.fa_name = st.text_input("Fighter A Name", value=st.session_state.fa_name)
    st.session_state.fb_name = st.text_input("Fighter B Name", value=st.session_state.fb_name)
    if st.button("Reset Session"):
        for k in ("uploaded_path", "output_path", "processing_done", "analytics"):
            st.session_state[k] = None if k != "processing_done" else False
        st.rerun()
    st.divider()
    st.header("Upload Video")
    uploaded_file = st.file_uploader("Choose a fight video", type=["mp4", "avi", "mov", "mkv"])
    st.divider()
    st.header("Analysis Settings")
    det_conf = st.slider("Detection Confidence", 0.1, 1.0, MIN_DETECTION_CONFIDENCE, 0.05)
    trk_conf = st.slider("Tracking Confidence", 0.1, 1.0, MIN_TRACKING_CONFIDENCE, 0.05)
    vel_thresh = st.slider("Strike Velocity Threshold", 0.5, 5.0, 1.2, 0.1)


def process_video(uploaded_file, det_conf, trk_conf, vel_thresh):
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_in.write(uploaded_file.read())
    tmp_in.close()
    cap = cv2.VideoCapture(tmp_in.name)
    if not cap.isOpened():
        st.error("Failed to open video.")
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    writer = cv2.VideoWriter(tmp_out.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    processor = FighterPoseProcessor(min_detection_confidence=det_conf, min_tracking_confidence=trk_conf)
    analytics = FightAnalytics(fps=fps)
    import config
    config.VELOCITY_THRESHOLD = vel_thresh / 100.0
    prog = st.progress(0, text="Starting analysis...")
    fc = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ann, mets = processor.process_frame(frame, timestamp_ms=int(fc * 1000 / fps))
            writer.write(ann)
            analytics.record_frame(fc, mets)
            fc += 1
            if fc % 5 == 0 or fc == total:
                prog.progress(min(fc / total, 1.0), text=f"Frame {fc}/{total}")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        cap.release()
        writer.release()
        processor.close()
    prog.progress(1.0, text="Done!")
    analytics.build_dataframe()
    return tmp_out.name, analytics


st.markdown("# Combat Sports Video Analytics")
st.markdown("Upload a fight video, process it with real-time pose tracking, and explore fight analytics.")

if uploaded_file and not st.session_state.processing_done:
    st.info("Video uploaded. Click below to start analysis.")

if uploaded_file and st.button("Process Video", type="primary", use_container_width=True):
    with st.spinner("Initializing pose detection model..."):
        out, an = process_video(uploaded_file, det_conf, trk_conf, vel_thresh)
    if out and an:
        st.session_state.output_path = out
        st.session_state.analytics = an
        st.session_state.processing_done = True
        st.success("Analysis complete! Scroll down for results.")
        st.rerun()

if st.session_state.processing_done and st.session_state.analytics:
    an = st.session_state.analytics
    fa, fb = st.session_state.fa_name, st.session_state.fb_name
    s = an.summary
    st.divider()
    st.header("Processed Video")
    if st.session_state.output_path and os.path.exists(st.session_state.output_path):
        st.video(st.session_state.output_path)
    st.divider()
    st.header("Summary Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fight Duration", f"{s['fight_duration_sec']:.1f}s")
    c2.metric(f"{fa} Strikes", s["fighter_a_total_strikes"], f"{s['fighter_a_landed']} landed")
    c3.metric(f"{fb} Strikes", s["fighter_b_total_strikes"], f"{s['fighter_b_landed']} landed")
    c4.metric("Max Speed", f"{s['max_speed']:.2f} m/s")
    st.divider()
    st.header("Full Fight Statistics")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(an.render_punches_bar(), use_container_width=True)
    with col2:
        st.plotly_chart(an.render_max_speed_gauge(), use_container_width=True)
    st.plotly_chart(an.render_strike_types_pie(), use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(an.render_velocity_timeline(), use_container_width=True)
    with col4:
        st.plotly_chart(an.render_guard_angle_timeline(), use_container_width=True)
    with st.expander("View Raw Frame-Level Data"):
        if an.df is not None:
            cols = [c for c in ["frame", "timestamp_sec",
                     "fighter_a_detected", "fighter_a_wrist_speed", "fighter_a_guard_angle", "fighter_a_strike_detected",
                     "fighter_b_detected", "fighter_b_wrist_speed", "fighter_b_guard_angle", "fighter_b_strike_detected"]
                    if c in an.df.columns]
            st.dataframe(an.df[cols], use_container_width=True)
else:
    st.info("Upload a fight video in the sidebar to begin analysis.")
'''

with open(os.path.join(PROJECT_DIR, "app.py"), "w") as f:
    f.write(app_code)
print("Written: app.py")

print("\nAll source files created successfully!")

# ============================================================================
# CELL 4: Start Streamlit via ngrok tunnel
# ============================================================================

from pyngrok import ngrok
import threading, time, os

# Kill any existing ngrok tunnels
ngrok.kill()

# Start Streamlit in a background thread
def run_streamlit():
    os.system(f"streamlit run {PROJECT_DIR}/app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false")

thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

# Wait for Streamlit to start
time.sleep(5)

# Create ngrok tunnel
public_url = ngrok.connect(8501)
print(f"\n{'='*60}")
print(f"  Streamlit is running!")
print(f"  Public URL: {public_url}")
print(f"{'='*60}")
print(f"\nOpen the URL above in a new tab to use the app.")
