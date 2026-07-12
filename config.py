"""
Combat Sports Video Analytics - Configuration Constants

Central configuration for MediaPipe pose detection parameters,
fighter visual identity, strike detection thresholds, and
MediaPipe Pose landmark indices used throughout the application.
"""

import cv2

# ---------------------------------------------------------------------------
# MediaPipe Pose Detection Parameters
# ---------------------------------------------------------------------------
# MODEL_COMPLEXITY: 0 = lite (fastest), 1 = full, 2 = heavy (slowest)
# On low-end CPUs (Pentium, Celeron, Core 2), use 0
MODEL_COMPLEXITY = 0
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ---------------------------------------------------------------------------
# Performance limits (critical for low-end PCs)
# ---------------------------------------------------------------------------
# Max width to process frames at (downscale before MediaPipe)
PROCESS_MAX_WIDTH = 480
# Skip every Nth frame for pose detection (1 = every frame, 2 = every other)
SKIP_FRAMES = 2

# ---------------------------------------------------------------------------
# Fighter Visual Identity (BGR color space for OpenCV)
# ---------------------------------------------------------------------------
FIGHTER_A_COLOR = (255, 100, 50)    # Blue for Fighter A
FIGHTER_B_COLOR = (50, 50, 255)     # Red for Fighter B
FIGHTER_A_NAME = "Fighter A"
FIGHTER_B_NAME = "Fighter B"

SKELETON_THICKNESS = 2
LANDMARK_RADIUS = 3
BOUNDING_BOX_THICKNESS = 2

# ---------------------------------------------------------------------------
# MediaPipe Pose Landmark Index Mapping
# ---------------------------------------------------------------------------
LANDMARKS = {
    "NOSE": 0,
    "LEFT_EYE_INNER": 1,
    "LEFT_EYE": 2,
    "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4,
    "RIGHT_EYE": 5,
    "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,
    "MOUTH_LEFT": 9,
    "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17,
    "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19,
    "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21,
    "RIGHT_THUMB": 22,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,
    "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

# Shorthand aliases for frequently used landmarks
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

# ---------------------------------------------------------------------------
# Strike Detection Thresholds
# ---------------------------------------------------------------------------
# Minimum normalized velocity (pixels-per-frame) to consider a wrist movement
VELOCITY_THRESHOLD = 0.012

# Minimum elbow extension angle (degrees) to count as a strike
ARM_EXTENSION_THRESHOLD = 140.0

# Cooldown in frames between consecutive strike detections for the same fighter
PUNCH_COOLDOWN = 6

# Velocity smoothing window (frames) for noise reduction
VELOCITY_SMOOTH_WINDOW = 3

# ---------------------------------------------------------------------------
# Strike Classification (degrees)
# ---------------------------------------------------------------------------
# Angle of wrist trajectory relative to shoulder line to distinguish strike types
JAB_MAX_ANGLE = 30.0        # Straight-line trajectory (lead hand)
CROSS_MAX_ANGLE = 30.0      # Straight-line trajectory (rear hand)
HOOK_MIN_ANGLE = 45.0       # Curved / angular trajectory

# ---------------------------------------------------------------------------
# Visualization / Chart Theme
# ---------------------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_dark"
CHART_HEIGHT = 350
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "staticPlot": False,
}

FIGHTER_A_CHART_COLOR = "#3B82F6"   # Blue (hex for Plotly)
FIGHTER_B_CHART_COLOR = "#EF4444"   # Red  (hex for Plotly)

# ---------------------------------------------------------------------------
# Annotation Toggles (set dynamically by Gradio UI)
# ---------------------------------------------------------------------------
SHOW_SKELETON = True
SHOW_BBOX = True
SHOW_METRICS = True
