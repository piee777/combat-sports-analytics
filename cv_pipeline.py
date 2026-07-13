"""
Combat Sports Video Analytics - Computer Vision Pipeline

Handles frame-by-frame pose detection using the MediaPipe Tasks API
(PoseLandmarker) with native multi-person support (num_poses=2).
Renders skeleton overlays, bounding boxes, and real-time metric text
(wrist velocity, guard angle) on each annotated frame.
"""

import os
import math
import cv2
import numpy as np
import mediapipe as mp

from config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    FIGHTER_A_COLOR,
    FIGHTER_B_COLOR,
    FIGHTER_A_NAME,
    FIGHTER_B_NAME,
    SKELETON_THICKNESS,
    LANDMARK_RADIUS,
    BOUNDING_BOX_THICKNESS,
    LANDMARKS,
    LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA,
    VELOCITY_SMOOTH_WINDOW,
    PROCESS_MAX_WIDTH,
    SKIP_FRAMES,
)

# Default path to the PoseLandmarker .task model file (lite variant)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pose_landmarker.task"
)

# ---------------------------------------------------------------------------
# Geometry Helpers
# ---------------------------------------------------------------------------

def calculate_angle(point_a, point_b, point_c):
    """
    Angle (degrees) at *point_b* formed by segments a -> b -> c.
    """
    a = np.array(point_a, dtype=np.float64)
    b = np.array(point_b, dtype=np.float64)
    c = np.array(point_c, dtype=np.float64)
    ba, bc = a - b, c - b
    cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))


def calculate_distance(point_a, point_b):
    """Euclidean distance between two (x, y) points."""
    return float(np.linalg.norm(
        np.array(point_a, dtype=np.float64) -
        np.array(point_b, dtype=np.float64)
    ))


# ---------------------------------------------------------------------------
# Landmark Coordinate Extraction
# ---------------------------------------------------------------------------

def get_landmark_px(landmarks, idx, width, height):
    """Return (x, y) pixel coords for landmark *idx*, or None if invisible."""
    lm = landmarks[idx]
    if lm.visibility < 0.3:
        return None
    return (lm.x * width, lm.y * height)


def get_all_landmarks_px(landmarks, width, height):
    """Convert all 33 landmarks to pixel coords (list of tuples or None)."""
    return [get_landmark_px(landmarks, i, width, height) for i in range(33)]


# ---------------------------------------------------------------------------
# Fighter Pose Processor (MediaPipe Tasks API - PoseLandmarker)
# ---------------------------------------------------------------------------

class FighterPoseProcessor:
    """
    Detects poses for up to two fighters in a single frame using the
    MediaPipe PoseLandmarker Tasks API with ``num_poses=2``.

    Detected poses are assigned to Fighter A and Fighter B using
    nearest-centroid tracking across frames, so identities remain
    consistent even when fighters cross or move quickly.

    Per-fighter state (previous wrist positions, velocity history) is
    maintained across frames for velocity and strike tracking.
    """

    def __init__(
        self,
        model_path=None,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        num_poses=2,
    ):
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

        # Connections for drawing the skeleton
        self._connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
        self._drawing_spec = mp.tasks.vision.drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=SKELETON_THICKNESS,
            circle_radius=LANDMARK_RADIUS,
        )

        # Per-fighter state
        self._prev_wrist_a = None
        self._prev_wrist_b = None
        self._velocity_history_a: list[float] = []
        self._velocity_history_b: list[float] = []
        self._prev_centroid_a: tuple[float, float] | None = None
        self._prev_centroid_b: tuple[float, float] | None = None

        # Frame skipping state
        self._frame_count = 0
        self._last_annotated = None
        self._last_metrics = None

    # ------------------------------------------------------------------
    # Internal drawing helpers
    # ------------------------------------------------------------------

    def _assign_fighters(self, all_poses_px, frame_width):
        """
        Assign poses to Fighter A / Fighter B using nearest-centroid
        tracking across frames. On the first frame, falls back to
        left/right assignment. On subsequent frames, each pose is
        matched to the closest previous centroid, maintaining identity
        even when fighters cross.
        """
        if not all_poses_px:
            self._prev_centroid_a = None
            self._prev_centroid_b = None
            return None, None

        # Compute centroid for each detected pose
        centroids = []
        for pose in all_poses_px:
            valid = [p for p in pose if p is not None]
            if valid:
                cx = float(np.mean([p[0] for p in valid]))
                cy = float(np.mean([p[1] for p in valid]))
            else:
                cx, cy = float("inf"), float("inf")
            centroids.append((cx, cy, pose))

        if len(centroids) == 1:
            cx, cy, pose = centroids[0]
            if self._prev_centroid_a is not None and self._prev_centroid_b is not None:
                da = np.hypot(cx - self._prev_centroid_a[0], cy - self._prev_centroid_a[1])
                db = np.hypot(cx - self._prev_centroid_b[0], cy - self._prev_centroid_b[1])
                if db < da:
                    self._prev_centroid_b = (cx, cy)
                    return None, pose
                self._prev_centroid_a = (cx, cy)
                return pose, None
            # First frame: assign by left/right
            self._prev_centroid_a = (cx, cy)
            return (pose, None) if cx < frame_width / 2 else (None, pose)

        # Two poses: match each to closest previous centroid
        c0 = (centroids[0][0], centroids[0][1])
        c1 = (centroids[1][0], centroids[1][1])

        if self._prev_centroid_a is None or self._prev_centroid_b is None:
            # First frame: leftmost = A, rightmost = B
            if c0[0] < c1[0]:
                self._prev_centroid_a = c0
                self._prev_centroid_b = c1
                return centroids[0][2], centroids[1][2]
            else:
                self._prev_centroid_a = c1
                self._prev_centroid_b = c0
                return centroids[1][2], centroids[0][2]

        da0 = np.hypot(c0[0] - self._prev_centroid_a[0], c0[1] - self._prev_centroid_a[1])
        da1 = np.hypot(c1[0] - self._prev_centroid_a[0], c1[1] - self._prev_centroid_a[1])

        if da0 <= da1:
            self._prev_centroid_a = c0
            self._prev_centroid_b = c1
            return centroids[0][2], centroids[1][2]
        else:
            self._prev_centroid_a = c1
            self._prev_centroid_b = c0
            return centroids[1][2], centroids[0][2]

    def _draw_skeleton_on_frame(self, frame, landmarks_px, color):
        """Draw connections and landmarks on *frame* using *color* (BGR)."""
        for conn in self._connections:
            pt_a = landmarks_px[conn.start]
            pt_b = landmarks_px[conn.end]
            if pt_a is not None and pt_b is not None:
                cv2.line(frame,
                         (int(pt_a[0]), int(pt_a[1])),
                         (int(pt_b[0]), int(pt_b[1])),
                         color, SKELETON_THICKNESS)
        for pt in landmarks_px:
            if pt is not None:
                cv2.circle(frame, (int(pt[0]), int(pt[1])),
                           LANDMARK_RADIUS, color, -1)

    @staticmethod
    def _draw_bounding_box(frame, landmarks_px, color, label):
        """Bounding box with label."""
        valid = [p for p in landmarks_px if p is not None]
        if not valid:
            return
        xs, ys = [p[0] for p in valid], [p[1] for p in valid]
        h, w = frame.shape[:2]
        x1 = max(0, int(min(xs)) - 10)
        y1 = max(0, int(min(ys)) - 10)
        x2 = min(w - 1, int(max(xs)) + 10)
        y2 = min(h - 1, int(max(ys)) + 10)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOUNDING_BOX_THICKNESS)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    @staticmethod
    def _overlay_metrics(frame, label, color, y_offset,
                         guard_angle, wrist_speed):
        """Real-time metric text overlay."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (4, y_offset - 10), (155, y_offset + 22),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, label, (6, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.putText(frame, f"S:{wrist_speed:.1f} G:{guard_angle:.0f}", (6, y_offset + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def process_frame(self, frame, timestamp_ms=0):
        """
        Run the full pipeline on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).
        timestamp_ms : int
            Frame timestamp in milliseconds (for video-mode tracking).

        Returns
        -------
        annotated_frame : np.ndarray
        metrics : dict
            Per-fighter metrics (see module docstring).
        """
        self._frame_count += 1
        orig_h, orig_w = frame.shape[:2]

        # Downscale for faster processing on low-end CPUs
        scale = 1.0
        if orig_w > PROCESS_MAX_WIDTH:
            scale = PROCESS_MAX_WIDTH / orig_w
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)

        # Skip frames for performance: reuse last result
        if SKIP_FRAMES > 1 and self._frame_count % SKIP_FRAMES != 0:
            if self._last_annotated is not None:
                # Scale back up for display
                display = self._last_annotated
                if scale < 1.0:
                    display = cv2.resize(display, (orig_w, orig_h),
                                         interpolation=cv2.INTER_LINEAR)
                return display, self._last_metrics
            # First frame can't be skipped
            scale = 1.0
            frame = cv2.resize(frame, None, fx=1/scale, fy=1/scale) if scale != 1.0 else frame

        h, w = frame.shape[:2]
        annotated = frame.copy()

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect up to 2 poses
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # Convert detected landmarks to pixel coordinates
        all_poses_px = []
        for pose_landmarks in result.pose_landmarks:
            px = get_all_landmarks_px(pose_landmarks, w, h)
            all_poses_px.append(px)

        # Assign to Fighter A / Fighter B (nearest-centroid tracking)
        px_a, px_b = self._assign_fighters(all_poses_px, w)

        # ---- Per-fighter metrics ----
        def _compute_fighter_metrics(px, prev_wrist, vel_history):
            guard = None
            speed = None
            lw = rw = bb = None
            if px is None:
                return guard, speed, lw, rw, bb, prev_wrist, vel_history

            lw = px[LW]
            rw = px[RW]

            # Guard angle (dominant elbow)
            if px[LS] and px[LE] and px[LW]:
                guard = calculate_angle(px[LS], px[LE], px[LW])
            elif px[RS] and px[RE] and px[RW]:
                guard = calculate_angle(px[RS], px[RE], px[RW])

            # Dominant wrist (most forward / prominent)
            dominant = lw if (lw and not rw) or (
                lw and rw and lw[0] > rw[0]) else rw

            if dominant and prev_wrist is not None:
                dist = calculate_distance(dominant, prev_wrist)
                speed = dist * 100.0
            prev_wrist = dominant

            if speed is not None:
                vel_history.append(speed)
                if len(vel_history) > VELOCITY_SMOOTH_WINDOW:
                    vel_history = vel_history[-VELOCITY_SMOOTH_WINDOW:]
                speed = float(np.mean(vel_history))

            valid = [p for p in px if p is not None]
            if valid:
                xs = [p[0] for p in valid]
                ys = [p[1] for p in valid]
                bb = (min(xs), min(ys), max(xs), max(ys))

            return guard, speed, lw, rw, bb, prev_wrist, vel_history

        guard_a, speed_a, lw_a, rw_a, bb_a, self._prev_wrist_a, self._velocity_history_a = \
            _compute_fighter_metrics(px_a, self._prev_wrist_a, self._velocity_history_a)
        guard_b, speed_b, lw_b, rw_b, bb_b, self._prev_wrist_b, self._velocity_history_b = \
            _compute_fighter_metrics(px_b, self._prev_wrist_b, self._velocity_history_b)

        # ---- Draw overlays ----
        if px_a:
            self._draw_skeleton_on_frame(annotated, px_a, FIGHTER_A_COLOR)
            self._draw_bounding_box(annotated, px_a, FIGHTER_A_COLOR, FIGHTER_A_NAME)
        if px_b:
            self._draw_skeleton_on_frame(annotated, px_b, FIGHTER_B_COLOR)
            self._draw_bounding_box(annotated, px_b, FIGHTER_B_COLOR, FIGHTER_B_NAME)

        ga = guard_a if guard_a is not None else 0.0
        sa = speed_a if speed_a is not None else 0.0
        gb = guard_b if guard_b is not None else 0.0
        sb = speed_b if speed_b is not None else 0.0

        self._overlay_metrics(annotated, FIGHTER_A_NAME, FIGHTER_A_COLOR, 25, ga, sa)
        self._overlay_metrics(annotated, FIGHTER_B_NAME, FIGHTER_B_COLOR, 75, gb, sb)

        metrics = {
            "fighter_a": {
                "detected": px_a is not None,
                "landmarks_px": px_a,
                "guard_angle": guard_a,
                "wrist_speed": speed_a,
                "left_wrist_px": lw_a,
                "right_wrist_px": rw_a,
                "bounding_box": bb_a,
            },
            "fighter_b": {
                "detected": px_b is not None,
                "landmarks_px": px_b,
                "guard_angle": guard_b,
                "wrist_speed": speed_b,
                "left_wrist_px": lw_b,
                "right_wrist_px": rw_b,
                "bounding_box": bb_b,
            },
        }

        # Scale back up to original resolution for display
        if scale < 1.0:
            annotated = cv2.resize(annotated, (orig_w, orig_h),
                                   interpolation=cv2.INTER_LINEAR)

        # Cache for frame skipping
        self._last_annotated = annotated
        self._last_metrics = metrics

        return annotated, metrics

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
