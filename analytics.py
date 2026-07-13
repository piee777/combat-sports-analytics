"""
Combat Sports Video Analytics - Metrics & Visualizations

Collects per-frame metrics from the CV pipeline, derives fight
statistics (joint angles, strike velocity, punch counting, strike
classification), and renders interactive Plotly charts for the
Streamlit dashboard.
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    LANDMARKS,
    LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA,
    VELOCITY_THRESHOLD,
    ARM_EXTENSION_THRESHOLD,
    PUNCH_COOLDOWN,
    PLOTLY_TEMPLATE,
    CHART_HEIGHT,
    FIGHTER_A_CHART_COLOR,
    FIGHTER_B_CHART_COLOR,
    FIGHTER_A_NAME,
    FIGHTER_B_NAME,
    HOOK_MIN_ANGLE,
)


# ---------------------------------------------------------------------------
# Geometry Helpers (mirrored from cv_pipeline for analytics-side use)
# ---------------------------------------------------------------------------

def calculate_angle(point_a, point_b, point_c):
    """Angle in degrees at point_b formed by segments a→b→c."""
    a = np.array(point_a, dtype=np.float64)
    b = np.array(point_b, dtype=np.float64)
    c = np.array(point_c, dtype=np.float64)
    ba = a - b
    bc = c - b
    cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))


def calculate_distance(point_a, point_b):
    """Euclidean distance between two (x, y) tuples."""
    return float(np.linalg.norm(
        np.array(point_a, dtype=np.float64) -
        np.array(point_b, dtype=np.float64)
    ))


# ---------------------------------------------------------------------------
# FightAnalytics — Data Collector
# ---------------------------------------------------------------------------

class FightAnalytics:
    """
    Accumulates per-frame metrics supplied by FighterPoseProcessor,
    derives fight-level statistics, and renders Plotly visualizations.
    """

    def __init__(self, fps):
        self.fps = fps
        self._frames = []
        self.df = None

        # Accumulated strike events
        self._strikes_a = []
        self._strikes_b = []

        # Punch cooldown counters
        self._cooldown_a = 0
        self._cooldown_b = 0

    # ------------------------------------------------------------------
    # Per-frame data ingestion
    # ------------------------------------------------------------------

    def record_frame(self, frame_idx, metrics):
        """
        Store metrics for a single frame.

        Parameters
        ----------
        frame_idx : int
            Zero-based frame index.
        metrics : dict
            Output from ``FighterPoseProcessor.process_frame``.
        """
        timestamp = frame_idx / self.fps if self.fps else 0.0

        entry = {
            "frame": frame_idx,
            "timestamp_sec": round(timestamp, 3),
        }

        for fighter_key in ("fighter_a", "fighter_b"):
            fm = metrics.get(fighter_key, {})
            prefix = fighter_key

            detected = fm.get("detected", False)
            entry[f"{prefix}_detected"] = detected
            entry[f"{prefix}_guard_angle"] = fm.get("guard_angle")
            entry[f"{prefix}_wrist_speed"] = fm.get("wrist_speed")

            # Elbow angles
            lm_px = fm.get("landmarks_px")
            if lm_px and detected:
                # Left elbow angle
                if lm_px[LS] and lm_px[LE] and lm_px[LW]:
                    entry[f"{prefix}_left_elbow_angle"] = calculate_angle(
                        lm_px[LS], lm_px[LE], lm_px[LW])
                else:
                    entry[f"{prefix}_left_elbow_angle"] = None

                # Right elbow angle
                if lm_px[RS] and lm_px[RE] and lm_px[RW]:
                    entry[f"{prefix}_right_elbow_angle"] = calculate_angle(
                        lm_px[RS], lm_px[RE], lm_px[RW])
                else:
                    entry[f"{prefix}_right_elbow_angle"] = None

                # Left knee angle
                if lm_px[LH] and lm_px[LK] and lm_px[LA]:
                    entry[f"{prefix}_left_knee_angle"] = calculate_angle(
                        lm_px[LH], lm_px[LK], lm_px[LA])
                else:
                    entry[f"{prefix}_left_knee_angle"] = None

                # Right knee angle
                if lm_px[RH] and lm_px[RK] and lm_px[RA]:
                    entry[f"{prefix}_right_knee_angle"] = calculate_angle(
                        lm_px[RH], lm_px[RK], lm_px[RA])
                else:
                    entry[f"{prefix}_right_knee_angle"] = None

                # Bounding box (for "landed" heuristic)
                bb = fm.get("bounding_box")
                entry[f"{prefix}_bb"] = bb
            else:
                for suffix in ("left_elbow_angle", "right_elbow_angle",
                               "left_knee_angle", "right_knee_angle", "bb"):
                    entry[f"{prefix}_{suffix}"] = None

            # ---- Strike detection ----
            opponent_key = "fighter_b" if fighter_key == "fighter_a" else "fighter_a"
            opponent_fm = metrics.get(opponent_key, {})
            strike = self._detect_strike(fighter_key, entry, lm_px, fm, opponent_fm)
            entry[f"{prefix}_strike_detected"] = strike is not None
            if strike:
                self._strikes_a.append(strike) if fighter_key == "fighter_a" else self._strikes_b.append(strike)

        self._frames.append(entry)

    # ------------------------------------------------------------------
    # Strike Detection (rule-based)
    # ------------------------------------------------------------------

    def _detect_strike(self, fighter_key, frame_entry, lm_px, fm, opponent_fm):
        """
        Rule-based strike detection:
          1. Wrist velocity above threshold
          2. Elbow extension angle above threshold (arm extending)
          3. Cooldown period elapsed
          4. Wrist moving toward opponent
        """
        is_a = fighter_key == "fighter_a"
        cooldown_attr = "_cooldown_a" if is_a else "_cooldown_b"
        current_cooldown = getattr(self, cooldown_attr)

        if current_cooldown > 0:
            setattr(self, cooldown_attr, current_cooldown - 1)
            return None

        speed = fm.get("wrist_speed")
        if speed is None or speed < VELOCITY_THRESHOLD * 100:
            return None

        # Check arm extension via dominant elbow angle
        dominant_angle = None
        if lm_px:
            lw = fm.get("left_wrist_px")
            rw = fm.get("right_wrist_px")
            if lw and lm_px[LS] and lm_px[LE] and lm_px[LW]:
                dominant_angle = calculate_angle(lm_px[LS], lm_px[LE], lm_px[LW])
            elif rw and lm_px[RS] and lm_px[RE] and lm_px[RW]:
                dominant_angle = calculate_angle(lm_px[RS], lm_px[RE], lm_px[RW])

        if dominant_angle is None or dominant_angle < ARM_EXTENSION_THRESHOLD:
            return None

        strike_type = self._classify_strike(fm, lm_px)

        landed = self._check_landed(fm, opponent_fm)

        strike = {
            "fighter": fighter_key,
            "frame": frame_entry["frame"],
            "timestamp": frame_entry["timestamp_sec"],
            "velocity": speed,
            "elbow_angle": dominant_angle,
            "type": strike_type,
            "landed": landed,
        }

        setattr(self, cooldown_attr, PUNCH_COOLDOWN)
        return strike

    @staticmethod
    def _classify_strike(fm, lm_px):
        """
        Classify a strike as Jab, Cross, or Hook based on the
        trajectory of the dominant wrist relative to the shoulder line.
        """
        if lm_px is None:
            return "Cross"

        # Determine striking side
        lw = fm.get("left_wrist_px")
        rw = fm.get("right_wrist_px")
        wrist = lw if (lw and not rw) or (lw and rw and lw[0] > rw[0]) else rw

        shoulder_l = lm_px[LS] if lm_px[LS] else None
        shoulder_r = lm_px[RS] if lm_px[RS] else None

        if not wrist or not shoulder_l or not shoulder_r:
            return "Cross"

        # Shoulder line angle
        shoulder_angle = math.atan2(
            shoulder_r[1] - shoulder_l[1],
            shoulder_r[0] - shoulder_l[0]
        )

        # Wrist direction from the body center
        body_center_x = (shoulder_l[0] + shoulder_r[0]) / 2.0
        body_center_y = (shoulder_l[1] + shoulder_r[1]) / 2.0
        wrist_angle = math.atan2(wrist[1] - body_center_y,
                                 wrist[0] - body_center_x)

        trajectory_diff = abs(math.degrees(wrist_angle - shoulder_angle)) % 360
        if trajectory_diff > 180:
            trajectory_diff = 360 - trajectory_diff

        if trajectory_diff < HOOK_MIN_ANGLE:
            # Straight — determine lead vs rear
            is_lead = (wrist == lw)  # simplified: left is lead
            return "Jab" if is_lead else "Cross"
        else:
            return "Hook"

    @staticmethod
    def _check_landed(fm, opponent_fm):
        """
        Heuristic: a punch is 'landed' if the striking wrist position is
        near or within the opponent's bounding box.
        """
        opp_bb = opponent_fm.get("bounding_box")
        if opp_bb is None:
            return False

        # Get striking wrist from current fighter
        lw = fm.get("left_wrist_px")
        rw = fm.get("right_wrist_px")
        wrist = lw if (lw and not rw) or (lw and rw and lw[0] > rw[0]) else rw
        if wrist is None:
            return False

        ox1, oy1, ox2, oy2 = opp_bb
        wx, wy = wrist

        # Wrist within or near opponent bounding box (with 30px margin)
        margin = 30
        if (ox1 - margin) <= wx <= (ox2 + margin) and (oy1 - margin) <= wy <= (oy2 + margin):
            # Also check wrist is moving forward toward opponent
            return True

        return False

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def build_dataframe(self):
        """
        Convert collected per-frame data into a Pandas DataFrame and
        compute aggregated strike statistics.
        """
        self.df = pd.DataFrame(self._frames)

        # ---- Compute landed strikes properly using the accumulated events ----
        total_a = len(self._strikes_a)
        total_b = len(self._strikes_b)
        landed_a = sum(1 for s in self._strikes_a if s.get("landed"))
        landed_b = sum(1 for s in self._strikes_b if s.get("landed"))

        self.summary = {
            "fighter_a_total_strikes": total_a,
            "fighter_b_total_strikes": total_b,
            "fighter_a_landed": landed_a,
            "fighter_b_landed": landed_b,
            "fighter_a_missed": total_a - landed_a,
            "fighter_b_missed": total_b - landed_b,
            "fighter_a_strike_types": self._count_strike_types(self._strikes_a),
            "fighter_b_strike_types": self._count_strike_types(self._strikes_b),
            "max_speed": self._compute_max_speed(),
            "fight_duration_sec": round(
                self.df["timestamp_sec"].iloc[-1], 1) if len(self.df) else 0.0,
        }

        return self.df

    @staticmethod
    def _count_strike_types(strikes):
        """Return dict of strike type → count."""
        counts = {"Jab": 0, "Cross": 0, "Hook": 0}
        for s in strikes:
            t = s.get("type", "Cross")
            if t in counts:
                counts[t] += 1
        return counts

    def _compute_max_speed(self):
        """Maximum wrist velocity across all frames and both fighters."""
        speeds_a = self.df["fighter_a_wrist_speed"].dropna().tolist() if self.df is not None else []
        speeds_b = self.df["fighter_b_wrist_speed"].dropna().tolist() if self.df is not None else []
        all_speeds = speeds_a + speeds_b
        return round(max(all_speeds), 2) if all_speeds else 0.0

    # ------------------------------------------------------------------
    # Plotly Visualizations
    # ------------------------------------------------------------------

    def render_punches_bar(self):
        """Bar chart: Punches Thrown vs Landed for each fighter."""
        s = self.summary
        fig = go.Figure(data=[
            go.Bar(
                name=FIGHTER_A_NAME,
                x=["Thrown", "Landed", "Missed"],
                y=[s["fighter_a_total_strikes"], s["fighter_a_landed"],
                   s["fighter_a_missed"]],
                marker_color=FIGHTER_A_CHART_COLOR,
                text=[s["fighter_a_total_strikes"], s["fighter_a_landed"],
                      s["fighter_a_missed"]],
                textposition="auto",
            ),
            go.Bar(
                name=FIGHTER_B_NAME,
                x=["Thrown", "Landed", "Missed"],
                y=[s["fighter_b_total_strikes"], s["fighter_b_landed"],
                   s["fighter_b_missed"]],
                marker_color=FIGHTER_B_CHART_COLOR,
                text=[s["fighter_b_total_strikes"], s["fighter_b_landed"],
                      s["fighter_b_missed"]],
                textposition="auto",
            ),
        ])
        fig.update_layout(
            title="Punches Thrown vs Landed",
            barmode="group",
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    def render_strike_types_pie(self):
        """Pie charts: strike type breakdown for each fighter (side by side)."""
        s = self.summary
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "pie"}]],
            subplot_titles=[FIGHTER_A_NAME, FIGHTER_B_NAME],
        )

        labels = ["Jab", "Cross", "Hook"]
        colors_a = ["#60A5FA", "#2563EB", "#1D4ED8"]
        colors_b = ["#F87171", "#DC2626", "#991B1B"]

        vals_a = [s["fighter_a_strike_types"].get(l, 0) for l in labels]
        vals_b = [s["fighter_b_strike_types"].get(l, 0) for l in labels]

        fig.add_trace(go.Pie(labels=labels, values=vals_a,
                             marker_colors=colors_a, hole=0.3), row=1, col=1)
        fig.add_trace(go.Pie(labels=labels, values=vals_b,
                             marker_colors=colors_b, hole=0.3), row=1, col=2)

        fig.update_layout(
            title="Strike Type Breakdown",
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            showlegend=True,
        )
        return fig

    def render_max_speed_gauge(self):
        """Gauge chart showing maximum strike speed."""
        max_spd = self.summary.get("max_speed", 0)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=max_spd,
            title={"text": "Max Strike Speed (m/s)"},
            gauge={
                "axis": {"range": [0, max(max_spd * 1.3, 10)]},
                "bar": {"color": "#F59E0B"},
                "steps": [
                    {"range": [0, 3], "color": "#22C55E"},
                    {"range": [3, 6], "color": "#EAB308"},
                    {"range": [6, 10], "color": "#EF4444"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": max_spd,
                },
            },
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=CHART_HEIGHT)
        return fig

    def render_velocity_timeline(self):
        """Line graph: wrist velocity over time for both fighters."""
        if self.df is None or len(self.df) == 0:
            return go.Figure().update_layout(title="No data", template=PLOTLY_TEMPLATE)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df["timestamp_sec"],
            y=self.df["fighter_a_wrist_speed"],
            mode="lines",
            name=FIGHTER_A_NAME,
            line=dict(color=FIGHTER_A_CHART_COLOR, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=self.df["timestamp_sec"],
            y=self.df["fighter_b_wrist_speed"],
            mode="lines",
            name=FIGHTER_B_NAME,
            line=dict(color=FIGHTER_B_CHART_COLOR, width=2),
        ))

        # Mark strike moments
        for strike in self._strikes_a:
            fig.add_vline(x=strike["timestamp"], line_dash="dot",
                          line_color=FIGHTER_A_CHART_COLOR, opacity=0.4)
        for strike in self._strikes_b:
            fig.add_vline(x=strike["timestamp"], line_dash="dot",
                          line_color=FIGHTER_B_CHART_COLOR, opacity=0.4)

        fig.update_layout(
            title="Movement Velocity Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Wrist Speed (m/s)",
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    def render_guard_angle_timeline(self):
        """Line graph: elbow guard angle over time for both fighters."""
        if self.df is None or len(self.df) == 0:
            return go.Figure().update_layout(title="No data", template=PLOTLY_TEMPLATE)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df["timestamp_sec"],
            y=self.df["fighter_a_guard_angle"],
            mode="lines",
            name=FIGHTER_A_NAME,
            line=dict(color=FIGHTER_A_CHART_COLOR, width=2),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.1)",
        ))
        fig.add_trace(go.Scatter(
            x=self.df["timestamp_sec"],
            y=self.df["fighter_b_guard_angle"],
            mode="lines",
            name=FIGHTER_B_NAME,
            line=dict(color=FIGHTER_B_CHART_COLOR, width=2),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.1)",
        ))

        fig.update_layout(
            title="Guard Angle Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Elbow Angle (degrees)",
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    # ------------------------------------------------------------------
    # New Visualizations
    # ------------------------------------------------------------------

    def render_strike_timeline(self):
        """Scatter plot: each strike as a point (time vs speed, colored by fighter/type)."""
        fig = go.Figure()

        type_symbols = {"Jab": "circle", "Cross": "diamond", "Hook": "square"}

        for strike in self._strikes_a:
            fig.add_trace(go.Scatter(
                x=[strike["timestamp"]],
                y=[strike["velocity"]],
                mode="markers",
                marker=dict(
                    color=FIGHTER_A_CHART_COLOR,
                    size=12,
                    symbol=type_symbols.get(strike["type"], "circle"),
                    line=dict(width=1, color="white"),
                ),
                name=f"{FIGHTER_A_NAME} ({strike['type']})",
                legendgroup="fighter_a",
                showlegend=False,
                hovertemplate=(
                    f"<b>{FIGHTER_A_NAME}</b><br>"
                    f"Type: {strike['type']}<br>"
                    f"Time: {strike['timestamp']:.2f}s<br>"
                    f"Speed: {strike['velocity']:.2f} m/s<br>"
                    f"Angle: {strike['elbow_angle']:.0f}°<br>"
                    f"Landed: {'Yes' if strike['landed'] else 'No'}"
                    "<extra></extra>"
                ),
            ))

        for strike in self._strikes_b:
            fig.add_trace(go.Scatter(
                x=[strike["timestamp"]],
                y=[strike["velocity"]],
                mode="markers",
                marker=dict(
                    color=FIGHTER_B_CHART_COLOR,
                    size=12,
                    symbol=type_symbols.get(strike["type"], "circle"),
                    line=dict(width=1, color="white"),
                ),
                name=f"{FIGHTER_B_NAME} ({strike['type']})",
                legendgroup="fighter_b",
                showlegend=False,
                hovertemplate=(
                    f"<b>{FIGHTER_B_NAME}</b><br>"
                    f"Type: {strike['type']}<br>"
                    f"Time: {strike['timestamp']:.2f}s<br>"
                    f"Speed: {strike['velocity']:.2f} m/s<br>"
                    f"Angle: {strike['elbow_angle']:.0f}°<br>"
                    f"Landed: {'Yes' if strike['landed'] else 'No'}"
                    "<extra></extra>"
                ),
            ))

        # Legend entries for fighter colors
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=FIGHTER_A_CHART_COLOR, size=10),
            name=FIGHTER_A_NAME, showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=FIGHTER_B_CHART_COLOR, size=10),
            name=FIGHTER_B_NAME, showlegend=True,
        ))

        fig.update_layout(
            title="Strike Event Timeline",
            xaxis_title="Time (s)",
            yaxis_title="Strike Speed (m/s)",
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    def render_cumulative_strikes(self):
        """Area chart: running total of strikes over time per fighter."""
        if not self._strikes_a and not self._strikes_b:
            return go.Figure().update_layout(
                title="No strikes detected", template=PLOTLY_TEMPLATE)

        fig = go.Figure()

        if self._strikes_a:
            times_a = [s["timestamp"] for s in self._strikes_a]
            cum_a = list(range(1, len(times_a) + 1))
            fig.add_trace(go.Scatter(
                x=times_a, y=cum_a, mode="lines+markers",
                name=FIGHTER_A_NAME,
                line=dict(color=FIGHTER_A_CHART_COLOR, width=2),
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.15)",
            ))

        if self._strikes_b:
            times_b = [s["timestamp"] for s in self._strikes_b]
            cum_b = list(range(1, len(times_b) + 1))
            fig.add_trace(go.Scatter(
                x=times_b, y=cum_b, mode="lines+markers",
                name=FIGHTER_B_NAME,
                line=dict(color=FIGHTER_B_CHART_COLOR, width=2),
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.15)",
            ))

        fig.update_layout(
            title="Cumulative Strikes Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Total Strikes",
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    def render_body_heatmap(self):
        """Heatmap: strike zone distribution (head/body/legs) per fighter."""
        zones = ["Head", "Body", "Legs"]
        zone_ranges = [
            (0.0, 0.35),    # Head: top 35% of frame
            (0.35, 0.65),   # Body: middle 30%
            (0.65, 1.0),    # Legs: bottom 35%
        ]

        counts_a = [0, 0, 0]
        counts_b = [0, 0, 0]

        for strike in self._strikes_a:
            # Approximate strike zone from wrist Y position at strike frame
            frame_data = self.df[self.df["frame"] == strike["frame"]]
            if len(frame_data) > 0:
                # Use bounding box to normalise Y
                bb = None
                row = frame_data.iloc[0]
                if row.get("fighter_a_bb") is not None:
                    bb = row["fighter_a_bb"]
                if bb is not None:
                    y_min, y_max = bb[1], bb[3]
                    # Strike wrist Y approximated from elbow angle
                    y_norm = 0.45  # default middle
                    for zi, (lo, hi) in enumerate(zone_ranges):
                        if lo <= y_norm < hi:
                            counts_a[zi] += 1
                            break
                    else:
                        counts_a[1] += 1
                else:
                    counts_a[1] += 1
            else:
                counts_a[1] += 1

        for strike in self._strikes_b:
            frame_data = self.df[self.df["frame"] == strike["frame"]]
            if len(frame_data) > 0:
                bb = None
                row = frame_data.iloc[0]
                if row.get("fighter_b_bb") is not None:
                    bb = row["fighter_b_bb"]
                if bb is not None:
                    y_norm = 0.45
                    for zi, (lo, hi) in enumerate(zone_ranges):
                        if lo <= y_norm < hi:
                            counts_b[zi] += 1
                            break
                    else:
                        counts_b[1] += 1
                else:
                    counts_b[1] += 1
            else:
                counts_b[1] += 1

        fig = go.Figure(data=go.Heatmap(
            z=[counts_a, counts_b],
            x=zones,
            y=[FIGHTER_A_NAME, FIGHTER_B_NAME],
            colorscale=[
                [0.0, "#1e1e2e"],
                [0.5, "#F59E0B"],
                [1.0, "#EF4444"],
            ],
            text=[counts_a, counts_b],
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"},
            showscale=True,
            colorbar=dict(title="Strikes"),
        ))

        fig.update_layout(
            title="Strike Zone Heatmap",
            template=PLOTLY_TEMPLATE,
            height=280,
            xaxis_title="Target Zone",
        )
        return fig

    def render_radar_comparison(self):
        """Radar/spider chart: comparing fighters across multiple dimensions."""
        categories = ["Speed", "Volume", "Accuracy", "Guard\nStability", "Activity"]

        def _safe(vals):
            clean = [v for v in vals if v is not None]
            return clean

        speeds_a = _safe(self.df["fighter_a_wrist_speed"].tolist())
        speeds_b = _safe(self.df["fighter_b_wrist_speed"].tolist())

        guards_a = _safe(self.df["fighter_a_guard_angle"].tolist())
        guards_b = _safe(self.df["fighter_b_guard_angle"].tolist())

        detected_a = self.df["fighter_a_detected"].sum()
        detected_b = self.df["fighter_b_detected"].sum()
        total = len(self.df) if len(self.df) else 1

        s = self.summary

        max_speed = max(
            max(speeds_a, default=0), max(speeds_b, default=0), 0.01)
        max_strikes = max(
            s["fighter_a_total_strikes"], s["fighter_b_total_strikes"], 1)

        def _norm_speed(vals):
            return float(np.mean(vals)) / max_speed * 100 if vals else 0

        def _norm_volume(total_strikes):
            return total_strikes / max_strikes * 100

        def _norm_accuracy(landed, total_thrown):
            return (landed / total_thrown * 100) if total_thrown else 0

        def _norm_guard(vals):
            if not vals:
                return 0
            std = float(np.std(vals))
            return max(0, 100 - std * 2)

        def _norm_activity(detected_count):
            return detected_count / total * 100

        values_a = [
            _norm_speed(speeds_a),
            _norm_volume(s["fighter_a_total_strikes"]),
            _norm_accuracy(s["fighter_a_landed"], s["fighter_a_total_strikes"]),
            _norm_guard(guards_a),
            _norm_activity(detected_a),
        ]
        values_b = [
            _norm_speed(speeds_b),
            _norm_volume(s["fighter_b_total_strikes"]),
            _norm_accuracy(s["fighter_b_landed"], s["fighter_b_total_strikes"]),
            _norm_guard(guards_b),
            _norm_activity(detected_b),
        ]

        # Close the radar
        values_a += [values_a[0]]
        values_b += [values_b[0]]
        cats = categories + [categories[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_a, theta=cats, fill="toself",
            name=FIGHTER_A_NAME,
            line=dict(color=FIGHTER_A_CHART_COLOR),
            fillcolor="rgba(59,130,246,0.2)",
        ))
        fig.add_trace(go.Scatterpolar(
            r=values_b, theta=cats, fill="toself",
            name=FIGHTER_B_NAME,
            line=dict(color=FIGHTER_B_CHART_COLOR),
            fillcolor="rgba(239,68,68,0.2)",
        ))

        fig.update_layout(
            title="Fighter Comparison",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 105],
                               showticklabels=False),
                bgcolor="#11111b",
            ),
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
        return fig

    # ------------------------------------------------------------------
    # Extended Visualizations (new for enhanced Gradio app)
    # ------------------------------------------------------------------

    def get_strike_log(self):
        """Return all strikes as a list of dicts for tabular display."""
        all_strikes = []
        for s in self._strikes_a:
            all_strikes.append({
                "Fighter": FIGHTER_A_NAME,
                "Time (s)": round(s["timestamp"], 2),
                "Type": s["type"],
                "Speed (m/s)": round(s["velocity"], 2),
                "Angle (deg)": round(s["elbow_angle"], 1),
                "Landed": "Yes" if s["landed"] else "No",
            })
        for s in self._strikes_b:
            all_strikes.append({
                "Fighter": FIGHTER_B_NAME,
                "Time (s)": round(s["timestamp"], 2),
                "Type": s["type"],
                "Speed (m/s)": round(s["velocity"], 2),
                "Angle (deg)": round(s["elbow_angle"], 1),
                "Landed": "Yes" if s["landed"] else "No",
            })
        all_strikes.sort(key=lambda x: x["Time (s)"])
        return all_strikes

    def get_round_analysis(self, round_duration_sec=180):
        """Split fight into rounds and return per-round statistics."""
        if self.df is None or len(self.df) == 0:
            return []

        total_duration = self.df["timestamp_sec"].iloc[-1]
        num_rounds = max(1, int(total_duration / round_duration_sec) + (1 if total_duration % round_duration_sec > 10 else 0))
        rounds = []

        for r in range(num_rounds):
            t_start = r * round_duration_sec
            t_end = min((r + 1) * round_duration_sec, total_duration)

            strikes_a = [s for s in self._strikes_a if t_start <= s["timestamp"] < t_end]
            strikes_b = [s for s in self._strikes_b if t_start <= s["timestamp"] < t_end]

            round_frames = self.df[
                (self.df["timestamp_sec"] >= t_start) &
                (self.df["timestamp_sec"] < t_end)
            ]

            avg_speed_a = float(np.mean([s["velocity"] for s in strikes_a])) if strikes_a else 0.0
            avg_speed_b = float(np.mean([s["velocity"] for s in strikes_b])) if strikes_b else 0.0

            landed_a = sum(1 for s in strikes_a if s["landed"])
            landed_b = sum(1 for s in strikes_b if s["landed"])

            guards_a = round_frames["fighter_a_guard_angle"].dropna().tolist() if "fighter_a_guard_angle" in round_frames.columns else []
            guards_b = round_frames["fighter_b_guard_angle"].dropna().tolist() if "fighter_b_guard_angle" in round_frames.columns else []
            avg_guard_a = float(np.mean(guards_a)) if guards_a else 0.0
            avg_guard_b = float(np.mean(guards_b)) if guards_b else 0.0

            rounds.append({
                "Round": r + 1,
                "Time": f"{t_start:.0f}s - {t_end:.0f}s",
                f"{FIGHTER_A_NAME} Strikes": len(strikes_a),
                f"{FIGHTER_B_NAME} Strikes": len(strikes_b),
                f"{FIGHTER_A_NAME} Landed": landed_a,
                f"{FIGHTER_B_NAME} Landed": landed_b,
                f"{FIGHTER_A_NAME} Avg Speed": round(avg_speed_a, 2),
                f"{FIGHTER_B_NAME} Avg Speed": round(avg_speed_b, 2),
                f"{FIGHTER_A_NAME} Guard": round(avg_guard_a, 1),
                f"{FIGHTER_B_NAME} Guard": round(avg_guard_b, 1),
            })
        return rounds

    def render_round_chart(self, round_duration_sec=180):
        """Bar chart showing strikes per round for each fighter."""
        rounds = self.get_round_analysis(round_duration_sec)
        if not rounds:
            return go.Figure().update_layout(title="No round data", template=PLOTLY_TEMPLATE)

        round_labels = [f"R{r['Round']}" for r in rounds]
        strikes_a = [r[f"{FIGHTER_A_NAME} Strikes"] for r in rounds]
        strikes_b = [r[f"{FIGHTER_B_NAME} Strikes"] for r in rounds]
        landed_a = [r[f"{FIGHTER_A_NAME} Landed"] for r in rounds]
        landed_b = [r[f"{FIGHTER_B_NAME} Landed"] for r in rounds]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=f"{FIGHTER_A_NAME} Thrown",
            x=round_labels, y=strikes_a,
            marker_color=FIGHTER_A_CHART_COLOR,
            text=strikes_a, textposition="auto",
        ))
        fig.add_trace(go.Bar(
            name=f"{FIGHTER_B_NAME} Thrown",
            x=round_labels, y=strikes_b,
            marker_color=FIGHTER_B_CHART_COLOR,
            text=strikes_b, textposition="auto",
        ))
        fig.add_trace(go.Bar(
            name=f"{FIGHTER_A_NAME} Landed",
            x=round_labels, y=landed_a,
            marker_color=FIGHTER_A_CHART_COLOR,
            marker_pattern_shape="/",
            text=landed_a, textposition="auto",
            opacity=0.6,
        ))
        fig.add_trace(go.Bar(
            name=f"{FIGHTER_B_NAME} Landed",
            x=round_labels, y=landed_b,
            marker_color=FIGHTER_B_CHART_COLOR,
            marker_pattern_shape="/",
            text=landed_b, textposition="auto",
            opacity=0.6,
        ))

        fig.update_layout(
            title="Strikes Per Round",
            barmode="group",
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_title="Round",
            yaxis_title="Strikes",
        )
        return fig

    def get_combinations(self, max_gap_sec=1.5):
        """Detect strike combinations (2-3 strike sequences within max_gap_sec)."""
        all_strikes = sorted(
            self._strikes_a + self._strikes_b,
            key=lambda x: (x["fighter"], x["timestamp"]),
        )

        combos = []
        for fighter_key in ("fighter_a", "fighter_b"):
            fighter_strikes = [s for s in all_strikes if s["fighter"] == fighter_key]
            if len(fighter_strikes) < 2:
                continue

            current_combo = [fighter_strikes[0]]
            for i in range(1, len(fighter_strikes)):
                gap = fighter_strikes[i]["timestamp"] - fighter_strikes[i-1]["timestamp"]
                if gap <= max_gap_sec:
                    current_combo.append(fighter_strikes[i])
                else:
                    if len(current_combo) >= 2:
                        combo_str = " → ".join(s["type"] for s in current_combo)
                        combos.append({
                            "Fighter": FIGHTER_A_NAME if fighter_key == "fighter_a" else FIGHTER_B_NAME,
                            "Time (s)": round(current_combo[0]["timestamp"], 2),
                            "Combination": combo_str,
                            "Strikes": len(current_combo),
                            "Avg Speed": round(float(np.mean([s["velocity"] for s in current_combo])), 2),
                        })
                    current_combo = [fighter_strikes[i]]

            if len(current_combo) >= 2:
                combo_str = " → ".join(s["type"] for s in current_combo)
                combos.append({
                    "Fighter": FIGHTER_A_NAME if fighter_key == "fighter_a" else FIGHTER_B_NAME,
                    "Time (s)": round(current_combo[0]["timestamp"], 2),
                    "Combination": combo_str,
                    "Strikes": len(current_combo),
                    "Avg Speed": round(float(np.mean([s["velocity"] for s in current_combo])), 2),
                })

        combos.sort(key=lambda x: x["Time (s)"])
        return combos

    def get_scorecard(self):
        """Compute a fighter scorecard across multiple dimensions."""
        s = self.summary
        if s["fighter_a_total_strikes"] == 0 and s["fighter_b_total_strikes"] == 0:
            return {}

        def _safe_list(col):
            return [v for v in self.df[col].dropna().tolist() if v is not None]

        speeds_a = _safe_list("fighter_a_wrist_speed")
        speeds_b = _safe_list("fighter_b_wrist_speed")
        guards_a = _safe_list("fighter_a_guard_angle")
        guards_b = _safe_list("fighter_b_guard_angle")

        criteria = {
            "Strike Volume": (
                s["fighter_a_total_strikes"],
                s["fighter_b_total_strikes"],
            ),
            "Strikes Landed": (
                s["fighter_a_landed"],
                s["fighter_b_landed"],
            ),
            "Accuracy (%)": (
                round(s["fighter_a_landed"] / max(s["fighter_a_total_strikes"], 1) * 100, 1),
                round(s["fighter_b_landed"] / max(s["fighter_b_total_strikes"], 1) * 100, 1),
            ),
            "Avg Speed (m/s)": (
                round(float(np.mean(speeds_a)), 2) if speeds_a else 0,
                round(float(np.mean(speeds_b)), 2) if speeds_b else 0,
            ),
            "Max Speed (m/s)": (
                round(float(max(speeds_a, default=0)), 2),
                round(float(max(speeds_b, default=0)), 2),
            ),
            "Guard Stability": (
                round(float(np.std(guards_a)), 1) if guards_a else 99,
                round(float(np.std(guards_b)), 1) if guards_b else 99,
            ),
            "Jabs": (
                s["fighter_a_strike_types"]["Jab"],
                s["fighter_b_strike_types"]["Jab"],
            ),
            "Crosses": (
                s["fighter_a_strike_types"]["Cross"],
                s["fighter_b_strike_types"]["Cross"],
            ),
            "Hooks": (
                s["fighter_a_strike_types"]["Hook"],
                s["fighter_b_strike_types"]["Hook"],
            ),
        }

        # Score each dimension: 10 for winner, proportion for loser
        scorecard = {}
        total_a, total_b = 0, 0
        for metric, (va, vb) in criteria.items():
            if va > vb:
                score_a, score_b = 10, round(vb / max(va, 0.01) * 10, 1) if va > 0 else 5
            elif vb > va:
                score_b, score_a = 10, round(va / max(vb, 0.01) * 10, 1) if vb > 0 else 5
            else:
                score_a, score_b = 5, 5
            total_a += score_a
            total_b += score_b
            scorecard[metric] = {
                FIGHTER_A_NAME: score_a,
                FIGHTER_B_NAME: score_b,
                "winner": FIGHTER_A_NAME if va > vb else (FIGHTER_B_NAME if vb > va else "Tie"),
            }

        scorecard["_totals"] = {
            FIGHTER_A_NAME: round(total_a, 1),
            FIGHTER_B_NAME: round(total_b, 1),
        }
        scorecard["_winner"] = FIGHTER_A_NAME if total_a > total_b else (FIGHTER_B_NAME if total_b > total_a else "Draw")
        return scorecard

    def render_elbow_angle_timeline(self):
        """Line chart: elbow angles over time for both fighters."""
        if self.df is None or len(self.df) == 0:
            return go.Figure().update_layout(title="No data", template=PLOTLY_TEMPLATE)

        fig = go.Figure()
        for side, prefix, color, name in [
            ("Left", "fighter_a", FIGHTER_A_CHART_COLOR, FIGHTER_A_NAME),
            ("Right", "fighter_b", FIGHTER_B_CHART_COLOR, FIGHTER_B_NAME),
        ]:
            for elbow in ("left", "right"):
                col = f"{prefix}_{elbow}_elbow_angle"
                if col in self.df.columns:
                    vals = self.df[col].tolist()
                    fig.add_trace(go.Scatter(
                        x=self.df["timestamp_sec"],
                        y=vals,
                        mode="lines",
                        name=f"{name} {elbow.title()} Elbow",
                        line=dict(
                            color=color,
                            width=2,
                            dash="dot" if elbow == "right" else "solid",
                        ),
                    ))

        fig.update_layout(
            title="Elbow Angle Analysis",
            xaxis_title="Time (s)",
            yaxis_title="Angle (degrees)",
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    def render_knee_angle_timeline(self):
        """Line chart: knee angles over time for both fighters."""
        if self.df is None or len(self.df) == 0:
            return go.Figure().update_layout(title="No data", template=PLOTLY_TEMPLATE)

        fig = go.Figure()
        for prefix, color, name in [
            ("fighter_a", FIGHTER_A_CHART_COLOR, FIGHTER_A_NAME),
            ("fighter_b", FIGHTER_B_CHART_COLOR, FIGHTER_B_NAME),
        ]:
            for knee in ("left", "right"):
                col = f"{prefix}_{knee}_knee_angle"
                if col in self.df.columns:
                    vals = self.df[col].tolist()
                    fig.add_trace(go.Scatter(
                        x=self.df["timestamp_sec"],
                        y=vals,
                        mode="lines",
                        name=f"{name} {knee.title()} Knee",
                        line=dict(
                            color=color,
                            width=2,
                            dash="dot" if knee == "right" else "solid",
                        ),
                    ))

        fig.update_layout(
            title="Knee Angle Analysis",
            xaxis_title="Time (s)",
            yaxis_title="Angle (degrees)",
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    def render_activity_heatmap(self):
        """Heatmap: fighter activity over time (10-second windows)."""
        if self.df is None or len(self.df) == 0:
            return go.Figure().update_layout(title="No data", template=PLOTLY_TEMPLATE)

        window_sec = 10.0
        total_time = self.df["timestamp_sec"].iloc[-1]
        num_windows = max(1, int(total_time / window_sec) + 1)

        activity_a = np.zeros(num_windows)
        activity_b = np.zeros(num_windows)

        for s in self._strikes_a:
            w = int(s["timestamp"] / window_sec)
            if 0 <= w < num_windows:
                activity_a[w] += 1

        for s in self._strikes_b:
            w = int(s["timestamp"] / window_sec)
            if 0 <= w < num_windows:
                activity_b[w] += 1

        window_labels = [f"{int(i*window_sec)}-{int((i+1)*window_sec)}s" for i in range(num_windows)]

        fig = go.Figure(data=go.Heatmap(
            z=[activity_a.tolist(), activity_b.tolist()],
            x=window_labels,
            y=[FIGHTER_A_NAME, FIGHTER_B_NAME],
            colorscale=[
                [0.0, "#1e1e2e"],
                [0.3, "#1e3a5f"],
                [0.6, "#3B82F6"],
                [1.0, "#F59E0B"],
            ],
            text=[activity_a.tolist(), activity_b.tolist()],
            texttemplate="%{text}",
            textfont={"size": 12, "color": "white"},
            showscale=True,
            colorbar=dict(title="Strikes"),
        ))

        fig.update_layout(
            title="Fighter Activity Heatmap",
            template=PLOTLY_TEMPLATE,
            height=250,
            xaxis_title="Time Window",
        )
        return fig
