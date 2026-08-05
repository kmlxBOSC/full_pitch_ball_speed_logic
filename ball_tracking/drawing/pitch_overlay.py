"""
ball_tracking.drawing.pitch_overlay
======================================
Draws a broadcast-style pitch graphic (side lines, creases, distance
markers, and clean synthetic "dummy" stumps redrawn over the detected
real ones) so a viewer can visually sanity-check the calibration —
that's the actual purpose of this overlay, not just decoration.

Every line is drawn as a sampled polyline via
`PitchCalibration.pitch_to_pixel` rather than assumed to be straight
between two endpoints — the along-pitch and lateral calibration are
fitted independently (see `analysis.calibration`), so a line that's
straight in reality might be a hair off dead-straight in the fitted
model; sampling several points and connecting them stays visually
correct either way.
"""

from typing import List, Tuple

import cv2
import numpy as np

from ball_tracking.analysis.calibration import PitchCalibration
from ball_tracking.core import config

Point = Tuple[float, float]

_SAMPLES_PER_LINE = 24


def _sample_along_x(calibration: PitchCalibration, y_m: float, x_range: Tuple[float, float]) -> List[Point]:
    x0, x1 = x_range
    return [
        calibration.pitch_to_pixel(x0 + (x1 - x0) * i / (_SAMPLES_PER_LINE - 1), y_m)
        for i in range(_SAMPLES_PER_LINE)
    ]


def _sample_along_y(calibration: PitchCalibration, x_m: float, y_range: Tuple[float, float]) -> List[Point]:
    y0, y1 = y_range
    return [
        calibration.pitch_to_pixel(x_m, y0 + (y1 - y0) * i / (_SAMPLES_PER_LINE - 1))
        for i in range(_SAMPLES_PER_LINE)
    ]


def _draw_polyline(frame: np.ndarray, points: List[Point], color, thickness: int) -> None:
    int_points = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
    cv2.polylines(frame, [int_points], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_pitch_overlay(frame: np.ndarray, calibration: PitchCalibration) -> None:
    """Draw the calibration/broadcast-style pitch graphic onto *frame* in place."""
    half_pitch_w = config.PITCH_WIDTH_M / 2.0
    x_span = (-config.POPPING_CREASE_OFFSET_M, config.PITCH_LENGTH_M + config.POPPING_CREASE_OFFSET_M)

    # Side ("wide") lines running the length of the pitch.
    _draw_polyline(frame, _sample_along_x(calibration, +half_pitch_w, x_span), config.PITCH_SIDE_LINE_COLOR, config.PITCH_OVERLAY_LINE_THICKNESS)
    _draw_polyline(frame, _sample_along_x(calibration, -half_pitch_w, x_span), config.PITCH_SIDE_LINE_COLOR, config.PITCH_OVERLAY_LINE_THICKNESS)

    # Centre line, straight through both middle stumps.
    _draw_polyline(frame, _sample_along_x(calibration, 0.0, x_span), config.PITCH_CENTER_LINE_COLOR, config.PITCH_OVERLAY_LINE_THICKNESS)

    # Bowling creases (through the stumps) and popping creases (in front of them), both ends.
    crease_y_span = (-half_pitch_w, half_pitch_w)
    crease_x_positions = (
        0.0,
        config.PITCH_LENGTH_M,
        config.POPPING_CREASE_OFFSET_M,
        config.PITCH_LENGTH_M - config.POPPING_CREASE_OFFSET_M,
    )
    for x_m in crease_x_positions:
        _draw_polyline(frame, _sample_along_y(calibration, x_m, crease_y_span), config.PITCH_CREASE_LINE_COLOR, config.PITCH_OVERLAY_LINE_THICKNESS)

    # Distance markers from the batting end (the small/far stump).
    marker_y_span = (-half_pitch_w * 0.5, half_pitch_w * 0.5)
    for meters in config.PITCH_DISTANCE_MARKERS_M:
        _draw_polyline(frame, _sample_along_y(calibration, meters, marker_y_span), config.PITCH_MARKER_LINE_COLOR, config.PITCH_MARKER_LINE_THICKNESS)
        label_point = calibration.pitch_to_pixel(meters, marker_y_span[1])
        cv2.putText(
            frame, f"{meters:g}m",
            (int(round(label_point[0])) + 6, int(round(label_point[1]))),
            cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.PITCH_MARKER_TEXT_COLOR, config.FONT_THICKNESS, cv2.LINE_AA,
        )

    _draw_dummy_stumps(frame, calibration, x_m=0.0, top_y=calibration.far_reference.top_center[1])
    _draw_dummy_stumps(frame, calibration, x_m=config.PITCH_LENGTH_M, top_y=calibration.near_reference.top_center[1])


def _draw_dummy_stumps(frame: np.ndarray, calibration: PitchCalibration, x_m: float, top_y: float) -> None:
    """Draw a clean synthetic 3-stump wicket at along-pitch position *x_m*.

    Reuses the *measured* top-of-stump image row for that end (`top_y`)
    rather than re-deriving stump height from the ground-plane
    calibration — both wickets are effectively single points at one
    depth each, so their real detected top edge is already an accurate,
    simpler stand-in for a full 3-D height projection.
    """
    half_w = config.STUMP_GROUP_WIDTH_M / 2.0
    base_points = [calibration.pitch_to_pixel(x_m, y) for y in (-half_w, 0.0, half_w)]

    for bx, by in base_points:
        top = (int(round(bx)), int(round(top_y)))
        base = (int(round(bx)), int(round(by)))
        cv2.line(frame, base, top, config.DUMMY_STUMP_COLOR, config.DUMMY_STUMP_THICKNESS)

    left_top = (int(round(base_points[0][0])), int(round(top_y)))
    right_top = (int(round(base_points[-1][0])), int(round(top_y)))
    cv2.line(frame, left_top, right_top, config.DUMMY_STUMP_COLOR, config.DUMMY_STUMP_THICKNESS)
