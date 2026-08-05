"""
ball_tracking.drawing.pitch_overlay
======================================
Draws a broadcast-style pitch graphic (side lines, centre line, bowling
and popping creases, return creases, wide guidelines, distance markers,
and clean synthetic "dummy" stumps redrawn over the detected real ones)
so a viewer can visually sanity-check the calibration — that's the
actual purpose of this overlay, not just decoration.

Every element here is a single straight line between two points
returned by `PitchCalibration.pitch_to_pixel` (or its `boundary_points_at`
convenience for the +-1.525 m pitch edges) — every real pitch marking is
a straight line in the world, and a homography always maps a straight
line to a straight line, so no curve-sampling is ever needed; the only
per-element work is picking the right pair of metric endpoints. Lines
that run off-frame (typically the bowling end, off the top/side of a
tight shot) are clipped to the image border with `cv2.clipLine` rather
than assumed to always land on-screen.
"""

from typing import Tuple

import cv2
import numpy as np

from ball_tracking.analysis.calibration import PitchCalibration
from ball_tracking.core import config

Point = Tuple[float, float]

# Coordinates are clamped to this range before rounding to int and handed to
# OpenCV, so an extrapolated off-screen endpoint (e.g. the far/small end's
# boundary projected a long way past the frame) can never overflow a 32-bit
# int in cv2.clipLine — any point already this far outside the frame clips
# identically regardless of exactly how far past the border it lands.
_COORD_CLAMP = 1_000_000.0


def _clamped_int_point(point: Point) -> Tuple[int, int]:
    x = max(-_COORD_CLAMP, min(_COORD_CLAMP, point[0]))
    y = max(-_COORD_CLAMP, min(_COORD_CLAMP, point[1]))
    return (int(round(x)), int(round(y)))


def _draw_line(frame: np.ndarray, p1: Point, p2: Point, color, thickness: int) -> None:
    """Draw a straight line, truncated naturally at the frame border if either end is off-screen."""
    h, w = frame.shape[:2]
    ok, c1, c2 = cv2.clipLine((0, 0, w, h), _clamped_int_point(p1), _clamped_int_point(p2))
    if ok:
        cv2.line(frame, c1, c2, color, thickness, cv2.LINE_AA)


def _draw_parallel_pair(
    frame: np.ndarray, calibration: PitchCalibration, x_range: Tuple[float, float], offset_m: float, color, thickness: int,
) -> None:
    """Draw the two lines at +-*offset_m* from the centre line, each spanning *x_range*."""
    x0, x1 = x_range
    for y_m in (offset_m, -offset_m):
        _draw_line(frame, calibration.pitch_to_pixel(x0, y_m), calibration.pitch_to_pixel(x1, y_m), color, thickness)


def draw_pitch_overlay(frame: np.ndarray, calibration: PitchCalibration) -> None:
    """Draw the calibration/broadcast-style pitch graphic onto *frame* in place."""
    x_span = (-config.POPPING_CREASE_OFFSET_M, config.PITCH_LENGTH_M + config.POPPING_CREASE_OFFSET_M)

    far_left, far_right = calibration.boundary_points_at(x_span[0])
    near_left, near_right = calibration.boundary_points_at(x_span[1])

    # Side ("wide") lines running the length of the pitch — each a single
    # straight segment between the two ends' projected boundary points.
    _draw_line(frame, far_left, near_left, config.PITCH_SIDE_LINE_COLOR, config.PITCH_OVERLAY_LINE_THICKNESS)
    _draw_line(frame, far_right, near_right, config.PITCH_SIDE_LINE_COLOR, config.PITCH_OVERLAY_LINE_THICKNESS)

    # Centre line, through both wickets.
    _draw_line(
        frame, calibration.pitch_to_pixel(x_span[0], 0.0), calibration.pitch_to_pixel(x_span[1], 0.0),
        config.PITCH_CENTER_LINE_COLOR, config.PITCH_OVERLAY_LINE_THICKNESS,
    )

    # Bowling creases (through the stumps) and popping creases (in front of them), both ends.
    crease_x_positions = (
        0.0,
        config.PITCH_LENGTH_M,
        config.POPPING_CREASE_OFFSET_M,
        config.PITCH_LENGTH_M - config.POPPING_CREASE_OFFSET_M,
    )
    for x_m in crease_x_positions:
        left, right = calibration.boundary_points_at(x_m)
        _draw_line(frame, left, right, config.PITCH_CREASE_LINE_COLOR, config.PITCH_OVERLAY_LINE_THICKNESS)

    # Return creases and wide guidelines: both run from the bowling crease to
    # the popping crease at each end, parallel to the centre line.
    end_x_ranges = (
        (0.0, config.POPPING_CREASE_OFFSET_M),
        (config.PITCH_LENGTH_M - config.POPPING_CREASE_OFFSET_M, config.PITCH_LENGTH_M),
    )
    for x_range in end_x_ranges:
        _draw_parallel_pair(frame, calibration, x_range, config.RETURN_CREASE_OFFSET_M, config.PITCH_RETURN_CREASE_COLOR, config.PITCH_SECONDARY_LINE_THICKNESS)
        _draw_parallel_pair(frame, calibration, x_range, config.WIDE_GUIDELINE_OFFSET_M, config.PITCH_WIDE_GUIDELINE_COLOR, config.PITCH_SECONDARY_LINE_THICKNESS)

    # Distance markers from the batting end (the small/far stump), drawn as
    # unlabelled ticks across the full pitch width. Spacing shrinks
    # non-linearly toward the horizon for free, since it's the same
    # homography used for every other marking on the plane.
    for meters in config.PITCH_DISTANCE_MARKERS_M:
        left, right = calibration.boundary_points_at(meters)
        _draw_line(frame, left, right, config.PITCH_MARKER_LINE_COLOR, config.PITCH_MARKER_LINE_THICKNESS)

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
