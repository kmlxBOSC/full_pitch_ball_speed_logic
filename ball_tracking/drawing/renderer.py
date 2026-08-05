"""
ball_tracking.drawing.renderer
=================================
Pass 2 entry point: composes the pitch overlay, raw per-frame detection
boxes/skeletons and the speed HUD onto one frame. This is the only
place in `drawing` that knows about all the other annotators; nothing
here imports `models` or `detection` — it only consumes the schemas and
analysis outputs already computed by pass 1 and the analysis stage.
"""

from typing import Optional

import numpy as np

from ball_tracking.analysis.calibration import PitchCalibration
from ball_tracking.core import config
from ball_tracking.core.schemas import FrameDetections, PersonDetection, SpeedResult
from ball_tracking.drawing.boxes import draw_boxes
from ball_tracking.drawing.pitch_overlay import draw_pitch_overlay
from ball_tracking.drawing.skeleton import draw_skeleton
from ball_tracking.drawing.speed_hud import draw_speed_hud


def _select_bowler(humans: "tuple[PersonDetection, ...]") -> Optional[PersonDetection]:
    """Return the bowler among *humans*, or None if that can't be determined.

    Only trusted when exactly two people are detected (bowler + batter) —
    the bowler is the one with the larger box, since this camera sits
    behind the bowler's stumps looking down the pitch, so the bowler is
    always closer to camera than the batter at the far end. With any
    other head-count the pairing is ambiguous, so nobody is selected.
    """
    if len(humans) != 2:
        return None
    return max(humans, key=lambda h: h.bbox.width * h.bbox.height)


def render_frame(
    frame: np.ndarray,
    detections: FrameDetections,
    calibration: Optional[PitchCalibration],
    speed: Optional[SpeedResult],
) -> np.ndarray:
    """Return an annotated copy of *frame*.

    The pitch overlay is skipped if *calibration* is None (couldn't
    calibrate this clip) and the speed HUD is skipped if *speed* is None
    (speed wasn't computed for this run) — both degrade gracefully rather
    than failing pass 2 for the whole clip. Human boxes are never drawn;
    the skeleton is only drawn for the bowler (see `_select_bowler`).
    """
    annotated = frame.copy()

    if calibration is not None:
        draw_pitch_overlay(annotated, calibration)

    draw_boxes(annotated, detections.balls, config.BALL_BOX_COLOR)

    bowler = _select_bowler(detections.humans)
    if bowler is not None:
        draw_skeleton(annotated, bowler)

    if speed is not None:
        draw_speed_hud(annotated, speed)

    return annotated
