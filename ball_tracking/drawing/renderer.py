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
from ball_tracking.core.schemas import FrameDetections, SpeedResult
from ball_tracking.drawing.boxes import draw_boxes
from ball_tracking.drawing.pitch_overlay import draw_pitch_overlay
from ball_tracking.drawing.skeleton import draw_skeleton
from ball_tracking.drawing.speed_hud import draw_speed_hud


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
    than failing pass 2 for the whole clip.
    """
    annotated = frame.copy()

    if calibration is not None:
        draw_pitch_overlay(annotated, calibration)

    draw_boxes(annotated, detections.balls, config.BALL_BOX_COLOR)
    draw_boxes(annotated, detections.stumps, config.STUMP_BOX_COLOR)
    for human in detections.humans:
        draw_boxes(annotated, [human], config.HUMAN_BOX_COLOR)
        draw_skeleton(annotated, human)

    if speed is not None:
        draw_speed_hud(annotated, speed)

    return annotated
