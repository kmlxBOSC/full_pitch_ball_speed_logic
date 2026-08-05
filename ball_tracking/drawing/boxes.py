"""
ball_tracking.drawing.boxes
==============================
Bounding-box + label drawing. Pure OpenCV drawing helpers with no
dependency on detection or analysis logic — pass 2 (`drawing.renderer`)
is the only thing that decides what to draw where.
"""

from typing import Sequence, Tuple

import cv2
import numpy as np

from ball_tracking.core import config
from ball_tracking.core.schemas import Detection


def draw_box(frame: np.ndarray, detection: Detection, color: Tuple[int, int, int]) -> None:
    """Draw a bounding box with a "<class> <confidence>" label above it. Mutates *frame* in place."""
    x1, y1, x2, y2 = detection.bbox.as_int_tuple()
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)

    label = f"{detection.class_name} {detection.confidence:.2f}"
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FONT_SCALE,
        color,
        config.FONT_THICKNESS,
        cv2.LINE_AA,
    )


def draw_boxes(frame: np.ndarray, detections: Sequence[Detection], color: Tuple[int, int, int]) -> None:
    """Draw every detection in *detections* with the same color. Mutates *frame* in place."""
    for detection in detections:
        draw_box(frame, detection, color)
