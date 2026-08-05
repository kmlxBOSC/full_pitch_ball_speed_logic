"""
ball_tracking.drawing.speed_hud
==================================
Headline release/peak speed overlay (broadcast-style "142.3 km/h" card),
drawn once per frame in the top-left corner.
"""

import cv2
import numpy as np

from ball_tracking.core import config
from ball_tracking.core.schemas import SpeedResult

_MARGIN = 16
_PADDING = 10


def draw_speed_hud(frame: np.ndarray, speed: SpeedResult) -> None:
    """Draw the computed speed (or a short "N/A" note) onto *frame* in place."""
    if speed.valid:
        value = speed.speed_kmh if config.SPEED_HUD_UNIT == "kmh" else speed.speed_mph
        unit = "km/h" if config.SPEED_HUD_UNIT == "kmh" else "mph"
        text = f"{value:.1f} {unit}"
    else:
        text = "Speed: N/A"

    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, config.SPEED_HUD_FONT_SCALE, config.SPEED_HUD_FONT_THICKNESS
    )
    x0, y0 = _MARGIN, _MARGIN
    x1, y1 = x0 + text_w + 2 * _PADDING, y0 + text_h + baseline + 2 * _PADDING

    cv2.rectangle(frame, (x0, y0), (x1, y1), config.SPEED_HUD_BG_COLOR, -1)
    cv2.putText(
        frame, text,
        (x0 + _PADDING, y1 - _PADDING - baseline // 2),
        cv2.FONT_HERSHEY_SIMPLEX, config.SPEED_HUD_FONT_SCALE, config.SPEED_HUD_TEXT_COLOR,
        config.SPEED_HUD_FONT_THICKNESS, cv2.LINE_AA,
    )
