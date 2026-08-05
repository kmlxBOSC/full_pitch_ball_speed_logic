"""
ball_tracking.drawing.skeleton
=================================
Pose skeleton drawing (COCO 17-point). Pure OpenCV drawing helper with
no dependency on detection or analysis logic.
"""

import cv2
import numpy as np

from ball_tracking.core import config
from ball_tracking.core.schemas import PersonDetection


def draw_skeleton(frame: np.ndarray, person: PersonDetection) -> None:
    """Draw keypoints and connecting skeleton edges for one detected person. Mutates *frame* in place."""
    visible = {
        idx: (int(kp.x), int(kp.y))
        for idx, kp in enumerate(person.keypoints)
        if kp.confidence >= config.HUMAN_KEYPOINT_CONF_THRESHOLD
    }

    for start_idx, end_idx in config.COCO_SKELETON_EDGES:
        if start_idx in visible and end_idx in visible:
            cv2.line(frame, visible[start_idx], visible[end_idx], config.SKELETON_COLOR, config.SKELETON_THICKNESS)

    for point in visible.values():
        cv2.circle(frame, point, config.KEYPOINT_RADIUS, config.KEYPOINT_COLOR, -1)
