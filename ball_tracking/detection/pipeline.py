"""
ball_tracking.detection.pipeline
===================================
Single-frame orchestration layer. Runs the ball, stump and human
detectors against one frame and returns one aggregated `FrameDetections`.
This is the primary per-frame entry point pass 1 (`detection.runner`)
uses — it contains no drawing and no analysis (calibration/speed) logic.
"""

import numpy as np

from ball_tracking.core.logger import get_logger
from ball_tracking.core.schemas import FrameDetections
from ball_tracking.detection.ball import BallDetector
from ball_tracking.detection.human import HumanDetector
from ball_tracking.detection.stump import StumpDetector
from models import obj_ball, obj_human, obj_stump

logger = get_logger(__name__)


class CricketDetectionPipeline:
    """Runs ball, stump and human detection on a frame in one call.

    Detectors are constructed once (at pipeline creation) and reused
    across frames, avoiding repeated model-access overhead.
    """

    def __init__(self) -> None:
        self._ball_detector = BallDetector(obj_ball)
        self._stump_detector = StumpDetector(obj_stump)
        self._human_detector = HumanDetector(obj_human)

    def detect(self, frame: np.ndarray) -> FrameDetections:
        """Run all three detectors on *frame* and return combined results."""
        balls = self._ball_detector.detect(frame)
        stumps = self._stump_detector.detect(frame)
        humans = self._human_detector.detect(frame)

        logger.debug(
            "Frame detections -> balls: %d, stumps: %d, humans: %d",
            len(balls), len(stumps), len(humans),
        )
        return FrameDetections(
            balls=tuple(balls),
            stumps=tuple(stumps),
            humans=tuple(humans),
        )
