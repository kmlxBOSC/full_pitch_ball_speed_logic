"""ball_tracking — cricket-scene object detection, pitch calibration, ball speed and drawing."""

from ball_tracking.core.schemas import (
    BoundingBox,
    Detection,
    DetectionRun,
    FrameDetections,
    Keypoint,
    PersonDetection,
    SpeedResult,
)
from ball_tracking.pipeline import process_clip, process_folder

__all__ = [
    "process_clip",
    "process_folder",
    "BoundingBox",
    "Detection",
    "DetectionRun",
    "FrameDetections",
    "Keypoint",
    "PersonDetection",
    "SpeedResult",
]
