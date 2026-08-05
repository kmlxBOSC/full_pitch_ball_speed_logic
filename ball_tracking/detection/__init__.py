"""ball_tracking.detection — pass 1: detection only, no drawing."""

from ball_tracking.detection.pipeline import CricketDetectionPipeline
from ball_tracking.detection.runner import cache_path_for, run_detection_pass

__all__ = ["CricketDetectionPipeline", "run_detection_pass", "cache_path_for"]
