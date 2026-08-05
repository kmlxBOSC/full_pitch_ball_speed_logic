"""ball_tracking.detection.ball — cricket ball detector."""

from ball_tracking.core import config
from ball_tracking.detection.base import BaseDetector
from models.infer import Inference


class BallDetector(BaseDetector):
    """Detects the cricket ball."""

    def __init__(self, inference: Inference) -> None:
        super().__init__(
            inference=inference,
            class_name=config.BALL_CLASS_NAME,
            conf_threshold=config.BALL_CONF_THRESHOLD,
            iou_threshold=config.BALL_IOU_THRESHOLD,
            use_sahi=config.BALL_USE_SAHI,
        )
