"""ball_tracking.detection.stump — stump (wicket) detector."""

from ball_tracking.core import config
from ball_tracking.detection.base import BaseDetector
from models.infer import Inference


class StumpDetector(BaseDetector):
    """Detects the stumps. Each wicket (3 stumps) is returned as a single box."""

    def __init__(self, inference: Inference) -> None:
        super().__init__(
            inference=inference,
            class_name=config.STUMP_CLASS_NAME,
            conf_threshold=config.STUMP_CONF_THRESHOLD,
            iou_threshold=config.STUMP_IOU_THRESHOLD,
            use_sahi=config.STUMP_USE_SAHI,
        )
