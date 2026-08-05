"""ball_tracking.detection.human — human / pose (COCO 17-point skeleton) detector."""

from typing import List, Sequence

from ball_tracking.core import config
from ball_tracking.core.schemas import BoundingBox, Keypoint, PersonDetection
from ball_tracking.detection.base import BaseDetector
from models.infer import Inference


class HumanDetector(BaseDetector):
    """Detects people and their pose keypoints (COCO 17-point skeleton).

    Always runs in standard (non-sliced) mode: SAHI's slice merge only
    understands bounding boxes, so tiling here would discard keypoints.
    """

    def __init__(self, inference: Inference) -> None:
        super().__init__(
            inference=inference,
            class_name=config.HUMAN_CLASS_NAME,
            conf_threshold=config.HUMAN_CONF_THRESHOLD,
            iou_threshold=config.HUMAN_IOU_THRESHOLD,
            use_sahi=config.HUMAN_USE_SAHI,
        )

    def _parse_result(self, result) -> Sequence[PersonDetection]:
        detections: List[PersonDetection] = []
        boxes = result.boxes
        if boxes is None:
            return detections

        names = result.names
        keypoints_data = result.keypoints
        for idx, box in enumerate(boxes):
            label = names[int(box.cls[0])]
            if label != self._class_name:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                PersonDetection(
                    class_name=label,
                    confidence=float(box.conf[0]),
                    bbox=BoundingBox(x1, y1, x2, y2),
                    keypoints=self._extract_keypoints(keypoints_data, idx),
                )
            )
        return detections

    @staticmethod
    def _extract_keypoints(keypoints_data, idx: int) -> tuple:
        """Build named `Keypoint`s for detection *idx*, if pose data is present."""
        if keypoints_data is None or keypoints_data.conf is None:
            return tuple()

        xy = keypoints_data.xy[idx].tolist()
        conf = keypoints_data.conf[idx].tolist()
        return tuple(
            Keypoint(name=name, x=float(pt[0]), y=float(pt[1]), confidence=float(c))
            for name, pt, c in zip(config.COCO_KEYPOINT_NAMES, xy, conf)
        )
