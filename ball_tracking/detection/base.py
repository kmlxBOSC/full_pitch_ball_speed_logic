"""
ball_tracking.detection.base
==============================
Common inference plumbing shared by every concrete detector. Two
inference modes are supported, selected per-detector via config:

- Standard: one whole-frame forward pass through ultralytics
  (`Inference.predict`). Fast, but small objects can shrink to just a
  few pixels once the frame is resized to the model's input size.
- Sliced (SAHI): the frame is tiled into overlapping slices, each slice
  is run through the model independently, and the per-slice boxes are
  merged back into full-frame coordinates. Substantially better recall
  on tiny objects (the ball, the stumps), at the cost of one extra
  forward pass per slice — see `ball_tracking.core.config` for the cost
  tradeoff notes and per-detector toggles.

Adding a new object type only requires a new subclass of `BaseDetector`
(or an override of `_parse_result` for non-standard outputs, as
`HumanDetector` does for pose keypoints) — no changes to the pipeline.
"""

from abc import ABC
from typing import List, Sequence

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from ball_tracking.core import config
from ball_tracking.core.logger import get_logger
from ball_tracking.core.schemas import BoundingBox, Detection
from models.infer import Inference

logger = get_logger(__name__)


class BaseDetector(ABC):
    """Common inference plumbing shared by every concrete detector."""

    def __init__(
        self,
        inference: Inference,
        class_name: str,
        conf_threshold: float,
        iou_threshold: float,
        image_size: int = config.INFERENCE_IMAGE_SIZE,
        use_sahi: bool = False,
    ) -> None:
        self._inference = inference
        self._class_name = class_name
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._image_size = image_size
        self._use_sahi = use_sahi
        self._sahi_model = self._build_sahi_model() if use_sahi else None

    def _build_sahi_model(self) -> AutoDetectionModel:
        """Wrap the already-loaded ultralytics model for sliced (SAHI) inference.

        Reuses `self._inference.model` directly (no re-loading weights from
        disk) via SAHI's `model=` pass-through.
        """
        return AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model=self._inference.model,
            confidence_threshold=self._conf_threshold,
            device=self._inference.device,
            image_size=self._image_size,
        )

    def detect(self, frame: np.ndarray) -> Sequence[Detection]:
        """Run inference on *frame* and return parsed detections."""
        detections = self._detect_sliced(frame) if self._use_sahi else self._detect_standard(frame)
        logger.debug("%s: found %d detection(s)", self.__class__.__name__, len(detections))
        return detections

    def _detect_standard(self, frame: np.ndarray) -> Sequence[Detection]:
        """Run one whole-frame ultralytics pass and parse its `Results`."""
        results = self._inference.predict(
            frame,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            imgsz=self._image_size,
        )
        return self._parse_result(results[0]) if results else []

    def _detect_sliced(self, frame: np.ndarray) -> Sequence[Detection]:
        """Run tiled (SAHI) inference on *frame* and return merged detections.

        SAHI treats numpy-array input as already being in RGB order (unlike
        ultralytics, which accepts OpenCV's native BGR), so the frame is
        converted before slicing.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = get_sliced_prediction(
            image=rgb_frame,
            detection_model=self._sahi_model,
            slice_height=config.SAHI_SLICE_HEIGHT,
            slice_width=config.SAHI_SLICE_WIDTH,
            overlap_height_ratio=config.SAHI_OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=config.SAHI_OVERLAP_WIDTH_RATIO,
            perform_standard_pred=config.SAHI_PERFORM_STANDARD_PRED,
            postprocess_type=config.SAHI_POSTPROCESS_TYPE,
            postprocess_match_metric=config.SAHI_POSTPROCESS_MATCH_METRIC,
            postprocess_match_threshold=config.SAHI_POSTPROCESS_MATCH_THRESHOLD,
            verbose=0,
        )
        return [
            Detection(
                class_name=object_prediction.category.name,
                confidence=object_prediction.score.value,
                bbox=BoundingBox(
                    object_prediction.bbox.minx,
                    object_prediction.bbox.miny,
                    object_prediction.bbox.maxx,
                    object_prediction.bbox.maxy,
                ),
            )
            for object_prediction in result.object_prediction_list
            if object_prediction.category.name == self._class_name
            and object_prediction.score.value >= self._conf_threshold
        ]

    def _parse_result(self, result) -> Sequence[Detection]:
        """Convert a single ultralytics `Results` object into `Detection`s.

        Only used in standard (non-sliced) mode. Subclasses that need
        richer output (e.g. pose keypoints) should override this method.
        """
        detections: List[Detection] = []
        boxes = result.boxes
        if boxes is None:
            return detections

        names = result.names
        for box in boxes:
            label = names[int(box.cls[0])]
            if label != self._class_name:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=label,
                    confidence=float(box.conf[0]),
                    bbox=BoundingBox(x1, y1, x2, y2),
                )
            )
        return detections
