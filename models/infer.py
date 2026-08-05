"""infer — YOLO model loading and inference wrapper."""

from typing import Any

import torch
from ultralytics import YOLO

from ball_tracking.core.logger import get_logger

logger = get_logger(__name__)


class Inference:
    """Thin wrapper around a YOLO model for ball / bat / stump / pose detection."""


    def __init__(self) -> None:
        self.model: YOLO | None = None
        self.device: str = "cpu"


    def load_model(self, model_path: str) -> None:
        """Load a YOLO model from *model_path* onto the best available device."""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = YOLO(model_path).to(self.device)
            logger.debug("Model loaded on %s: %s", self.device.upper(), model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model: {exc}") from exc


    def predict(
        self,
        source: Any,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        verbose: bool = False,
    ) -> list:
        """Run inference on *source* and return the raw ultralytics results.

        Args:
            source: Image / frame (e.g. a numpy array) or path accepted by ultralytics.
            conf: Minimum confidence score required to keep a detection.
            iou: IoU threshold used for non-max suppression.
            imgsz: Inference image size (longest side, in pixels).
            verbose: Whether ultralytics should print its own inference logs.

        Returns:
            One `ultralytics.engine.results.Results` object per input frame.

        Raises:
            RuntimeError: If the model has not been loaded yet, or inference fails.
        """
        if self.model is None:
            raise RuntimeError("Model has not been loaded. Call load_model() first.")

        try:
            return self.model.predict(source, conf=conf, iou=iou, imgsz=imgsz, verbose=verbose)
        except Exception as exc:
            raise RuntimeError(f"Inference failed: {exc}") from exc
