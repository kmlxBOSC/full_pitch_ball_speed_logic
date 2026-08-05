"""
ball_tracking.core.video_io
=============================
Thin, reusable OpenCV video I/O helpers shared by both passes: clip
discovery, a frame-iterating reader (with the metadata analysis needs)
and a writer. Kept free of any detection/drawing logic so both passes
depend on the same primitives instead of duplicating `cv2.VideoCapture`
plumbing.
"""

import os
from typing import Iterator, List, Tuple

import cv2
import numpy as np

from ball_tracking.core import config
from ball_tracking.core.logger import get_logger
from ball_tracking.core.schemas import VideoMeta

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_clips(folder: str) -> List[str]:
    """Return the sorted list of video file paths directly inside *folder*.

    Raises:
        NotADirectoryError: If *folder* does not exist.
    """
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Input clips folder does not exist: {folder}")

    clips = [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if os.path.splitext(name)[1].lower() in config.VIDEO_EXTENSIONS
    ]

    if not clips:
        logger.warning("No video clips found in: %s", folder)
    return clips


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
class VideoReader:
    """Frame-by-frame reader that also exposes the source's `VideoMeta`."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        fps = self._capture.get(cv2.CAP_PROP_FPS) or config.DEFAULT_OUTPUT_FPS
        self.meta = VideoMeta(
            source_path=path,
            fps=fps,
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            total_frames=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_index, frame) for every frame in the video, in order."""
        index = 0
        while True:
            read_ok, frame = self._capture.read()
            if not read_ok:
                break
            yield index, frame
            index += 1

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.release()


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
class VideoWriter:
    """Thin wrapper around `cv2.VideoWriter` using the project's default codec."""

    def __init__(self, path: str, fps: float, width: int, height: int) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_FOURCC)
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.release()
