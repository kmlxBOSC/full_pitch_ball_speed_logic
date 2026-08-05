"""
ball_tracking.detection.runner
=================================
Pass 1 — detection only. Reads a video frame-by-frame, runs
`CricketDetectionPipeline` on every frame, and returns a `DetectionRun`
(video metadata + every frame's raw detections). No drawing happens
here, and none of pass 1's own code depends on OpenCV drawing or on the
analysis (calibration/speed) layer.

Results are cached as JSON next to the eventual output video
(`config.DETECTION_CACHE_SUFFIX`) so re-running calibration, filtering,
speed calculation or drawing never has to pay for inference again.
"""

import json
import os

from ball_tracking.core import config
from ball_tracking.core.logger import get_logger
from ball_tracking.core.schemas import DetectionRun, FrameRecord
from ball_tracking.core.video_io import VideoReader
from ball_tracking.detection.pipeline import CricketDetectionPipeline

logger = get_logger(__name__)


def cache_path_for(output_video_path: str) -> str:
    """Return the JSON cache path a given output video path's detections live at."""
    return output_video_path + config.DETECTION_CACHE_SUFFIX


def load_cached_run(cache_path: str) -> DetectionRun:
    with open(cache_path, "r", encoding="utf-8") as fh:
        return DetectionRun.from_dict(json.load(fh))


def save_run(run: DetectionRun, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(run.to_dict(), fh)


def run_detection_pass(
    pipeline: CricketDetectionPipeline,
    input_path: str,
    cache_path: str,
) -> DetectionRun:
    """Run pass 1 on *input_path*, returning a `DetectionRun`.

    If `config.USE_DETECTION_CACHE` is enabled and a cache already exists
    at *cache_path*, inference is skipped entirely and the cached result
    is loaded instead.
    """
    if config.USE_DETECTION_CACHE and os.path.exists(cache_path):
        logger.info("Pass 1: using cached detections for %s -> %s", input_path, cache_path)
        return load_cached_run(cache_path)

    logger.info("Pass 1: running detection on %s", input_path)
    frames = []
    with VideoReader(input_path) as reader:
        for frame_index, frame in reader.frames():
            detections = pipeline.detect(frame)
            frames.append(FrameRecord(frame_index=frame_index, detections=detections))
        video_meta = reader.meta

    run = DetectionRun(video=video_meta, frames=tuple(frames))
    logger.info("Pass 1: finished %s (%d frames)", input_path, len(frames))

    if config.USE_DETECTION_CACHE:
        save_run(run, cache_path)
        logger.info("Pass 1: cached detections -> %s", cache_path)

    return run
