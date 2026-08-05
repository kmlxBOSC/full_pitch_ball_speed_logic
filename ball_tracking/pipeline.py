"""
ball_tracking.pipeline
=========================
Top-level 2-pass orchestrator:

  Pass 1 (detection.runner)  -> raw per-frame detections (cached as JSON)
  Analysis (analysis.*)      -> calibrate the pitch, filter/track the ball, compute speed
  Pass 2 (drawing.renderer)  -> re-read the source video and write an annotated copy

Also provides batch (folder) processing, reusing one loaded
`CricketDetectionPipeline` across every clip.
"""

import os
from typing import List, Optional, Tuple

from ball_tracking.analysis.calibration import PitchCalibration, build_calibration
from ball_tracking.analysis.filters import build_ball_trajectory
from ball_tracking.analysis.speed import calculate_speed
from ball_tracking.core.logger import get_logger
from ball_tracking.core.schemas import DetectionRun, SpeedResult
from ball_tracking.core.video_io import VideoReader, VideoWriter, discover_clips
from ball_tracking.detection.pipeline import CricketDetectionPipeline
from ball_tracking.detection.runner import cache_path_for, run_detection_pass
from ball_tracking.drawing.renderer import render_frame

logger = get_logger(__name__)


def analyze(run: DetectionRun) -> Tuple[Optional[PitchCalibration], SpeedResult]:
    """Calibrate the pitch, build the ball trajectory and compute speed for one `DetectionRun`.

    Degrades gracefully: if calibration fails (e.g. only one wicket ever
    visible), returns `(None, SpeedResult(valid=False, ...))` rather than
    raising — pass 2 still draws raw detections without the pitch overlay.
    """
    try:
        calibration = build_calibration(run)
    except ValueError as exc:
        logger.warning("Pitch calibration unavailable: %s", exc)
        return None, SpeedResult(valid=False, reason=f"Calibration failed: {exc}")

    trajectory = build_ball_trajectory(run, calibration)
    speed = calculate_speed(trajectory, run.video.fps)
    if not speed.valid:
        logger.warning("Speed unavailable: %s", speed.reason)
    return calibration, speed


def render(run: DetectionRun, calibration: Optional[PitchCalibration], speed: Optional[SpeedResult], output_path: str) -> None:
    """Pass 2 — re-read the source video and write the annotated copy to *output_path*."""
    detections_by_frame = {f.frame_index: f.detections for f in run.frames}

    with VideoReader(run.video.source_path) as reader, \
         VideoWriter(output_path, run.video.fps, run.video.width, run.video.height) as writer:
        for frame_index, frame in reader.frames():
            detections = detections_by_frame.get(frame_index)
            if detections is None:
                writer.write(frame)
                continue
            writer.write(render_frame(frame, detections, calibration, speed))

    logger.info("Pass 2: wrote annotated video -> %s", output_path)


def process_clip(detector: CricketDetectionPipeline, input_path: str, output_path: str) -> SpeedResult:
    """Run the full 2-pass pipeline on one clip. Returns the computed `SpeedResult`."""
    cache_path = cache_path_for(output_path)
    run = run_detection_pass(detector, input_path, cache_path)
    calibration, speed = analyze(run)
    render(run, calibration, speed, output_path)
    return speed


def process_folder(input_folder: str, output_folder: str) -> List[str]:
    """Run the full 2-pass pipeline on every clip in *input_folder*."""
    clips = discover_clips(input_folder)
    detector = CricketDetectionPipeline()

    output_paths = []
    for clip_path in clips:
        clip_name = os.path.basename(clip_path)
        output_path = os.path.join(output_folder, clip_name)
        speed = process_clip(detector, clip_path, output_path)
        if speed.valid:
            logger.info("%s -> %.1f km/h", clip_name, speed.speed_kmh)
        else:
            logger.info("%s -> speed unavailable (%s)", clip_name, speed.reason)
        output_paths.append(output_path)

    return output_paths
