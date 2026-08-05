"""
ball_tracking.analysis.filters
=================================
Cleans raw pass-1 ball detections into a trustworthy trajectory: picks
at most one ball per frame (highest confidence), converts each to pitch
metres via `PitchCalibration`, and rejects points that land well outside
the pitch or imply an impossible speed jump from the previous accepted
point (noise / false positives from the bat, crowd, etc.).
"""

from dataclasses import dataclass
from typing import Iterator, List, Tuple

from ball_tracking.analysis.calibration import PitchCalibration
from ball_tracking.core import config
from ball_tracking.core.logger import get_logger
from ball_tracking.core.schemas import Detection, DetectionRun

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrajectoryPoint:
    """One accepted ball observation, in both pixel and pitch (metre) space."""

    frame_index: int
    pixel: Tuple[float, float]
    x_m: float
    y_m: float
    confidence: float


def _best_ball_per_frame(run: DetectionRun) -> Iterator[Tuple[int, Detection]]:
    """Yield (frame_index, detection) for the highest-confidence ball in each frame that has one."""
    for frame in run.frames:
        if not frame.detections.balls:
            continue
        best = max(frame.detections.balls, key=lambda d: d.confidence)
        yield frame.frame_index, best


def build_ball_trajectory(run: DetectionRun, calibration: PitchCalibration) -> List[TrajectoryPoint]:
    """Return the filtered, time-ordered ball trajectory for *run*."""
    fps = run.video.fps
    max_speed_mps = config.BALL_MAX_PLAUSIBLE_SPEED_KMH / config.BALL_SPEED_UNIT_CONVERSIONS["kmh"]

    margin = config.PITCH_LENGTH_M * 0.15
    lo, hi = -margin, config.PITCH_LENGTH_M + margin

    trajectory: List[TrajectoryPoint] = []
    for frame_index, detection in _best_ball_per_frame(run):
        center = detection.bbox.center
        try:
            x_m, y_m = calibration.pixel_to_pitch(center)
        except ValueError:
            continue

        if not (lo <= x_m <= hi):
            logger.debug("Rejected ball at frame %d: X=%.2fm outside pitch bounds", frame_index, x_m)
            continue

        if trajectory:
            prev = trajectory[-1]
            dt = (frame_index - prev.frame_index) / fps
            if dt <= 0:
                continue
            implied_speed = abs(x_m - prev.x_m) / dt
            if implied_speed > max_speed_mps:
                logger.debug(
                    "Rejected ball at frame %d: implied speed %.0f km/h since frame %d exceeds ceiling",
                    frame_index, implied_speed * config.BALL_SPEED_UNIT_CONVERSIONS["kmh"], prev.frame_index,
                )
                continue

        trajectory.append(TrajectoryPoint(frame_index, center, x_m, y_m, detection.confidence))

    logger.info("Ball trajectory: %d accepted point(s) out of %d frame(s)", len(trajectory), len(run.frames))
    return trajectory
