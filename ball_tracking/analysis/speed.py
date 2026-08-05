"""
ball_tracking.analysis.speed
===============================
Computes one release/peak ball speed for a delivery from its filtered
along-pitch trajectory. A robust linear fit of along-pitch distance vs
frame index gives an averaged velocity that resists the occasional
noisy/misdetected point, converted to km/h and mph using the clip's fps.

The ball is airborne, so projecting it onto the ground plane (what
`PitchCalibration` does) has some parallax error — using the fitted
slope over many points, rather than the raw difference between two
single points, is the main defence against that noise; the residual
bias is a documented limitation, not something this module tries to
correct with a 3-D camera model.
"""

from typing import List, Tuple

from ball_tracking.analysis.filters import TrajectoryPoint
from ball_tracking.core import config
from ball_tracking.core.logger import get_logger
from ball_tracking.core.schemas import SpeedResult

logger = get_logger(__name__)

_INLIER_RESIDUAL_M = 1.0  # drop the worst point and refit while its residual exceeds this
_MIN_POINTS_FOR_FIT = 3


def _fit_line(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Ordinary least squares slope/intercept for y = m*x + c."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        raise ValueError("Cannot fit a line through points with no spread in frame index.")
    m = num / den
    c = mean_y - m * mean_x
    return m, c


def _robust_fit(frame_indices: List[int], x_values: List[float]) -> Tuple[float, float, List[int]]:
    """Iteratively drop the single worst outlier and refit; returns (slope, intercept, kept_indices)."""
    kept = list(range(len(frame_indices)))
    m, c = _fit_line(frame_indices, x_values)

    while len(kept) > _MIN_POINTS_FOR_FIT:
        xs = [frame_indices[i] for i in kept]
        ys = [x_values[i] for i in kept]
        m, c = _fit_line(xs, ys)
        residuals = [abs(x_values[i] - (m * frame_indices[i] + c)) for i in kept]
        worst_pos = max(range(len(kept)), key=lambda k: residuals[k])
        if residuals[worst_pos] <= _INLIER_RESIDUAL_M:
            break
        del kept[worst_pos]

    return m, c, kept


def calculate_speed(trajectory: List[TrajectoryPoint], fps: float) -> SpeedResult:
    """Compute one release/peak speed from a filtered ball trajectory."""
    if len(trajectory) < _MIN_POINTS_FOR_FIT:
        return SpeedResult(valid=False, reason=f"Only {len(trajectory)} ball point(s) tracked — need at least {_MIN_POINTS_FOR_FIT}.")

    frame_indices = [p.frame_index for p in trajectory]
    x_values = [p.x_m for p in trajectory]

    try:
        slope, _, kept = _robust_fit(frame_indices, x_values)
    except ValueError as exc:
        return SpeedResult(valid=False, reason=str(exc))

    if len(kept) < _MIN_POINTS_FOR_FIT:
        return SpeedResult(valid=False, reason="Too few consistent points remained after outlier rejection.")

    kept_frames = [frame_indices[i] for i in kept]
    kept_x = [x_values[i] for i in kept]
    distance_m = max(kept_x) - min(kept_x)
    start_frame, end_frame = min(kept_frames), max(kept_frames)
    elapsed_seconds = (end_frame - start_frame) / fps

    if distance_m < config.BALL_MIN_TRACK_DISTANCE_M:
        return SpeedResult(
            valid=False,
            reason=(
                f"Ball only tracked across {distance_m:.2f}m — below the "
                f"{config.BALL_MIN_TRACK_DISTANCE_M}m minimum for a reliable speed."
            ),
        )
    if elapsed_seconds <= 0:
        return SpeedResult(valid=False, reason="Non-positive elapsed time between tracked points.")

    speed_mps = abs(slope) * fps
    speed_kmh = speed_mps * config.BALL_SPEED_UNIT_CONVERSIONS["kmh"]
    speed_mph = speed_mps * config.BALL_SPEED_UNIT_CONVERSIONS["mph"]

    if speed_kmh > config.BALL_MAX_PLAUSIBLE_SPEED_KMH:
        return SpeedResult(
            valid=False,
            reason=(
                f"Fitted speed {speed_kmh:.0f} km/h exceeds the plausibility ceiling "
                f"({config.BALL_MAX_PLAUSIBLE_SPEED_KMH} km/h) — trajectory is probably still noisy."
            ),
        )

    logger.info(
        "Speed: %.1f km/h (%.1f mph) over %.2fm in %.3fs (frames %d-%d, %d/%d points kept)",
        speed_kmh, speed_mph, distance_m, elapsed_seconds, start_frame, end_frame, len(kept), len(trajectory),
    )
    return SpeedResult(
        valid=True,
        speed_kmh=speed_kmh,
        speed_mph=speed_mph,
        distance_m=distance_m,
        elapsed_seconds=elapsed_seconds,
        start_frame=start_frame,
        end_frame=end_frame,
    )
