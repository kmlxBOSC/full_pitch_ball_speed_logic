"""
ball_tracking.analysis.calibration
=====================================
Turns the two detected stump groups (bowling end = near = large in
frame, batting end = far = small in frame — this camera sits behind
the bowler's stumps looking down the pitch) into a pixel <-> real-world
ground-plane mapping, via a single planar homography.

The prepared pitch strip is a flat rectangle lying on the ground, so
the true pixel <-> real-world mapping for every point on it is
*exactly* one projective transform (homography) — a single consistent
model for both along-pitch and lateral directions, rather than two
separately-fitted approximations. It's fully determined by four point
correspondences: the two real corners of the pitch boundary
(+-1.525 m from the centre line) at each wicket, in both their known
metric ground coordinates and their pixel positions.

Those four pixel corners aren't directly observable (only the stumps
are detected) — they're estimated the same way the lateral scale
always was here: each wicket's own stump height in pixels, divided by
the real 0.711 m stump height, gives that end's local pixels-per-metre
scale (independent of the other end, no shared vanishing point
needed); the real 3.05 m pitch width scaled by that factor, projected
outward from the stump base along the image-plane perpendicular to the
(straight) centre-line direction, gives that end's two boundary
corners.

Once those four correspondences are known, `cv2.getPerspectiveTransform`
solves the exact 3x3 homography (8 degrees of freedom = 4 point
pairs), and every other pitch marking — length markers, creases, wide
guidelines, return creases — is just a `pitch_to_pixel` call on its
known metric coordinates. Length-marker spacing shrinks correctly
(non-linearly) toward the horizon for free, because that is what a
homography does to equally-spaced collinear points; no separate
cross-ratio machinery is needed for that.
"""

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from ball_tracking.core import config
from ball_tracking.core.logger import get_logger
from ball_tracking.core.schemas import BoundingBox, DetectionRun

logger = get_logger(__name__)

Point = Tuple[float, float]

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Pitch calibration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _StumpReference:
    """Pixel-space anchor points of one detected stump group (one wicket).

    `bbox_left`/`bbox_right` are the detected box's left/right edges at its
    base — purely positional (whichever has the smaller pixel x), unrelated
    to the calibration's signed Y convention.
    """

    bbox_left: Point
    bbox_right: Point
    base_center: Point
    top_center: Point
    height_px: float


class PitchCalibration:
    """Pixel <-> real-world (along-pitch, lateral) ground-plane mapping, built from both wickets via a single homography.

    Coordinate convention: X is metres from the batting end (the far /
    smaller stumps) towards the bowling end; Y is signed lateral metres
    from the pitch centre line, positive towards the near wicket's own
    detected left edge.
    """

    def __init__(self, far: _StumpReference, near: _StumpReference) -> None:
        self._far = far
        self._near = near

        scale_far = far.height_px / config.STUMP_HEIGHT_M
        scale_near = near.height_px / config.STUMP_HEIGHT_M

        half_pitch_w_m = config.PITCH_WIDTH_M / 2.0
        far_half_width_px = half_pitch_w_m * scale_far
        near_half_width_px = half_pitch_w_m * scale_near

        lateral_unit = _perpendicular_unit(far, near)
        far_left_px = _offset(far.base_center, lateral_unit, far_half_width_px)
        far_right_px = _offset(far.base_center, lateral_unit, -far_half_width_px)
        near_left_px = _offset(near.base_center, lateral_unit, near_half_width_px)
        near_right_px = _offset(near.base_center, lateral_unit, -near_half_width_px)

        # Ground-truth (X metres, Y metres) <-> estimated pixel corners of the
        # real pitch rectangle, in matching order, for the homography solve.
        world_pts = np.array(
            [
                [0.0, half_pitch_w_m],
                [0.0, -half_pitch_w_m],
                [config.PITCH_LENGTH_M, half_pitch_w_m],
                [config.PITCH_LENGTH_M, -half_pitch_w_m],
            ],
            dtype=np.float32,
        )
        pixel_pts = np.array([far_left_px, far_right_px, near_left_px, near_right_px], dtype=np.float32)

        try:
            self._to_pixel = cv2.getPerspectiveTransform(world_pts, pixel_pts)
            self._to_world = cv2.getPerspectiveTransform(pixel_pts, world_pts)
        except cv2.error as exc:
            raise ValueError(
                "Could not compute the pitch homography — the four estimated "
                "boundary corners are degenerate (collinear or coincident)."
            ) from exc

        logger.info(
            "Pitch calibration built (homography): far_base=%s near_base=%s "
            "height_scale(px/m): far=%.1f near=%.1f pitch_half_width(px): far=%.1f near=%.1f",
            _fmt_point(far.base_center), _fmt_point(near.base_center),
            scale_far, scale_near, far_half_width_px, near_half_width_px,
        )

    # --- forward: pitch metres -> pixel ------------------------------------ #
    def pitch_to_pixel(self, x_m: float, y_m: float) -> Point:
        """Convert (X metres from batting end, Y lateral metres) to a pixel coordinate."""
        return _apply_homography(self._to_pixel, (x_m, y_m))

    # --- inverse: pixel -> pitch metres ------------------------------------ #
    def pixel_to_pitch(self, point: Point) -> Point:
        """Convert a pixel coordinate to (X metres from batting end, Y lateral metres)."""
        return _apply_homography(self._to_world, point)

    # --- pitch boundary: straight-line projection --------------------------- #
    def boundary_points_at(self, x_m: float) -> Tuple[Point, Point]:
        """Return the (left, right) pixel points of the real pitch boundary (10 ft strip) at along-pitch position *x_m*.

        Every real pitch marking is a straight line in the world, and a
        homography always maps a straight line to a straight line, so
        callers connect these (and any other `pitch_to_pixel` pair) with a
        single straight segment — no curve-sampling is ever needed.
        """
        half_w = config.PITCH_WIDTH_M / 2.0
        return self.pitch_to_pixel(x_m, half_w), self.pitch_to_pixel(x_m, -half_w)

    # --- reference geometry, exposed for drawing dummy stumps -------------- #
    @property
    def far_reference(self) -> _StumpReference:
        return self._far

    @property
    def near_reference(self) -> _StumpReference:
        return self._near


def _apply_homography(matrix: np.ndarray, point: Point) -> Point:
    x, y = point
    px, py, w = matrix @ np.array([x, y, 1.0])
    if abs(w) < _EPS:
        raise ValueError(f"Point {point} maps to infinity under this homography.")
    return (px / w, py / w)


def _offset(point: Point, unit: Point, distance: float) -> Point:
    return (point[0] + unit[0] * distance, point[1] + unit[1] * distance)


def _perpendicular_unit(far: "_StumpReference", near: "_StumpReference") -> Point:
    """Unit vector in the image plane, perpendicular to the centre-line direction, pointing towards the "left" (bbox_left) side.

    The real centre line is straight, so its image is a single straight
    line; rotating that direction by 90 degrees gives the direction each
    wicket's pitch-boundary corners are estimated along. The sign is fixed
    using the near wicket's own left/right box edges (the larger, less
    noisy box) so "left"/"right" stay consistent with the detected stump
    orientation.
    """
    fx, fy = far.base_center
    nx, ny = near.base_center
    dx, dy = nx - fx, ny - fy
    length = math.hypot(dx, dy)
    if length < _EPS:
        raise ValueError("Degenerate centre line: far and near base points coincide.")
    ux, uy = dx / length, dy / length
    perp = (-uy, ux)

    lx, ly = near.bbox_left[0] - nx, near.bbox_left[1] - ny
    if perp[0] * lx + perp[1] * ly < 0:
        perp = (-perp[0], -perp[1])
    return perp


def _fmt_point(point: Point) -> str:
    return f"({point[0]:.1f}, {point[1]:.1f})"


def _median_bbox(boxes: Sequence[BoundingBox]) -> BoundingBox:
    xs1 = sorted(b.x1 for b in boxes)
    ys1 = sorted(b.y1 for b in boxes)
    xs2 = sorted(b.x2 for b in boxes)
    ys2 = sorted(b.y2 for b in boxes)
    mid = len(boxes) // 2

    def _median(values: List[float]) -> float:
        if len(values) % 2 == 1:
            return values[mid]
        return (values[mid - 1] + values[mid]) / 2.0

    return BoundingBox(_median(xs1), _median(ys1), _median(xs2), _median(ys2))


def _bbox_to_reference(bbox: BoundingBox) -> _StumpReference:
    return _StumpReference(
        bbox_left=(bbox.x1, bbox.y2),
        bbox_right=(bbox.x2, bbox.y2),
        base_center=bbox.bottom_center,
        top_center=bbox.top_center,
        height_px=bbox.height,
    )


def build_calibration(run: DetectionRun) -> PitchCalibration:
    """Aggregate stump detections across every frame of *run* into a `PitchCalibration`.

    Stumps are static, so all stump boxes across the whole clip are pooled
    and split into the "near" (bowling end, larger in frame) and "far"
    (batting end, smaller in frame) groups by a gap search on box height,
    then each group's box is taken as the per-coordinate median across its
    samples for a stable, jitter-resistant reference.

    Raises:
        ValueError: If too few confident stump detections are available, or
            only one wicket was ever seen (both ends must be visible to
            calibrate).
    """
    all_boxes = [d.bbox for frame in run.frames for d in frame.detections.stumps]
    if len(all_boxes) < 2 * config.MIN_STUMP_SAMPLES_FOR_CALIBRATION:
        raise ValueError(
            f"Not enough stump detections to calibrate ({len(all_boxes)} found across "
            f"{len(run.frames)} frames; need both wickets seen at least "
            f"{config.MIN_STUMP_SAMPLES_FOR_CALIBRATION} times each)."
        )

    heights = sorted(b.height for b in all_boxes)
    split_idx = max(range(1, len(heights)), key=lambda i: heights[i] - heights[i - 1])
    gap = heights[split_idx] - heights[split_idx - 1]
    spread = heights[-1] - heights[0]
    if spread < _EPS or gap < 0.25 * spread:
        raise ValueError(
            "Stump detections don't split into two clearly separated size groups — "
            "only one wicket (end of the pitch) appears to be visible in this clip, "
            "so calibration needs both ends and cannot proceed."
        )
    threshold = (heights[split_idx - 1] + heights[split_idx]) / 2.0

    near_boxes = [b for b in all_boxes if b.height >= threshold]
    far_boxes = [b for b in all_boxes if b.height < threshold]
    if len(near_boxes) < config.MIN_STUMP_SAMPLES_FOR_CALIBRATION or len(far_boxes) < config.MIN_STUMP_SAMPLES_FOR_CALIBRATION:
        raise ValueError(
            f"Not enough samples in each wicket group after clustering "
            f"(near={len(near_boxes)}, far={len(far_boxes)})."
        )

    near_ref = _bbox_to_reference(_median_bbox(near_boxes))
    far_ref = _bbox_to_reference(_median_bbox(far_boxes))
    logger.info(
        "Stump calibration samples: near(bowling end)=%d far(batting end)=%d",
        len(near_boxes), len(far_boxes),
    )
    return PitchCalibration(far=far_ref, near=near_ref)
