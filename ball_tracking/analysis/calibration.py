"""
ball_tracking.analysis.calibration
=====================================
Turns the two detected stump groups (bowling end = near = large in
frame, batting end = far = small in frame — this camera sits behind
the bowler's stumps looking down the pitch) into a pixel <-> real-world
ground-plane mapping.

Two different, axis-appropriate techniques are used:

- **Along the pitch** (uses stump *height*): the ground line through
  both stump bases and the line through both stump tops (0.711 m up)
  are images of two real, parallel 3-D lines. Their intersection is the
  vanishing point of the pitch-length direction. Combined with the
  known 20.12 m pitch length, this gives a projective (cross-ratio)
  pixel <-> metres map along the centre line.
- **Across the pitch** (uses stump *width*): for this camera placement
  (centred behind the stumps, looking straight down the lane — the
  common case for a fixed practice-net rig), the two stump-group width
  lines land exactly horizontal in the image, i.e. their vanishing
  point is at infinity and can't be intersected. Real lateral distance
  instead scales with a perspective (Möbius) function of along-pitch
  position, fitted from the two known width measurements (0.2286 m at
  each end); a point's lateral offset is its horizontal pixel distance
  from the centre line, divided by that position's interpolated
  pixels-per-metre scale.

This intentionally avoids fitting a general 4-point homography from the
raw stump corners: that quadrilateral is ~0.23 m wide by ~20 m long, so
a homography fit from it would extrapolate pixel noise in the short
(lateral) direction by an order of magnitude once used to draw the
~3 m-wide pitch overlay. Both techniques above only ever use the stump
width/height as a *scale factor* or a *line direction*, never as raw
corner positions to extrapolate from.
"""

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ball_tracking.core import config
from ball_tracking.core.logger import get_logger
from ball_tracking.core.schemas import BoundingBox, DetectionRun

logger = get_logger(__name__)

Point = Tuple[float, float]

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Generic 2-D line / projective helpers
# --------------------------------------------------------------------------- #
def line_intersection(a1: Point, a2: Point, b1: Point, b2: Point):
    """Return the intersection of line (a1,a2) with line (b1,b2), or None if parallel."""
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < _EPS:
        return None

    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    px = ((x1 - x2) * b - (x3 - x4) * a) / denom
    py = ((y1 - y2) * b - (y3 - y4) * a) / denom
    return (px, py)


def signed_param(point: Point, origin: Point, through: Point) -> float:
    """Signed distance of *point* from *origin*, measured along the line origin->through.

    `origin` always maps to 0.0 and `through` always maps to +||through-origin||.
    The three points are assumed collinear (or nearly so).
    """
    ox, oy = origin
    tx, ty = through
    dx, dy = tx - ox, ty - oy
    length = math.hypot(dx, dy)
    if length < _EPS:
        raise ValueError("Degenerate reference line: origin and through-point coincide.")
    ux, uy = dx / length, dy / length
    return (point[0] - ox) * ux + (point[1] - oy) * uy


def point_at_param(origin: Point, through: Point, t: float) -> Point:
    """Inverse of `signed_param`: the point at distance *t* from origin, towards through."""
    ox, oy = origin
    tx, ty = through
    length = math.hypot(tx - ox, ty - oy)
    ux, uy = (tx - ox) / length, (ty - oy) / length
    return (ox + ux * t, oy + uy * t)


def real_from_param(t: float, t_a: float, x_a: float, t_b: float, x_b: float, t_inf: float) -> float:
    """Invert a 1-D projective (cross-ratio) map: pixel-line param `t` -> real value.

    Calibrated so param `t_a` <-> real value `x_a`, param `t_b` <-> real value
    `x_b`, and param `t_inf` is the vanishing point (maps to real infinity).
    """
    denom = (t - t_inf) * (t_a - t_b)
    if abs(denom) < _EPS:
        return math.copysign(math.inf, t_a - t_b) if (t - t_inf) == 0 else x_b
    return x_b + (x_a - x_b) * (t - t_b) * (t_a - t_inf) / denom


def param_from_real(x: float, t_a: float, x_a: float, t_b: float, x_b: float, t_inf: float) -> float:
    """Forward direction of `real_from_param`: real value `x` -> pixel-line param `t`."""
    if abs(x_a - x_b) < _EPS:
        raise ValueError("Degenerate metric reference: x_a and x_b must differ.")
    k = (x - x_b) * (t_a - t_b) / (x_a - x_b)
    denom = k - t_a + t_inf
    if abs(denom) < _EPS:
        raise ValueError("Degenerate projective solve (point maps to the vanishing point).")
    return (k * t_inf - t_b * (t_a - t_inf)) / denom


def fit_mobius_scale(x0: float, scale0: float, x1: float, scale1: float):
    """Fit scale(x) = A / (B - x) from two known (position, scale) samples.

    Models how pixels-per-metre for a fixed real length varies with
    along-pitch position, which for a pinhole camera is exactly this
    Möbius form (magnification is proportional to 1 / depth, and depth is
    an affine function of along-pitch position).
    """
    if abs(scale1 - scale0) < _EPS:
        # No detectable perspective change between the two samples - treat as
        # a constant scale (B effectively at infinity).
        return (scale0, math.inf)
    b = (scale1 * x1 - scale0 * x0) / (scale1 - scale0)
    a = scale0 * (b - x0)
    return (a, b)


def eval_mobius_scale(x: float, a: float, b: float) -> float:
    if math.isinf(b):
        return a
    denom = b - x
    if abs(denom) < _EPS:
        raise ValueError(f"Position x={x} coincides with the fitted scale's singular point.")
    return a / denom


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
    half_width_px: float


class PitchCalibration:
    """Pixel <-> real-world (along-pitch, lateral) mapping built from both wickets.

    Coordinate convention: X is metres from the batting end (the far /
    smaller stumps) towards the bowling end; Y is signed lateral metres
    from the pitch centre line.
    """

    def __init__(self, far: _StumpReference, near: _StumpReference) -> None:
        self._far = far
        self._near = near

        v_along = line_intersection(far.base_center, near.base_center, far.top_center, near.top_center)
        if v_along is None:
            raise ValueError(
                "Could not compute the along-pitch vanishing point — the stump "
                "base line and top line are parallel in pixel space, which "
                "shouldn't happen for a real perspective shot."
            )
        self._v_along = v_along

        self._t_far_base = 0.0
        self._t_near_base = signed_param(near.base_center, far.base_center, near.base_center)
        self._t_v_along = signed_param(v_along, far.base_center, near.base_center)

        half_w_m = config.STUMP_GROUP_WIDTH_M / 2.0
        scale_far = far.half_width_px / half_w_m
        scale_near = near.half_width_px / half_w_m
        self._lateral_scale_a, self._lateral_scale_b = fit_mobius_scale(0.0, scale_far, config.PITCH_LENGTH_M, scale_near)

        logger.info(
            "Pitch calibration built: v_along=%s far_base=%s near_base=%s "
            "lateral_scale(px/m): far=%.1f near=%.1f",
            _fmt_point(v_along), _fmt_point(far.base_center), _fmt_point(near.base_center),
            scale_far, scale_near,
        )

    def _centerline_x_at_y(self, y: float) -> float:
        """x-coordinate of the (near-vertical) centre line at image row *y*, extrapolated if needed."""
        fx, fy = self._far.base_center
        nx, ny = self._near.base_center
        if abs(ny - fy) < _EPS:
            raise ValueError("Degenerate centre line: far and near base points share the same image row.")
        return fx + (nx - fx) * (y - fy) / (ny - fy)

    # --- forward: pixel -> pitch metres ------------------------------------ #
    def pixel_to_pitch(self, point: Point) -> Point:
        """Convert a pixel coordinate to (X metres from batting end, Y lateral metres)."""
        px, py = point
        centerline_x = self._centerline_x_at_y(py)
        centerline_point = (centerline_x, py)
        t_x = signed_param(centerline_point, self._far.base_center, self._near.base_center)
        x_m = real_from_param(t_x, self._t_far_base, 0.0, self._t_near_base, config.PITCH_LENGTH_M, self._t_v_along)

        scale = eval_mobius_scale(x_m, self._lateral_scale_a, self._lateral_scale_b)
        y_m = (px - centerline_x) / scale
        return (x_m, y_m)

    # --- inverse: pitch metres -> pixel ------------------------------------ #
    def pitch_to_pixel(self, x_m: float, y_m: float) -> Point:
        """Convert (X metres from batting end, Y lateral metres) to a pixel coordinate."""
        t_x = param_from_real(x_m, self._t_far_base, 0.0, self._t_near_base, config.PITCH_LENGTH_M, self._t_v_along)
        centerline_point = point_at_param(self._far.base_center, self._near.base_center, t_x)

        scale = eval_mobius_scale(x_m, self._lateral_scale_a, self._lateral_scale_b)
        px = centerline_point[0] + y_m * scale
        return (px, centerline_point[1])

    # --- reference geometry, exposed for drawing dummy stumps -------------- #
    @property
    def far_reference(self) -> _StumpReference:
        return self._far

    @property
    def near_reference(self) -> _StumpReference:
        return self._near


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
        half_width_px=bbox.width / 2.0,
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
