"""
ball_tracking.core.schemas
============================
Lightweight, immutable data containers shared across detection, analysis
and drawing. Keeping these separate from detector / analysis / drawing
logic means any consumer (API layer, CLI, tests) can depend on a stable
shape without importing ultralytics or OpenCV.

`FrameRecord` / `DetectionRun` also know how to serialise themselves
to/from plain dicts so pass-1 detection output can be cached as JSON and
re-used by the analysis and drawing passes without re-running inference.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in absolute pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        """(x, y) coordinates of the box centre."""
        return (self.x1 + self.width / 2, self.y1 + self.height / 2)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """(x, y) coordinates of the box's bottom-centre (ground contact point)."""
        return (self.x1 + self.width / 2, self.y2)

    @property
    def top_center(self) -> Tuple[float, float]:
        """(x, y) coordinates of the box's top-centre."""
        return (self.x1 + self.width / 2, self.y1)

    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) rounded to int pixel coordinates."""
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def to_dict(self) -> Dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "BoundingBox":
        return cls(x1=data["x1"], y1=data["y1"], x2=data["x2"], y2=data["y2"])


# --------------------------------------------------------------------------- #
# Detections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Detection:
    """A single detected object: class label, confidence and box."""

    class_name: str
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": self.bbox.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        return cls(
            class_name=data["class_name"],
            confidence=data["confidence"],
            bbox=BoundingBox.from_dict(data["bbox"]),
        )


@dataclass(frozen=True)
class Keypoint:
    """A single named pose keypoint (e.g. "left_wrist")."""

    name: str
    x: float
    y: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "x": self.x, "y": self.y, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Keypoint":
        return cls(name=data["name"], x=data["x"], y=data["y"], confidence=data["confidence"])


@dataclass(frozen=True)
class PersonDetection(Detection):
    """A detected person, extended with pose keypoints."""

    keypoints: Tuple[Keypoint, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["keypoints"] = [kp.to_dict() for kp in self.keypoints]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonDetection":
        return cls(
            class_name=data["class_name"],
            confidence=data["confidence"],
            bbox=BoundingBox.from_dict(data["bbox"]),
            keypoints=tuple(Keypoint.from_dict(kp) for kp in data.get("keypoints", [])),
        )


# --------------------------------------------------------------------------- #
# Aggregated frame result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrameDetections:
    """All detections found in a single frame, grouped by object type."""

    balls: Tuple[Detection, ...] = field(default_factory=tuple)
    stumps: Tuple[Detection, ...] = field(default_factory=tuple)
    humans: Tuple[PersonDetection, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "balls": [d.to_dict() for d in self.balls],
            "stumps": [d.to_dict() for d in self.stumps],
            "humans": [h.to_dict() for h in self.humans],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameDetections":
        return cls(
            balls=tuple(Detection.from_dict(d) for d in data.get("balls", [])),
            stumps=tuple(Detection.from_dict(d) for d in data.get("stumps", [])),
            humans=tuple(PersonDetection.from_dict(d) for d in data.get("humans", [])),
        )


# --------------------------------------------------------------------------- #
# Pass-1 (detection-only) output for a whole clip
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrameRecord:
    """Detections for one frame, tagged with its index in the source video."""

    frame_index: int
    detections: FrameDetections

    def to_dict(self) -> Dict[str, Any]:
        return {"frame_index": self.frame_index, "detections": self.detections.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameRecord":
        return cls(
            frame_index=data["frame_index"],
            detections=FrameDetections.from_dict(data["detections"]),
        )


@dataclass(frozen=True)
class VideoMeta:
    """Source video properties needed by analysis (timing) and drawing (canvas size)."""

    source_path: str
    fps: float
    width: int
    height: int
    total_frames: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoMeta":
        return cls(
            source_path=data["source_path"],
            fps=data["fps"],
            width=data["width"],
            height=data["height"],
            total_frames=data["total_frames"],
        )


@dataclass(frozen=True)
class DetectionRun:
    """Complete pass-1 (detection-only) output for one clip: metadata + every frame's detections."""

    video: VideoMeta
    frames: Tuple[FrameRecord, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video": self.video.to_dict(),
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionRun":
        return cls(
            video=VideoMeta.from_dict(data["video"]),
            frames=tuple(FrameRecord.from_dict(f) for f in data.get("frames", [])),
        )


# --------------------------------------------------------------------------- #
# Analysis (pass between pass 1 and pass 2) results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpeedResult:
    """Computed release/peak ball speed for one delivery, or the reason it failed."""

    valid: bool
    speed_kmh: Optional[float] = None
    speed_mph: Optional[float] = None
    distance_m: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "speed_kmh": self.speed_kmh,
            "speed_mph": self.speed_mph,
            "distance_m": self.distance_m,
            "elapsed_seconds": self.elapsed_seconds,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "reason": self.reason,
        }
