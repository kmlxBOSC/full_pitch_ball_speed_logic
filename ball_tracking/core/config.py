"""
ball_tracking.core.config
===========================
Centralised configuration for the ball_tracking package.

Every tunable value used across logging, detection, pitch calibration,
speed calculation and drawing lives here so behaviour can be changed in
one place without touching detector, analysis or drawing logic.
"""

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Folder containing the input clips (videos) to run detection on in bulk.
INPUT_CLIPS_DIR = r"E:\full_pitch_clips"

# Root folder for annotated output videos. A run's results are written to:
#   OUTPUT_ROOT_DIR/<today's date>/<input folder name>/<clip filename>
OUTPUT_ROOT_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUTPUT_DATE_FORMAT = "%d-%m-%Y"

# Pass-1 raw detections are cached as JSON next to the annotated video so
# calibration / filtering / speed / drawing can be re-run repeatedly without
# paying for inference again. Set to False to always re-run detection.
USE_DETECTION_CACHE = True
DETECTION_CACHE_SUFFIX = ".detections.json"

# --------------------------------------------------------------------------- #
# Video I/O
# --------------------------------------------------------------------------- #
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
OUTPUT_VIDEO_FOURCC = "mp4v"  # codec used when writing annotated output videos
DEFAULT_OUTPUT_FPS = 30.0     # fallback if a source video reports no/invalid FPS

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
ROOT_LOGGER_NAME = "ball_tracking"

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "ball_tracking.log")

LOG_MAX_BYTES = 5 * 1024 * 1024  # rotate after 5 MB
LOG_BACKUP_COUNT = 5

LOG_FILE_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_CONSOLE_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FMT = "%d-%m-%Y %H:%M:%S"

# --------------------------------------------------------------------------- #
# Model / class identity
# --------------------------------------------------------------------------- #
# Class names below must match the label each YOLO weight file was trained
# with (see `model.names` on the corresponding `ultralytics.YOLO` instance).
BALL_CLASS_NAME = "ball"
STUMP_CLASS_NAME = "stump"
HUMAN_CLASS_NAME = "person"

# --------------------------------------------------------------------------- #
# Inference thresholds
# --------------------------------------------------------------------------- #
INFERENCE_IMAGE_SIZE = 640  # long-side pixels fed to every YOLO model

BALL_CONF_THRESHOLD = 0.25
BALL_IOU_THRESHOLD = 0.45

STUMP_CONF_THRESHOLD = 0.35
STUMP_IOU_THRESHOLD = 0.45

HUMAN_CONF_THRESHOLD = 0.40
HUMAN_IOU_THRESHOLD = 0.45
HUMAN_KEYPOINT_CONF_THRESHOLD = 0.50  # per-keypoint visibility cutoff for drawing

# --------------------------------------------------------------------------- #
# SAHI (Slicing Aided Hyper Inference)
# --------------------------------------------------------------------------- #
# Tiles a frame into overlapping slices and runs detection on each tile
# before merging boxes back to full-frame coordinates, which substantially
# improves recall on small objects (the ball, the stumps) that would
# otherwise shrink to only a few pixels once the whole frame is resized
# down to `INFERENCE_IMAGE_SIZE`.
#
# Cost warning: each slice is a full extra forward pass through the model,
# so slicing multiplies inference time per detector (roughly x(slices),
# +1 more if SAHI_PERFORM_STANDARD_PRED is enabled). On CPU this is
# significant — keep it enabled only for detectors that need it.
BALL_USE_SAHI = True
STUMP_USE_SAHI = True
HUMAN_USE_SAHI = False  # pose keypoints aren't produced by SAHI's slice merge

SAHI_SLICE_HEIGHT = 640
SAHI_SLICE_WIDTH = 640
SAHI_OVERLAP_HEIGHT_RATIO = 0.2
SAHI_OVERLAP_WIDTH_RATIO = 0.2

# Also run one un-sliced full-frame pass and merge it in, so objects large
# enough to not need slicing (or that straddle a slice boundary) are still
# reliably picked up.
SAHI_PERFORM_STANDARD_PRED = True

# How overlapping per-slice predictions are merged back into one detection.
SAHI_POSTPROCESS_TYPE = "GREEDYNMM"       # "NMM", "GREEDYNMM" or "NMS"
SAHI_POSTPROCESS_MATCH_METRIC = "IOS"     # "IOU" or "IOS" (intersection over smaller area)
SAHI_POSTPROCESS_MATCH_THRESHOLD = 0.5

# --------------------------------------------------------------------------- #
# COCO 17-point skeleton (used by the human / pose model)
# --------------------------------------------------------------------------- #
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Pairs of keypoint indices connected when drawing the skeleton.
COCO_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),             # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),    # arms
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # legs
]

# --------------------------------------------------------------------------- #
# Cricket pitch real-world dimensions (metres)
# --------------------------------------------------------------------------- #
# These are standard regulation dimensions. If the stumps used in a clip are
# non-regulation (e.g. coaching/practice stumps), override them here.
STUMP_HEIGHT_M = 0.711            # 28 in, ground to top of stump
STUMP_GROUP_WIDTH_M = 0.2286      # 9 in, width spanning all 3 stumps of one wicket
PITCH_LENGTH_M = 20.12            # 22 yd, between the two sets of stumps
POPPING_CREASE_OFFSET_M = 1.22    # 4 ft, in front of each set of stumps
PITCH_WIDTH_M = 3.05              # 10 ft, prepared pitch strip width ("wide" side lines)

# Distance markers drawn along the pitch centre line, measured in metres
# from the batting end (the far / smaller stumps in frame).
PITCH_DISTANCE_MARKERS_M = (2.0, 4.0, 6.0, 8.0)

# Minimum number of frames in which a stump group must be detected before
# it is trusted as a calibration reference (stumps are static, so a few
# confident detections are enough — no need to see them in every frame).
MIN_STUMP_SAMPLES_FOR_CALIBRATION = 3

# --------------------------------------------------------------------------- #
# Ball speed calculation
# --------------------------------------------------------------------------- #
# Ball detections are rejected as noise if the implied along-pitch speed
# between consecutive accepted frames exceeds this (generous) ceiling.
BALL_MAX_PLAUSIBLE_SPEED_KMH = 140.0

# A speed result needs the ball tracked across at least this many
# along-pitch metres to be considered reliable (too short a baseline makes
# the frame-count/fps timing noisy).
BALL_MIN_TRACK_DISTANCE_M = 3.0

BALL_SPEED_UNIT_CONVERSIONS = {
    "kmh": 3.6,
    "mph": 2.23694,
}

# --------------------------------------------------------------------------- #
# Visualisation (BGR tuples — OpenCV convention)
# --------------------------------------------------------------------------- #
BALL_BOX_COLOR = (0, 215, 255)
STUMP_BOX_COLOR = (0, 255, 0)
HUMAN_BOX_COLOR = (255, 128, 0)
SKELETON_COLOR = (0, 165, 255)
KEYPOINT_COLOR = (0, 0, 255)

BOX_THICKNESS = 2
KEYPOINT_RADIUS = 4
SKELETON_THICKNESS = 2
FONT_SCALE = 0.5
FONT_THICKNESS = 1

# Pitch overlay (calibration / broadcast-style graphic)
PITCH_SIDE_LINE_COLOR = (255, 255, 255)
PITCH_CREASE_LINE_COLOR = (255, 255, 255)
PITCH_CENTER_LINE_COLOR = (0, 255, 255)
PITCH_MARKER_LINE_COLOR = (0, 200, 255)
PITCH_MARKER_TEXT_COLOR = (0, 200, 255)
DUMMY_STUMP_COLOR = (0, 0, 255)
PITCH_OVERLAY_LINE_THICKNESS = 1
PITCH_MARKER_LINE_THICKNESS = 1
DUMMY_STUMP_THICKNESS = 2

# Speed HUD (headline release/peak speed overlay)
SPEED_HUD_TEXT_COLOR = (255, 255, 255)
SPEED_HUD_BG_COLOR = (0, 0, 0)
SPEED_HUD_FONT_SCALE = 1.0
SPEED_HUD_FONT_THICKNESS = 2
SPEED_HUD_UNIT = "kmh"  # "kmh" or "mph"
