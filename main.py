"""
main
=====
Single entry point to run the full 2-pass ball_tracking pipeline over
every clip in the configured input folder
(`ball_tracking.core.config.INPUT_CLIPS_DIR`):

    Pass 1: detect balls/stumps/humans only (cached as JSON)
    Analysis: calibrate the pitch, filter the ball trajectory, compute speed
    Pass 2: draw the pitch overlay + detections + speed HUD and write the video

Each clip's annotated result is written to:

    outputs/<today's date>/<input folder name>/<clip filename>

Usage:
    python main.py
"""

import os
from datetime import date

from ball_tracking.core import config
from ball_tracking.core.logger import get_logger
from ball_tracking.pipeline import process_folder

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Output path resolution
# --------------------------------------------------------------------------- #
def _build_output_folder() -> str:
    """Resolve this run's output folder: OUTPUT_ROOT_DIR/<today's date>/<input folder name>."""
    today = date.today().strftime(config.OUTPUT_DATE_FORMAT)
    input_folder_name = os.path.basename(os.path.normpath(config.INPUT_CLIPS_DIR))
    return os.path.join(config.OUTPUT_ROOT_DIR, today, input_folder_name)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    output_folder = _build_output_folder()
    os.makedirs(output_folder, exist_ok=True)

    logger.info("Input folder: %s", config.INPUT_CLIPS_DIR)
    logger.info("Output folder: %s", output_folder)

    output_paths = process_folder(config.INPUT_CLIPS_DIR, output_folder)

    logger.info("Done. %d clip(s) processed.", len(output_paths))


if __name__ == "__main__":
    main()
