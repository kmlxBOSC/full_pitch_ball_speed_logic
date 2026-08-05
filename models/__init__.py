"""models — pre-loaded YOLO model instances for ball and bat detection."""

import os

from models.infer import Inference

_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

obj_ball: Inference = Inference()
obj_bat: Inference = Inference()
obj_stump: Inference = Inference()
obj_human: Inference = Inference()

obj_ball.load_model(os.path.join(_WEIGHTS_DIR, "ball_detection.pt"))
obj_bat.load_model(os.path.join(_WEIGHTS_DIR, "bat_detection.pt"))
obj_stump.load_model(os.path.join(_WEIGHTS_DIR, "stump_detection.pt"))
obj_human.load_model(os.path.join(_WEIGHTS_DIR, "pose_detection.pt"))
