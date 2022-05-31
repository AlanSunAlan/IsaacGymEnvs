from isaacgym import gymtorch
from isaacgym import gymapi

from tasks.base.vec_task import VecTask

import os, sys

FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(FILE_DIR)


ASSET_FOLDER = os.path.join(ROOT_DIR, "assets")
URDF_FOLDER = os.path.join(ASSET_FOLDER, "quadfinger_ik")
finger_xacro_file = os.path.join(URDF_FOLDER, "quad_finger.urdf.xacro")
