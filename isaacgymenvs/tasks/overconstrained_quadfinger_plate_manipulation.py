import os, sys
from types import SimpleNamespace
from typing import Tuple
import numpy as np

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import *

from tasks.module.single_plate_manipulation_task import SinglePlateManipulationTask

class OverconstrainedQuadfingerPlateManipulation(SinglePlateManipulationTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless) -> None:
        super().__init__(cfg, sim_device, graphics_device_id, headless)