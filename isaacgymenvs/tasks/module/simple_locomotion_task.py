import os, sys
from types import SimpleNamespace
from typing import Tuple
import numpy as np

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import *

from tasks.module.module_task import ModuleTask

class SimpleLocomotionTask(ModuleTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless) -> None:
        super().__init__(cfg, sim_device, graphics_device_id, headless)

        self.cfg = cfg
        self.sim_device = sim_device

        # get some configs that are helpful in this class
        self._vel_weight = cfg['env']['reward_weight']['vel_weight']
        self._baseline_vel = cfg['env']['reward_weight']['baseline_vel']
        self._deviation_weight = cfg['env']['reward_weight']['deviation_weight']

        # Create your assets here
        ## Nothing is done here, no need to create custom assets

        # Config your observations here
        ## No need for other observations

        super().__init__(self.cfg, self.sim_device, graphics_device_id, headless)

        # The direction you want your robot to head on
        self.heading_direction = torch.tensor([0, -1, 0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        # Deviation direction, the direction that is perpendicular to heading direction
        self.deviation_direction = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)