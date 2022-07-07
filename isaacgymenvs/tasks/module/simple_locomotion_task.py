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

        self.cfg = cfg
        self.sim_device = sim_device

        self._locomotion_cfg = self.cfg['locomotion']

        # get some configs that are helpful in this class
        self._vel_weight = self._locomotion_cfg['vel_weight']
        self._baseline_vel = self._locomotion_cfg['baseline_vel']
        self._deviation_weight = cfg['env']['reward_weight']['deviation_weight']
        self._heading_direction = list(self._locomotion_cfg['heading_direction'])

        super().__init__(self.cfg, self.sim_device, graphics_device_id, headless)

        # The direction you want your robot to head on
        self._heading_direction = torch.tensor(self._heading_direction, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        # Deviation direction, the direction that is perpendicular to heading direction
        self.deviation_direction = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)

@torch.jit.script
def _compute_deviation_direction(
    heading_direction: torch.Tensor,
    num_envs: int
) -> torch.Tensor:
    '''
        Compute the deviation direction that is perpendicular to
        heading direction. Only calculate 2D direction, i.e. in 
        x-y plane
    '''
    xy_vector = heading_direction[0][0:2]
    