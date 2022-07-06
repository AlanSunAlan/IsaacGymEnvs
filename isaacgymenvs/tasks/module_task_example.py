import os, sys
from types import SimpleNamespace
from typing import Tuple
import numpy as np

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import *

from tasks.module.module_task import ModuleTask

class ModuleTaskExample(ModuleTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        
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

    def compute_reward(self):
        self.robot_root_tensor_buf = self.get_robot_root_tensor()
        self.rew_buf[:], heading_vel_reward, deviation_vel_penalty = _compute_obs(
            self.robot_root_tensor_buf,
            self.heading_direction,
            self.deviation_direction,
            self._baseline_vel,
            self._vel_weight,
            self._deviation_weight
        )
        
        rew_info = {
            "TotalReward": self.rew_buf,
            "HeadingVelocityReward": heading_vel_reward,
            "DeviationVelocityPenelty": deviation_vel_penalty
        }
        
        self.extras.update({"env/rewards/"+k: v.mean() for k, v in rew_info.items()})


@torch.jit.script
def _compute_obs(robot_root_tensor: torch.Tensor,
                 heading_direction: torch.Tensor,
                 deviation_direction: torch.Tensor,
                 baseline_vel: float,
                 vel_weight: float,
                 deviation_weight: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot_world_vel = robot_root_tensor[:, 7:10]
    heading_vel = torch.mul(heading_direction, robot_world_vel)
    flat_raw_heading_vel = torch.sum(heading_vel, 1)
    deviation_vel = torch.mul(deviation_direction, robot_world_vel)
    flat_deviation_vel = torch.abs(torch.sum(deviation_vel, 1))

    flat_heading_vel = flat_raw_heading_vel - baseline_vel

    heading_vel_reward = flat_heading_vel * vel_weight
    deviation_vel_penalty = flat_deviation_vel * deviation_weight
    total_reward = heading_vel_reward + deviation_vel_penalty

    return total_reward, heading_vel_reward, deviation_vel_penalty



'''
@torch.jit.script
def _calc_deviation_direction(heading_direction: torch.Tensor) -> torch.Tensor:
'''
