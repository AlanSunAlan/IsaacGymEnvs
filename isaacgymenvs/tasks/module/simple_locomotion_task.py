from collections import deque
import os, sys
from types import SimpleNamespace
from typing import Deque, Tuple
import numpy as np

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import *

from tasks.module.module_task import ModuleTask

class SimpleLocomotionTask(ModuleTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        
        self.cfg = cfg
        self.sim_device = sim_device

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.dt = self.cfg['sim']['dt']

        # get some configs that are helpful in this class
        self._falling_height = cfg['locomotion']['falling_height']
        self._baseline_vel = cfg['locomotion']['baseline_vel']
        self._baseline_knee_z = cfg['locomotion']['min_knee_z']
        self._vel_weight = cfg['locomotion']['reward_weight']['vel_weight']
        self._deviation_weight = cfg['locomotion']['reward_weight']['deviation_weight']
        self._knee_touch_ground_weight = cfg['locomotion']['reward_weight']['knee_touch_ground_penalty']
        self._falling_penalty = cfg['locomotion']['reward_weight']['falling_penalty']

        self.add_obs(4, [0]*4, [0.12]*4, self.get_knee_z_values)

        super().__init__(self.cfg, self.sim_device, graphics_device_id, headless)

        # The direction you want your robot to head on
        self.heading_direction = torch.tensor([0, -1, 0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        # Deviation direction, the direction that is perpendicular to heading direction
        self.deviation_direction = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)

    def check_termination(self):
        body_z_position = self.get_robot_root_tensor()[:, 2]
        
        # Reset the robot if it falls 
        falling_indices = (body_z_position < self._falling_height).nonzero()
        if len(falling_indices) > 0:
            self.reset_buf[falling_indices] = 1

            # Add falling penelty for those robots fallen
            ## The penalty will be equal to the falling_penalty * timesteps_left
            self.rew_buf[falling_indices] += (-1*self.progress_buf[falling_indices] + self.max_episode_length) * self._falling_penalty

        # Reset robot if any one of its knees touch the ground
        knee_z_values = self.get_knee_z_values()
        knee_touching = torch.where(knee_z_values < self._baseline_knee_z, torch.ones_like(knee_z_values), torch.zeros_like(knee_z_values))
        knee_flat = torch.sum(knee_touching, 1)
        knee_reset_indices = (knee_flat > 0).nonzero() # The indices where one of its knees falls
        if len(knee_reset_indices) > 0:
            self.reset_buf[knee_reset_indices] = 1

            # Add touching ground penalty
            ## The penalty will be equal to the touching_penalty * timesteps_left
            self.rew_buf[knee_reset_indices] += (-1*self.progress_buf[knee_reset_indices] + self.max_episode_length) * self._knee_touch_ground_weight

        return super().check_termination()

    def compute_reward(self):
        self.robot_root_tensor_buf = self.get_robot_root_tensor()
        self.rew_buf[:], heading_vel_reward, deviation_vel_penalty = _compute_rew(
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
            "DeviationVelocityPenelty": deviation_vel_penalty,
        }

        self.extras.update({"env/rewards/"+k: v.mean() for k, v in rew_info.items()})

    def get_knee_z_values(self):
        return self.knee_states[:, :, 2]

@torch.jit.script
def _compute_rew(robot_root_tensor: torch.Tensor,
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

    # velocity reward and penalty
    heading_vel_reward = flat_heading_vel * vel_weight
    deviation_vel_penalty = flat_deviation_vel * deviation_weight

    total_reward = heading_vel_reward + deviation_vel_penalty

    return total_reward, heading_vel_reward, deviation_vel_penalty