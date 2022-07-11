from tasks.module.simple_locomotion_task import SimpleLocomotionTask
import torch

class OverconstrainedQuadrupedLocomotion(SimpleLocomotionTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self._standstill_penalty = self.cfg['locomotion']['reward_weight']['stand_still_penalty']

        super().__init__(cfg, sim_device, graphics_device_id, headless)

        self._standstill_tensor = torch.tensor([self._standstill_penalty]*self.num_envs, dtype=torch.float32, device=self.device)

    def compute_reward(self):
        robot_root_state = self.get_robot_root_tensor()
        robot_world_vel_heading = -1 * robot_root_state[:, 8]

        standstill_p = torch.where(robot_world_vel_heading < self._baseline_vel, self._standstill_tensor, torch.zeros_like(self._standstill_tensor))

        self.rew_buf = standstill_p

        self.rew_info["StandStillPenalty"] = standstill_p

        super().compute_reward()