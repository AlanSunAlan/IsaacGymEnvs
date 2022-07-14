from tasks.module.simple_locomotion_task import SimpleLocomotionTask
import torch

class OverconstrainedQuadrupedLocomotion(SimpleLocomotionTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        super().__init__(cfg, sim_device, graphics_device_id, headless)