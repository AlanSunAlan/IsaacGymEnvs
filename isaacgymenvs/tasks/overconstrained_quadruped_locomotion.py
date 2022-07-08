from tasks.module.simple_locomotion_task import SimpleLocomotionTask

class OverconstrainedQuadrupedLocomotion(SimpleLocomotionTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        super().__init__(cfg, sim_device, graphics_device_id, headless)