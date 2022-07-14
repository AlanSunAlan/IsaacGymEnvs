from tasks.module.single_plate_manipulation_task import SinglePlateManipulationTask

class DualfingerPlateManipulation(SinglePlateManipulationTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless) -> None:
        super().__init__(cfg, sim_device, graphics_device_id, headless)