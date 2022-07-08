from modulefinder import Module
import os, sys
from types import SimpleNamespace
from typing import Tuple
import numpy as np

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import *

from tasks.module.module_task import ModuleTask

gym = gymapi.acquire_gym()

class SinglePlateManipulationTask(ModuleTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless) -> None:
        
        self.add_asset('boundary', self._boundary_asset, True, 1)
        self.add_asset('plate', self._plate_asset, True, 0)
        self.add_asset('goal_plate', self._goal_plate_asset, False, 0)
        self.add_asset('table', self._table_asset, True, 1)
        
        

        super().__init__(cfg, sim_device, graphics_device_id, headless)

    def _plate_asset(self):
        asset_folder = '../assets/plate_manipulation'
        plate_urdf = 'plate_color.urdf'

        plate_asset_options = gymapi.AssetOptions()
        plate_asset_options.disable_gravity = False
        plate_asset_options.thickness = 0.001      

        plate_asset = self.gym.load_asset(self.sim, asset_folder,
                                            plate_urdf, plate_asset_options)
        
        plate_props = self.gym.get_asset_rigid_shape_properties(plate_asset)
        for p in plate_props:
            p.friction = 1.0
            p.torsion_friction = 0.001
            p.restitution = 0.0
        
        self.gym.set_asset_rigid_shape_properties(plate_asset, plate_props)
        
        return plate_asset

    def _goal_plate_asset(self):
        asset_folder = '../assets/plate_manipulation'
        goal_plate_urdf = 'goal_indicator.urdf'

        goal_plate_asset_options = gymapi.AssetOptions()
        goal_plate_asset_options.disable_gravity = True
        goal_plate_asset_options.fix_base_link = True

        goal_plate_asset = self.gym.load_asset(self.sim, asset_folder,
                                               goal_plate_urdf, goal_plate_asset_options)

        return goal_plate_asset

    def _boundary_asset(self):
        asset_folder = '../assets/plate_manipulation'
        boundary_urdf = 'boundary.urdf'

        boundary_asset_options = gymapi.AssetOptions()
        boundary_asset_options.disable_gravity = True
        boundary_asset_options.fix_base_link = True
        boundary_asset_options.thickness = 0.001

        boundary_asset_options.vhacd_enabled = True
        boundary_asset_options.vhacd_params = gymapi.VhacdParams()
        boundary_asset_options.vhacd_params.resolution = 100000
        boundary_asset_options.vhacd_params.concavity = 0.0
        boundary_asset_options.vhacd_params.alpha = 0.04
        boundary_asset_options.vhacd_params.beta = 1.0
        boundary_asset_options.vhacd_params.max_num_vertices_per_ch = 1024

        boundary_asset = self.gym.load_asset(self.sim, asset_folder,
                                           boundary_urdf, boundary_asset_options)
        
        boundary_props = self.gym.get_asset_rigid_shape_properties(boundary_asset)
        for p in boundary_props:
            p.friction = 0.1
            p.torsion_friction = 0.1
        self.gym.set_asset_rigid_shape_properties(boundary_asset, boundary_props) 

        return boundary_asset

    def _table_asset(self):
        asset_folder = '../assets/plate_manipulation'
        table_urdf = 'table_without_border.urdf'

        # define stage asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.thickness = 0.001

        # load stage asset
        table_asset = self.gym.load_asset(self.sim, asset_folder,
                                           table_urdf, table_asset_options)
        # set stage properties
        table_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        # iterate over each mesh
        for p in table_props:
            p.friction = 0.1
            p.torsion_friction = 0.1
        self.gym.set_asset_rigid_shape_properties(table_asset, table_props)

        # return the asset
        return table_asset
