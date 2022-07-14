from cmath import pi
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
    _boundary_diameter = 0.7
    _plate_size = 0.4
    _boundary_z_offset = 0.02
    _tip_min_position = [-1*_boundary_diameter/2, -1*_boundary_diameter/2, 0]
    _tip_max_position = [_boundary_diameter/2, _boundary_diameter/2, 0.5]
    _tip_min_vel = [-1, -1, -1]
    _tip_max_vel = [1, 1, 1]
    _tip_min_angular_vel = [-np.math.pi, -np.math.pi, -np.math.pi]
    _tip_max_angular_vel = [np.math.pi, np.math.pi, np.math.pi]

    _safety_offset = 0.01
    _max_com_distance = (_boundary_diameter - np.math.sqrt(2)*_plate_size - _safety_offset)/2.0

    def __init__(self, cfg, sim_device, graphics_device_id, headless) -> None:
        self.cfg = cfg
        # enable tip state observations
        self.cfg['module']['observations']['enable_tip_obs'] = True
        self.cfg['module']['observations']['tip_position_lower'] = self._tip_min_position
        self.cfg['module']['observations']['tip_position_upper'] = self._tip_max_position
        self.cfg['module']['observations']['tip_vel_lower'] = self._tip_min_vel
        self.cfg['module']['observations']['tip_vel_upper'] = self._tip_max_vel
        self.cfg['module']['observations']['tip_angular_vel_lower'] = self._tip_min_angular_vel
        self.cfg['module']['observations']['tip_angular_vel_upper'] = self._tip_max_angular_vel

        # Some useful configs
        self._dis_weight = self.cfg['manipulation']['reward_weight']['distance_weight']
        self._yaw_weight = self.cfg['manipulation']['reward_weight']['yaw_weight']

        self.add_asset('boundary', self._boundary_asset, True, 1)
        self.add_asset('plate', self._plate_asset, True, 0)
        self.add_asset('goal_plate', self._goal_plate_asset, False, 0)
        self.add_asset('table', self._table_asset, True, 1)
        
        # add plate obs; x, y, yaw, linear velocity (x,y), angular velocity(yaw)
        plate_obs_lower = [-self._max_com_distance, -self._max_com_distance, 0, -0.2, -0.2, -0.5*np.math.pi]
        plate_obs_upper = [self._max_com_distance, self._max_com_distance, 2*np.math.pi, 0.2, 0.2, 0.5*np.math.pi]
        self.add_obs(6, plate_obs_lower, plate_obs_upper, self._calc_plate_state)
        goal_obs_lower = [-self._max_com_distance, -self._max_com_distance, 0]
        goal_obs_upper = [self._max_com_distance, self._max_com_distance, 2*np.math.pi]
        self.add_obs(3, goal_obs_lower, goal_obs_upper, self._calc_goal_plate_state)

        super().__init__(cfg, sim_device, graphics_device_id, headless)

        self._plate_2d_states = []
        self._goal_plate_2d_states = []

    def reset_idx(self, env_ids):
        plate_angle = 150
        goal_angle = 0

        reset_len = len(env_ids)

        # Generate random poses for plate and goal plate
        plate_pos_x, plate_pos_y = random_xy(reset_len*2, self._max_com_distance, self.device)
        plate_pos_z = torch.tensor(self._boundary_z_offset + self._safety_offset, dtype=torch.float32, device=self.device).repeat(reset_len*2).view(reset_len*2, 1)
        plate_pos_x = plate_pos_x.view(reset_len*2, 1)
        plate_pos_y = plate_pos_y.view(reset_len*2, 1)
        plate_positions = torch.cat((plate_pos_x, plate_pos_y, plate_pos_z), dim=-1)
        plate_quaternions = random_yaw_orientation(reset_len*2, self.device)

        plate_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), torch.tensor(deg_to_rad(plate_angle))).repeat(reset_len, 1).to(self.device)
        goal_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), torch.tensor(deg_to_rad(goal_angle))).repeat(reset_len, 1).to(self.device)
        # Reset plate pose
        #self.reset_obj_root_tensor(env_ids, 
        #                          "plate",
        #                           plate_positions[0:reset_len, :],
        #                           plate_quaternions[0:reset_len, :])
        
        self.reset_obj_root_tensor(env_ids, 
                                  "plate",
                                   plate_positions[0:reset_len, :],
                                   plate_quat[:, :])
        # Reset goal plate pose
        #self.reset_obj_root_tensor(env_ids,
        #                           'goal_plate',
        #                           plate_positions[reset_len:, :],
        #                           plate_quaternions[reset_len:, :])
        self.reset_obj_root_tensor(env_ids,
                                   'goal_plate',
                                   plate_positions[reset_len:, :],
                                   goal_quat[:, :])

        return super().reset_idx(env_ids)

    def compute_reward(self):
        self._plate_2d_states = self._calc_plate_state()
        self._goal_plate_2d_states = self._calc_goal_plate_state()

        dis = self._plate_2d_states[:, 0:2] - self._goal_plate_2d_states[:, 0:2]
        dis_l2 = torch.norm(dis, p=2, dim=-1)

        yaw_diff = self._plate_2d_states[:, 2] - self._goal_plate_2d_states[:, 2]

        # Yaw difference should be no larger than PI
        # Should be min{yaw_diff, 2PI - yaw_diff}
        yaw_diff = torch.abs(yaw_diff)
        two_pi_minus_yaw = 2*np.pi - yaw_diff
        # Element-wise compare
        min_yaw_diff = torch.minimum(yaw_diff, two_pi_minus_yaw)

        #print("Plate: ", rad_to_deg(self._plate_2d_states[:, 2]))
        #print("Goal : ", rad_to_deg(self._goal_plate_2d_states[:, 2]))
        #print("Raw diff: ", rad_to_deg(yaw_diff))
        #print("Min diff: ", rad_to_deg(min_yaw_diff))

        dis_weighted = self._dis_weight * dis_l2
        yaw_weighted = self._yaw_weight * min_yaw_diff

        self.rew_buf[:] = dis_weighted + yaw_weighted

        print("Rew: ", self.rew_buf[0])

        update_info = {
            "raw_distance_l2": dis_l2,
            "raw_yaw_diff": yaw_diff,
            "distance_penalty_avg": dis_weighted,
            "yaw_penalty_avg": yaw_weighted,
            "reward_avg": self.rew_buf
        }

        self.extras.update({"env/rewards/"+k: v.mean() for k, v in update_info.items()})

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

    def _calc_plate_state(self):
        return computePlate2DState(self.num_envs,
                                   self.get_obj_root_tensor('plate'),
                                   self.device)

    def _calc_goal_plate_state(self):
        return computeGoalPlate2DState(self.num_envs,
                                       self.get_obj_root_tensor('goal_plate'),
                                       self.device)

def rad_to_deg(rad):
    return rad*180/pi

def deg_to_rad(deg):
    return deg*pi/180

@torch.jit.script
def computePlate2DState(num_envs: int, plate_states: torch.Tensor, device: str) -> torch.Tensor:
    '''
        Simplify the state representation of the plate

        Output will be in size of [num_envs, 2D_states]

        2D state include: [x, y, yaw, x_velocity, y_velocity, angular velocity]
    '''
    temp_tensor = torch.zeros((num_envs, 6), dtype=torch.float32, device=device)

    # Just copy x, y values
    temp_tensor[:, 0:2] = plate_states[:, 0:2]

    q = plate_states[:, 3:7]
    _, _, yaw = get_euler_xyz(q)

    temp_tensor[:, 2] = yaw

    # Just copy x, y, yaw  velocity values
    temp_tensor[:, 3:5] = plate_states[:, 7:9]
    temp_tensor[:, 5 ] = plate_states[:, 12]

    # Forming up 2D states
    return temp_tensor        

@torch.jit.script
def computeGoalPlate2DState(num_envs: int, goal_plate_states: torch.Tensor, device: str) -> torch.Tensor:
    '''
        Simplify the state representation of the goal plate (should stay still all the time)

        Output will be in size of [num_envs, 2D_states]

        2D state include: [x, y, yaw]
    '''
    temp_tensor = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    temp_tensor[:, 0:2] = goal_plate_states[:, 0:2]
    q = goal_plate_states[:, 3:7]
    _, _, yaw  = get_euler_xyz(q)
    temp_tensor[:, 2] = yaw

    return temp_tensor

@torch.jit.script
def random_xy(num: int, max_com_distance_to_center: float, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Copied from tasks.trifinger.py'''
    """Returns sampled uniform positions in circle (https://stackoverflow.com/a/50746409)"""
    # sample radius of circle
    radius = torch.sqrt(torch.rand(num, dtype=torch.float, device=device))
    radius *= max_com_distance_to_center
    # sample theta of point
    theta = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)
    # x,y-position of the cube
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    return x, y

@torch.jit.script
def random_yaw_orientation(num: int, device: str) -> torch.Tensor:
    '''Copied from tasks.trifinger.py'''
    """Returns sampled rotation around z-axis."""
    roll = torch.zeros(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)

    return quat_from_euler_xyz(roll, pitch, yaw)