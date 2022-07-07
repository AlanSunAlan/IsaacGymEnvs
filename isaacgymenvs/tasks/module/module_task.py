from http.client import UnimplementedFileMode
import os, sys
from re import L
from types import SimpleNamespace
from typing import Dict, Tuple, List
import numpy as np
import torch
from collections import namedtuple

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import *

from tasks.base.vec_task import VecTask
from abc import abstractmethod

Observation = namedtuple('Observation', ['length', 'lower', 'upper', 'tensor_ref'])
Asset = namedtuple('Asset', ['asset_name', 'asset', 'collision_with_robot', 'filter'])

class ModuleTask(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless) -> None:
        '''
            Params in cfg:
                robot_urdf_path: the path to the urdf file of the robot

                num_modules: the number of modules used in the robot

                fix_base: if the robot base link should be fixed (i.e. for a 
                        quadruped robot, it should set to False)
        '''
        # Some common variables
        self._drive_mode = gymapi.DOF_MODE_POS # Currently only support position control
        self._num_dof_per_module = 3 # Each module has 3 Degrees of Freedom of actuation

        self.cfg = cfg
        self._randomize = self.cfg['task']['randomize']
        self._randomization_params = self.cfg['task']['randomization_params']
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        # For module construction
        self._module_cfg = cfg['module']
        self._asset_folder = self._module_cfg['asset_folder']
        self._robot_urdf_name = self._module_cfg['robot_urdf_name']
        self._num_modules = self._module_cfg['num_modules']
        self._prefix_list = self._module_cfg['prefix_list']
        self._fix_base = self._module_cfg['fix_base']
        self._default_robot_position = list(self._module_cfg['default_world_position'])
        self._default_robot_orientation = list(self._module_cfg['default_world_orientation'])
        # Module control configs
        self._module_control_cfg = self._module_cfg['control']
        self._module_control_params = {
            'stiffness': list(self._module_control_cfg['stiffness']),
            'damping': list(self._module_control_cfg['damping']),
            'max_joint_vel': list(self._module_control_cfg['max_joint_vel']),
            'joint_lower': list(self._module_control_cfg['joint_lower']),
            'joint_upper': list(self._module_control_cfg['joint_upper']),
            'joint_torque_limit': list(self._module_control_cfg['joint_torque_limit']),
            'default_joint_positions': list(self._module_control_cfg['default_joint_positions'])
        }
        '''
        self._module_stiffness = self._module_control_cfg['stiffness']
        self._module_damping = self._module_control_cfg['damping']
        self._module_max_joint_vel = self._module_control_cfg['max_joint_vel']
        self._module_joint_lower = self._module_control_cfg['joint_lower']
        self._module_joint_lower = self._module_control_cfg['joint_upper']
        self._module_joint_torque_limit = self._module_control_cfg['joint_torque_limit']
        self._module_default_joint_positions = self._module_control_cfg['default_joint_positions']
        '''
        
        # Transform the control parameters into correct size
        for param_name in self._module_control_params.keys():
            param_values = self._module_control_params[param_name]
            if len(param_values) == 3:
                self._module_control_params[param_name] = param_values * self._num_modules
            else:
                # the length of the param values must be equal to the number of dofs
                assert len(param_values) == self._num_modules * self._num_dof_per_module, \
                    'Length of the parameter list {} must be consistent with the module number'.format(param_name)

        # Module asset options
        self._module_asset_config = self._module_cfg['asset_options']

        # Knee frame names
        self._knee_frame_names = []
        # Fingertip frame names
        self._fingertip_frame_names = []
        for prefix in self._prefix_list:
            self._fingertip_frame_names.append("{}-fingertip_frame".format(prefix))
            self._knee_frame_names.append("{}-knee_frame".format(prefix))
        self._fingertip_indices = []
        self._knee_indices = []

        # Custom asset_list
        self._custom_asset_list = []

        # Configure observations
        if not hasattr(self, '_custom_obs_list'):
            setattr(self, '_custom_obs_list', [])
        self._enable_tip_obs = self._module_cfg['observations']['enable_tip_obs']
        if self._enable_tip_obs:
            self._tip_pos_lower = self._module_cfg['observations']['tip_position_lower']
            self._tip_pos_upper = self._module_cfg['observations']['tip_position_upper']
            self._tip_vel_lower = self._module_cfg['observations']['tip_vel_lower']
            self._tip_vel_upper = self._module_cfg['observations']['tip_vel_upper']
        # Compute number of observations
        self.cfg["env"]["numObservations"] = 0
        for obs in self._custom_obs_list:
            self.cfg["env"]["numObservations"] += obs.length
        self.cfg["env"]["numObservations"] += self._num_modules*3  + self._num_modules*3 # Joint angles&vels
        self.cfg["env"]["numObservations"] += self._num_modules * self._num_dof_per_module
        if self._enable_tip_obs:
            # Add the tip state observations
            self.cfg["env"]["numObservations"] += 3 + 4 + 3 # Position, quaternion, linear velocity

        self.cfg["env"]["numStates"] = self.cfg["env"]["numObservations"]
        self.cfg["env"]["numActions"] = self._num_modules * self._num_dof_per_module

        # Envs container
        self.envs = []

        # Indices
        self._robot_indices = []
        self._env_obj_indices = dict()
        self._env_obj_root_states = dict()

        # Tensor buffers
        # root_state_tensors
        self.dof_state_tensor = [] # size [num_envs, num_dof*2]
        self.rb_state_tensor = [] # size [num_envs, num_rigid_bodies*13]
        self.root_state_tensor = [] # will be in size [num_envs*num_actors, 13], 2D
        # Construct default robot pose tensor
        self.default_robot_state_tensor = []
        self.default_dof_tensor = []
        # Action tensor
        self.actions = []

        # Call VecTask init
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # Create a default camera view
        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.7, 0.0, 0.7)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Configure action limits
        self._action_lower = torch.tensor(self._module_control_params['joint_lower'], device=self.device)
        self._action_upper = torch.tensor(self._module_control_params['joint_upper'], device=self.device)

        self._reset_indices = torch.tensor([], dtype=torch.int32, device=self.device)

        # Set limits
        self._config_obs()

    def create_sim(self):
        '''
            Create simulation environment
            1. Configure the simulation parameters
            2. Create simulation environments
            3. Create actors from loading the robot&object urdf files
        '''    
        # Configure the simulation parameters
        self.up_axis = self.cfg["sim"]["up_axis"]
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        # Create environment
        self._create_ground_plane()
        self._create_envs()
        if self._randomize:
            self.apply_randomizations(self._randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        # Just to ensure the plane will not collide with the finger
        plane_params.distance = 0.00
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        # Get robot asset
        robot_module_asset = self._robot_module_asset()
        # Set robot properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_module_asset)
        robot_dof_props['driveMode'].fill(self._drive_mode)
        robot_dof_props['stiffness'] = self._module_control_params['stiffness']
        robot_dof_props['damping'] = self._module_control_params['damping']
        robot_dof_props['effort'] = self._module_control_params['joint_torque_limit']
        robot_dof_props['upper'] = self._module_control_params['joint_upper']
        robot_dof_props['lower'] = self._module_control_params['joint_lower']
        robot_dof_props['velocity'] = self._module_control_params['max_joint_vel']

        # Load other assets
        asset_list = self._custom_asset_list

        env_lower_bound = gymapi.Vec3(-self.cfg["env"]["envSpacing"], -self.cfg["env"]["envSpacing"], 0.0)
        env_upper_bound = gymapi.Vec3(self.cfg["env"]["envSpacing"], self.cfg["env"]["envSpacing"], self.cfg["env"]["envSpacing"])
        num_envs_per_row = int(np.sqrt(self.num_envs))

        # Prepare to aggregate actors
        max_agg_bodies = 0
        max_agg_shapes = 0
        # For the robot
        max_agg_bodies += self.gym.get_asset_rigid_body_count(robot_module_asset) 
        max_agg_shapes += self.gym.get_asset_rigid_shape_count(robot_module_asset)
        # For all other assets
        for _asset in asset_list:
            max_agg_bodies += self.gym.get_asset_rigid_body_count(_asset.asset)
            max_agg_shapes += self.gym.get_asset_rigid_shape_count(_asset.asset)

        # Aggregate actors in each environment
        for env_i in range(0, self.num_envs):
            env = self.gym.create_env(self.sim, env_lower_bound, env_upper_bound, num_envs_per_row)

            # Start aggregating
            self.gym.begin_aggregate(env, max_agg_bodies, max_agg_shapes, True)

            # Add robot
            default_world_pose = gymapi.Transform()
            default_position = gymapi.Vec3(self._default_robot_position[0],
                                           self._default_robot_position[1],
                                           self._default_robot_position[2])
            default_rotation = gymapi.Quat(self._default_robot_orientation[0],
                                           self._default_robot_orientation[1],
                                           self._default_robot_orientation[2],
                                           self._default_robot_orientation[3])
            default_world_pose.p = default_position
            default_world_pose.r = default_rotation
            robot_actor = self.gym.create_actor(env, robot_module_asset, default_world_pose,
                                        "robot", env_i, 0)
            robot_index = self.gym.get_actor_index(env, robot_actor, gymapi.DOMAIN_SIM)
            self.gym.set_actor_dof_properties(env, robot_actor, robot_dof_props) # Set DoF control properties
            self._robot_indices.append(robot_index)

            # Add all other assets
            for _asset in asset_list:
                obj_asset = _asset.asset
                asset_name = _asset.asset_name
                _filter = _asset.filter
                if _asset.collision_with_robot:
                    collision_group = env_i
                else:
                    collision_group = self.num_envs + env_i

                actor = self.gym.create_actor(env, obj_asset, gymapi.Transform(),
                                        asset_name, collision_group, _filter)
                actor_index = self.gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)

                self._env_obj_indices[asset_name].append(actor_index)

            self.envs.append(env)
        # Convert the indices into tensors
        self._robot_indices = torch.tensor(self._robot_indices, dtype=torch.long, device=self.device)
        for env_obj_name in self._env_obj_indices.keys():
            indices = self._env_obj_indices[env_obj_name]
            self._env_obj_indices[env_obj_name] = torch.tensor(indices, dtype=torch.long, device=self.device)
        
        # Update the state tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(self.num_envs, -1, 2)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_tensor = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        
        # Create reference for other env obj
        for env_obj_name in self._env_obj_indices.keys():
            env_obj_indices = self._env_obj_indices[env_obj_name]
            self._env_obj_root_states[env_obj_name] = self.root_state_tensor[env_obj_indices]

        # Set values for default dof states
        self.default_dof_tensor = torch.clone(gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))).to(self.device)
        # Change values to default joint positions
        self.default_dof_tensor[:,0] = torch.tensor(self._module_control_params['default_joint_positions'], dtype=torch.float32, device=self.device).repeat(self.num_envs)
        # Set values for default root state tensor
        default_state_array = self._default_robot_position + self._default_robot_orientation + [0,0,0,0,0,0]
        self.default_robot_state_tensor = torch.tensor(default_state_array, dtype=torch.float32, device=self.device)

    def _robot_module_asset(self):
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.fix_base_link = self._fix_base
        robot_asset_options.use_physx_armature = self._module_asset_config['use_physx_armature']
        robot_asset_options.thickness = self._module_asset_config['thickness']

        robot_asset_options.vhacd_enabled = self._module_asset_config['vhacd_enabled']
        if robot_asset_options.vhacd_enabled:
            robot_asset_options.vhacd_params = gymapi.VhacdParams()
            robot_asset_options.vhacd_params.resolution = self._module_asset_config['vhacd_params_resolution']
            robot_asset_options.vhacd_params.concavity = self._module_asset_config['vhacd_params_concavity']
            robot_asset_options.vhacd_params.alpha = self._module_asset_config['vhacd_params_alpha']
            robot_asset_options.vhacd_params.beta = self._module_asset_config['vhacd_params_beta']
            robot_asset_options.vhacd_params.convex_hull_downsampling = self._module_asset_config['vhacd_params_convex_hull_downsampling']
            robot_asset_options.vhacd_params.max_num_vertices_per_ch = self._module_asset_config['vhacd_params_max_num_vertices_per_ch']

        robot_asset = self.gym.load_asset(self.sim, self._asset_folder,
                                          self._robot_urdf_name, robot_asset_options)
        
        robot_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        for p in robot_props:
            p.friction = 3
            p.torsion_friction = 3.0
            p.restitution = 0.8
        self.gym.set_asset_rigid_shape_properties(robot_asset, robot_props)   

        for tip_frame in self._fingertip_frame_names:
            self._fingertip_indices.append(self.gym.find_asset_rigid_body_index(robot_asset, tip_frame))
            if self._fingertip_indices[-1] == gymapi.INVALID_HANDLE:
                print("Invalid handle for {}".format(tip_frame))
                exit(1)

        for knee_frame in self._knee_frame_names:
            self._knee_indices.append(self.gym.find_asset_rigid_body_index(robot_asset, knee_frame))
            if self._knee_indices[-1] == gymapi.INVALID_HANDLE:
                print("Invalid handle for {}".format(knee_frame))
                exit(1)

        return robot_asset

    def _config_obs(self):
        '''
            Set up observation scales
        '''
        # Joint information
        joint_angle_limit = SimpleNamespace(
            low=torch.tensor(self._module_control_params['joint_lower'], dtype=torch.float32, device=self.device),
            high=torch.tensor(self._module_control_params['joint_upper'], dtype=torch.float32, device=self.device)
        )

        joint_velocity_limit = SimpleNamespace(
            low=-1*torch.tensor(self._module_control_params['max_joint_vel'], dtype=torch.float32, device=self.device),
            high=torch.tensor(self._module_control_params['max_joint_vel'], dtype=torch.float32, device=self.device)
        )

        # Fingertip information
        tip_state_limit = SimpleNamespace(
            low=torch.tensor([], dtype=torch.float32, device=self.device),
            high=torch.tensor([], dtype=torch.float32, device=self.device)
        )
        if self._enable_tip_obs:
            tip_position_limit = SimpleNamespace(
                low=torch.tensor(self._tip_pos_lower, dtype=torch.float32, device=self.device),
                high=torch.tensor(self._tip_pos_upper, dtype=torch.float32, device=self.device)
            )

            tip_orientation_limit = SimpleNamespace(
                low=-torch.ones(4, dtype=torch.float32, device=self.device),
                high=torch.ones(4, dtype=torch.float32, device=self.device),
            )

            tip_velocity_limit = SimpleNamespace(
                low=torch.tensor(self._tip_vel_lower, dtype=torch.float32, device=self.device),
                high=torch.tensor(self._tip_vel_upper, dtype=torch.float32, device=self.device),
            )

            tip_state_limit = SimpleNamespace(
            low=torch.cat((
                tip_position_limit.low,
                tip_orientation_limit.low,
                tip_velocity_limit.low
            ), 0).repeat(self._num_modules),
            high=torch.cat((
                tip_position_limit.high,
                tip_orientation_limit.high,
                tip_velocity_limit.high
            ), 0).repeat(self._num_modules)
        )

        # load all other custom objects
        custom_obs_limits = SimpleNamespace(
            low=[],
            high=[]
        )
        for custom_obs in self._custom_obs_list:
            custom_obs_limits.low += custom_obs.lower
            custom_obs_limits.high += custom_obs.upper
        # Convert custom observation limit to tensor
        custom_obs_limits.low = torch.tensor(custom_obs_limits.low, dtype=torch.float32, device=self.device)
        custom_obs_limits.high = torch.tensor(custom_obs_limits.high, dtype=torch.float32, device=self.device)

        action_lower_limit = -1*torch.ones(self._num_modules * self._num_dof_per_module, dtype=torch.float32, device=self.device)
        action_upper_limit = torch.ones(self._num_modules * self._num_dof_per_module, dtype=torch.float32, device=self.device)

        # Combine all limits
        self.obs_lower_limit = torch.cat((joint_angle_limit.low,
                                          joint_velocity_limit.low,
                                          tip_state_limit.low,
                                          custom_obs_limits.low,
                                          action_lower_limit), 0)

        self.obs_upper_limit = torch.cat((joint_angle_limit.high,
                                          joint_velocity_limit.high,
                                          tip_state_limit.high,
                                          custom_obs_limits.high,
                                          action_upper_limit), 0)

        self.state_lower_limit = self.obs_lower_limit
        self.state_upper_limit = self.obs_upper_limit

        # To make sure the size is as intended
        assert len(self.obs_lower_limit) == self.cfg["env"]["numObservations"], \
            "Inconsistency of observation size. Tensor size: {}; Supposed to be {}".format(len(self.obs_lower_limit), self.cfg["env"]["numObservations"])

        assert len(self.state_lower_limit) == self.cfg["env"]["numStates"], \
            "Inconsistency of observation size. Tensor size: {}; Supposed to be {}".format(len(self.state_lower_limit), self.cfg["env"]["numStates"])            

        print("Action limits: ")
        print(self._action_lower)
        print(self._action_upper)
        print("Observation limit: ")
        print(self.obs_lower_limit)
        print(self.obs_upper_limit)
        print("State limit: ")
        print(self.state_lower_limit)
        print(self.state_upper_limit)

        
    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)

        # Transform the scale of the action
        action_transformed = unscale_transform(self.actions, self._action_lower, self._action_upper)

        # Conduct action
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action_transformed))

    def post_physics_step(self):
        self.progress_buf += 1

        # Compute the observations
        self.compute_observation()
        # Compute the rewards
        self.compute_reward()

        # Check if the envs should be terminated and reset
        self.check_termination()

    def compute_observation(self):
        # Refresh tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)     
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        for env_obj_name in self._env_obj_indices.keys():
            env_obj_indices = self._env_obj_indices[env_obj_name]
            self._env_obj_root_states[env_obj_name] = self.root_state_tensor[env_obj_indices]
        
        # Calculate observations
        tip_states = torch.tensor([], dtype=torch.float32, device=self.device)
        if self._enable_tip_obs:
            tip_states = self.fingertip_states.view(-1, self._num_modules*13)
        custom_obs_tensors = torch.tensor([], dtype=torch.float32, device=self.device)
        for custom_obs in self._custom_obs_list:
            custom_obs_tensors = torch.cat((custom_obs_tensors, custom_obs.tensor_ref()), dim=-1)

        self.obs_buf[:], self.states_buf[:] = aggregate_obs_state(
            self.dof_positions,
            self.dof_velocities,
            tip_states,
            custom_obs_tensors,
            self.actions
        )

        # Normalize the observations
        self.obs_buf = scale_transform(
            self.obs_buf,
            lower=self.obs_lower_limit,
            upper=self.obs_upper_limit
        )  
        self.states_buf = scale_transform(
            self.states_buf,
            lower=self.state_lower_limit,
            upper=self.state_upper_limit
        )

    def check_termination(self):
        '''
            Only provide termination mechanism where the task ends when
            it reaches the maximum episode timesteps.

            Implement your own function it this is not enough. You can
            still call super().check_termination() to avoid writing overtime
            termination code again
        '''
        # Reaching the maximum episode timesteps
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def reset_idx(self, env_ids):
        '''
            reset_idx() will help you reset the buffers and robot poses.

            Robot will return to the initial state including resetting the
            dof and the world position and orientations defined in the urdf
            files.

            If you still need to reset the state of your custom objects, you
            have to modify the corresponding state tensor buffer. But remember
            do not call set_actor_root_state_tensor_indexed in your reset_idx.

            You have to call super().reset_idx(env_ids) after you have done your
            resettings. And to reset your custom objects, we recommend to use 
            self.reset_obj_root_tensor()
        '''
        # Reset buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # Reset robot dof states
        reset_robot_indices = self._robot_indices[env_ids]
        self.root_state_tensor[reset_robot_indices] = self.default_robot_state_tensor.repeat(len(reset_robot_indices), 1) 

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.default_dof_tensor),
                                                gymtorch.unwrap_tensor(reset_robot_indices.to(torch.int32)), len(reset_robot_indices))

        # Combine all the indices
        self._reset_indices = torch.cat((self._reset_indices, reset_robot_indices))
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                                                            gymtorch.unwrap_tensor(self._reset_indices.to(torch.int32)), len(self._reset_indices))

        # Empty the reset_indices
        self._reset_indices = torch.tensor([], dtype=torch.int32, device=self.device)

    def reset_obj_root_tensor(self, 
                              reset_env_ids,
                              obj_name:str,
                              position: torch.Tensor,
                              orientation: torch.Tensor,
                              linear_vel: torch.Tensor=torch.tensor([0,0,0],dtype=torch.float32),
                              angular_vel: torch.Tensor=torch.tensor([0,0,0],dtype=torch.float32)) -> None:
        obj_indices = self._env_obj_indices[obj_name]
        obj_reset_indices = obj_indices[reset_env_ids]

        # Update the reset indices
        self._reset_indices = torch.cat((self._reset_indices, obj_reset_indices))

        # Combine the state tensors into one
        linear_vel = linear_vel.to(self.device)
        angular_vel = linear_vel.to(self.device)
        state_tensor = torch.cat((
            position,
            orientation,
            linear_vel,
            angular_vel
        ), dim=-1)
        
        self.root_state_tensor[obj_reset_indices] = state_tensor

    def get_obj_root_tensor(self, obj_name: str) -> torch.Tensor:
        '''
            Return the root state tensor given the name.

            Return:
                Tensor will be in shape of (num_envs, 13)
        '''
        return self._env_obj_root_states[obj_name]

    def get_robot_root_tensor(self):
        return self.robot_root_tensor

    def add_asset(self, asset_name, asset, collision_with_robot, filter):
        '''
            asset_name: name of the asset in str; it will be used to
            access the root state tensor of the object corresponding
            to this asset

            asset: the asset handler

            collision_with_robot: should be set to True if you want
            the asset to be able to collide with the robot, otherwise
            set it to False

            filter: corresponding to the paremeter <filter> in isaacgym.gymapi.Gym.create_actor
        '''
        self._custom_asset_list.append(Asset(asset_name,
                                       asset,
                                       collision_with_robot,
                                       filter 
        ))

    def add_obs(self, length, lower, upper, tensor_ref):
        '''
            length: length of the observation tensor

            lower: lower bound of the observation tensor

            upper: upper bound of the observation tensor

            tensor_ref: a reference to get the observation tensor; could be a function
            whose return value is a tensor in shape (num_envs, observation) a tensor reference.

            Note: lower and upper is for the tensor of size 1D.
            (effective for all envs, do not need to be in size (num_env, obs_tensor))
        '''
        # If the list is not created, create one
        if not hasattr(self, '_custom_obs_list'):
            setattr(self, '_custom_obs_list', [])
        self._custom_obs_list.append(Observation(length, lower, upper, tensor_ref))

    def get_fingertip_states(self):
        return self.rb_state_tensor[:, self._fingertip_indices]

    def get_knee_states(self):
        return self.rb_state_tensor[:, self._knee_indices]

    @property
    def robot_root_tensor(self):
        return self.root_state_tensor[self._robot_indices]

    @property
    def dof_positions(self):
        return self.dof_state_tensor[:,:,0]

    @property
    def dof_velocities(self):
        return self.dof_state_tensor[:,:,1]

    @property
    def fingertip_states(self):
        return self.rb_state_tensor[:, self._fingertip_indices]

    @property
    def knee_states(self):
        return self.rb_state_tensor[:, self._knee_indices]

    @abstractmethod
    def compute_reward(self):
        '''
            Compute the reward and update the self.rew_buf.
        '''
        raise NotImplementedError

@torch.jit.script
def aggregate_obs_state(joint_angles: torch.Tensor,
                        joint_velocities: torch.Tensor,
                        tip_states: torch.Tensor,
                        custom_obs_tensors: torch.Tensor,
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    obs = torch.cat([joint_angles,
                     joint_velocities,
                     tip_states,
                     custom_obs_tensors,
                     actions], dim=-1)

    state = obs

    return obs, state        