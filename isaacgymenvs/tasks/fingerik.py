import os, sys
import enum
from xml.etree.ElementTree import QName
import numpy as np

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import *

from tasks.base.vec_task import VecTask

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
gym = gymapi.acquire_gym()

class FingerIK(VecTask):
    GoalPosDimension = 3
    JointVelocityDimension = 3
    JointPositionDimension = 3
    JointTorqueDimension = 3

    _actuation_mode = gymapi.DOF_MODE_EFFORT
    _stiffness = 1
    _damping = 0.1
    _lower_limit = [-3.14, -1.57, 0.2]
    _upper_limit = [3.14, 1.57, 2.3]

    _torque_lower_limit = [-2.0, -2.0, -3.5] # 3 Nm maximum torque
    _torque_upper_limit = [2.0, 2.0, 3.5] # 3 Nm maximum torque
    _torque_control_step = 0.1 # The minimum controllable torque

    _asset_root = os.path.join(ROOT_DIR, "assets")
    _robot_urdf = "fingerik/finger.urdf"
    _goal_obj_urdf = "fingerik/goal_indicator.urdf"

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.obs_dimension = {
            "goal_pos": self.GoalPosDimension,
            "joint_vel": self.JointVelocityDimension,
            "joint_pos": self.JointPositionDimension,
            "joint_torque": self.JointTorqueDimension
        }

        self.state_dimension = {
            **self.obs_dimension,
            "fingertip_pos": 3,
            "fingertip_vel": 3,
            "fingertop_wrench": 3
        }
        self.cfg["env"]["numObservations"] = sum(self.obs_dimension.values())
        self.cfg["env"]["numStates"] = sum(self.state_dimension.values())
        self.cfg["env"]["numActions"] = self.JointTorqueDimension
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.7, 0.0, 0.7)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


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

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        # Just to ensure the plane will not collide with the finger
        plane_params.distance = 0
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        # Loading assets
        # Load robot finger
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.armature = 0.001
        robot_asset = self.gym.load_asset(self.sim, self._asset_root, self._robot_urdf, asset_options)
        # Load goal_object
        goal_obj_asset = self.gym.load_asset(self.sim, self._asset_root, self._goal_obj_urdf, asset_options)

        # Configure the dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        robot_dof_props["driveMode"].fill(self._actuation_mode)
        robot_dof_props["stiffness"].fill(self._stiffness)
        robot_dof_props["damping"].fill(self._stiffness)
 
        for i in range(3):
            robot_dof_props["lower"][i] = float(self._lower_limit[i]) 
            robot_dof_props["upper"][i] = float(self._upper_limit[i])

        self.envs = []
        self.finger_handles = []
        self.goal_obj_handles = []

        # Grid size of each env
        spacing = 0.5
        space_lower = gymapi.Vec3(-spacing/2, -spacing/2, 0)
        space_upper = gymapi.Vec3(spacing/2, spacing/2, spacing)

        num_envs_per_row = int(np.sqrt(self.num_envs))

        # count number of shapes and bodies
        max_agg_bodies = 0
        max_agg_shapes = 0
        max_agg_bodies += self.gym.get_asset_rigid_body_count(robot_asset)
        max_agg_bodies += self.gym.get_asset_rigid_body_count(goal_obj_asset)
        max_agg_shapes += self.gym.get_asset_rigid_shape_count(robot_asset)  
        max_agg_shapes += self.gym.get_asset_rigid_shape_count(goal_obj_asset)                  

        for env_i in range(0, self.num_envs):
            env = self.gym.create_env(self.sim, space_lower, space_upper, num_envs_per_row)

            # Aggregates the robot and goal_object
            self.gym.begin_aggregate(env, max_agg_bodies, max_agg_shapes, True)

            # Create finger actor
            finger_init_pose = gymapi.Transform()
            finger_init_pose.p = gymapi.Vec3(0,0,spacing)
            finger_init_pose.r = self._random_finger_poses()
            finger_actor = self.gym.create_actor(env, robot_asset, finger_init_pose, 
                                                "finger", env_i, 0, 1)


            # Create goal object actor
            goal_obj_init_pose = gymapi.Transform()
            goal_obj_init_pose.p = gymapi.Vec3(0, 0, 0.1)
            goal_obj_actor = self.gym.create_actor(env, goal_obj_asset, goal_obj_init_pose,
                                                  "goal_obj", env_i, 1, 0)

            self.gym.end_aggregate(env)

            self.finger_handles.append(finger_actor)
            self.goal_obj_handles.append(goal_obj_actor)
            self.envs.append(env)

        self._random_reachable_goal_pose(0)

    def pre_physics_step(self, actions):
        '''
            Excute actions from the actor neural network output layer
        '''
        self.actions = actions.clone().to(self.device)
        # From the raw output actions, transform it
        # Calculate upper limit and lower limit of torque values
        upper_limit = torch.tensor(self._torque_upper_limit).to(self.device)
        lower_limit = torch.tensor(self._torque_lower_limit).to(self.device)
        upper_limit_tensor = upper_limit.repeat(self.num_environments, 1).to(self.device)
        lower_limit_tensor = lower_limit.repeat(self.num_environments, 1).to(self.device)
        # Normalize the action
        action_normalized = unscale_transform(self.actions,
                                            lower_limit_tensor,
                                            upper_limit_tensor)


        self.gym.set_dof_actuation_force_tensor(self.sim, 
                                            gymtorch.unwrap_tensor(action_normalized))
        

    def post_physics_step(self):
        self.progress_buf += 1

    def reset_idx(self):
        pass

    def _random_reachable_goal_pose(self, env_id):
        '''
            Randomly select a point within the workspace 
            of the finger as the goal pose
        '''
        # sample 3 random joint positions that is within the joint limits
        lower_limit_tensor = torch.tensor(self._lower_limit, dtype=torch.float32).to(self.device)
        upper_limit_tensor = torch.tensor(self._upper_limit, dtype=torch.float32).to(self.device)
        
        rand_joint_angles = (upper_limit_tensor-lower_limit_tensor) * torch.rand(3, device=self.device) + lower_limit_tensor

        _dof_states = gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states).to(self.device)
        # Set the dof angles for the specific actor
        start_index = env_id * 3
        dof_states[start_index:start_index+3, 0] = rand_joint_angles
        _dof_states = gymtorch.unwrap_tensor(dof_states)
        actor_indices = torch.tensor([env_id], dtype=torch.int32, device=self.device)
        gym.set_dof_state_tensor_indexed(self.sim, _dof_states, gymtorch.unwrap_tensor(actor_indices), 1)

        # Get the fingertip position
        env_rb_positions = self.gym.get_env_rigid_body_states(self.envs[env_id], gymapi.STATE_POS)
        fingertip_frame_index = self.gym.find_actor_rigid_body_index(self.envs[env_id], 
                                                                       self.finger_handles[env_id], 
                                                                       "0-fingertip_frame",
                                                                       gymapi.DOMAIN_ENV)
        
        print("*******")
        print(env_rb_positions[fingertip_frame_index][0][0])
        


    def _random_finger_poses(self):
        '''
            Generate a random quaternion to set the initial finger pose
        '''
        # Randomly choose roll, pitch. yaw
        rand_rpy = torch_rand_float(-np.pi, np.pi, (1, 3), self.device)
        quat_tensor = quat_from_euler_xyz(rand_rpy[0,0], rand_rpy[0,1], rand_rpy[0,2])
        return gymapi.Quat(quat_tensor[0], quat_tensor[1], quat_tensor[2], quat_tensor[3])
