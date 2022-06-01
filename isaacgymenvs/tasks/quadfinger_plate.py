import os, sys
from types import SimpleNamespace
from typing import Tuple
import numpy as np

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import *

from tasks.base.vec_task import VecTask

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
gym = gymapi.acquire_gym()

class QuadfingerPlate(VecTask):
    # Path of the asset folder
    AssetFolder = os.path.join(ROOT_DIR, "assets", "quadfinger_plate")

    # URDF file path
    RobotURDF =  "gripper.urdf"
    PlateURDF = "plate_color.urdf"
    GoalPlateURDF = "goal_indicator.urdf"
    BoundaryURDF = "boundary.urdf"
    TableURDF = "table_without_border.urdf"
    FingertipFrameNames = ["0-fingertip_frame", "1-fingertip_frame", "2-fingertip_frame", "3-fingertip_frame"]

    # Boundary and Object info
    BoundaryDiameter = 0.7
    PlateSize = 0.4
    BoundaryZoffset = 0.02
    TipMinPosition = [-BoundaryDiameter/2, -BoundaryDiameter/2, 0] # xyz limits for the finger tip
    TipMaxPosition = [BoundaryDiameter/2, BoundaryDiameter/2, 0.5]

    # Some important parameters specific to each task
    ModuleNum = 4 # How many 3DoF modules do you use

    # Control related parameters
    # Kp and Kd gains for each DoF in a single module
    ModuleKp = [0.8, 0.8, 0.8] # Equivalent to stiffness
    ModuleKd = [0.3, 0.3, 0.3] # Equivalent to damping
    # Joint velocity limits
    ModuleMaxJointVelocites = [5, 5, 5] # In rad/s
    # Joint position limits
    ModuleJointLowerLimits = [-3.14, -1.57, -2.3] # in radians
    ModuleJointUpperLimits = [3.14, 1.57, 2.3]
    # Joint torque limits
    ModuleJointTorqueLimit = [2, 2, 2]

    # Default joint positions for the robot
    DefaultJointPositions = [2.6, -1.57, 2.3] * ModuleNum

    # For random object pose generation
    SafetyOffset = 0.01
    MaxComDistance = (BoundaryDiameter - np.math.sqrt(2)*PlateSize - SafetyOffset)/2.0

    # Observation space calculation
    ObsDimensions = {"joint_angles": ModuleNum*3,
                     "joint_velocities": ModuleNum*3,
                     "tip_state": 13*ModuleNum, # Including: position, orientation (quaternion), linear and angular velocity
                     "goal_object_pose": 3 , # Including x,y and yaw_angle
                     "current_object_state": 3 + 3 # Including x,y,yaw_angle, linear velocity (2D vector), angular velocity (1D) 
                     }
    
    StateDimensions = {** ObsDimensions,
                       "actions": ModuleNum*3
                      }

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.cfg["env"]["numObservations"] = sum(self.ObsDimensions.values())
        self.cfg["env"]["numStates"] = sum(self.StateDimensions.values())
        self.cfg["env"]["numActions"] = self.ModuleNum*3
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        self.envs = []
        # For the ease of indexing
        self.robot_indices = []
        self.boundary_indices = []
        self.plate_indices = []
        self.goal_plate_indices = []
        self.table_indices = []
        self.tip_indices = []

        # root_state_tensors
        self.dof_state_tensor = [] # size [num_envs, num_dof*2]
        self.dof_positions = []
        self.dof_velocities = []
        self.rb_state_tensor = [] # size [num_envs, num_rigid_bodies*13]
        self.root_state_tensor = [] # will be in size [num_envs*num_actors, 13], 2D
        self.plate_states = []
        self.goal_plate_states = []
        self.plate_2d_state = []
        self.goal_plate_2d_state = []
        # Construct default robot pose tensor
        self.default_dof_tensor = []
        # Action tensor
        self.actions = []

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.7, 0.0, 0.7)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Configure action limits
        self.action_lower_limit = torch.tensor(self.ModuleJointLowerLimits, device=self.device).repeat(self.ModuleNum)
        self.action_upper_limit = torch.tensor(self.ModuleJointUpperLimits, device=self.device).repeat(self.ModuleNum)

        # Configure observation limits
        self.obs_lower_limit = []
        self.obs_upper_limit = []
        # Configure state limits
        self.state_lower_limit = []
        self.state_upper_limit = []
        # Set limits
        self._set_obs_scale()

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
        plane_params.distance = 0.05
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        # Retrieve all assets
        robot_asset = self._robot_asset()
        boundary_asset = self._boundary_asset()
        table_asset = self._table_asset()
        plate_assset = self._plate_asset()
        goal_plate_asset = self._goal_plate_asset()

        # Set robot properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        robot_dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
        robot_dof_props['stiffness'] = self.ModuleKp * self.ModuleNum
        robot_dof_props['damping'] = self.ModuleKd * self.ModuleNum
        robot_dof_props['effort'] = self.ModuleJointTorqueLimit * self.ModuleNum
        robot_dof_props['upper'] = self.ModuleJointUpperLimits * self.ModuleNum
        robot_dof_props['lower'] = self.ModuleJointLowerLimits * self.ModuleNum
        robot_dof_props['velocity'] = self.ModuleMaxJointVelocites * self.ModuleNum

        # Environment boundings
        env_lower_bound = gymapi.Vec3(-self.cfg["env"]["envSpacing"], -self.cfg["env"]["envSpacing"], 0.0)
        env_upper_bound = gymapi.Vec3(self.cfg["env"]["envSpacing"], self.cfg["env"]["envSpacing"], self.cfg["env"]["envSpacing"])
        num_envs_per_row = int(np.sqrt(self.num_envs))

        # Prepare to aggregate actors
        max_agg_bodies = 0
        max_agg_shapes = 0
        # Count bodies
        max_agg_bodies += self.gym.get_asset_rigid_body_count(robot_asset)
        max_agg_bodies += self.gym.get_asset_rigid_body_count(boundary_asset)
        max_agg_bodies += self.gym.get_asset_rigid_body_count(table_asset)
        max_agg_bodies += self.gym.get_asset_rigid_body_count(plate_assset)
        max_agg_bodies += self.gym.get_asset_rigid_body_count(goal_plate_asset)
        # Count shapes
        max_agg_shapes += self.gym.get_asset_rigid_shape_count(robot_asset)
        max_agg_shapes += self.gym.get_asset_rigid_shape_count(boundary_asset)
        max_agg_shapes += self.gym.get_asset_rigid_shape_count(table_asset)
        max_agg_shapes += self.gym.get_asset_rigid_shape_count(plate_assset)
        max_agg_shapes += self.gym.get_asset_rigid_shape_count(goal_plate_asset)
        # Aggregate actors in each environment
        for env_i in range(0, self.num_envs):
            env = self.gym.create_env(self.sim, env_lower_bound, env_upper_bound, num_envs_per_row)
            # Start aggregating
            self.gym.begin_aggregate(env, max_agg_bodies, max_agg_shapes, True)

            # Add robot
            robot_actor = self.gym.create_actor(env, robot_asset, gymapi.Transform(),
                                        "robot", env_i, 0)
            robot_index = self.gym.get_actor_index(env, robot_actor, gymapi.DOMAIN_SIM)
            self.gym.set_actor_dof_properties(env, robot_actor, robot_dof_props) # Set DoF control properties

            # Add boundary
            boundary_actor = self.gym.create_actor(env, boundary_asset, gymapi.Transform(),
                                        "boundary", env_i, 1)
            boundary_index = self.gym.get_actor_index(env, boundary_actor, gymapi.DOMAIN_SIM)

            # Add table
            table_actor = self.gym.create_actor(env, table_asset, gymapi.Transform(),
                                        "table", env_i, 1)
            table_index = self.gym.get_actor_index(env, table_actor, gymapi.DOMAIN_SIM)

            # Add plate
            plate_actor = self.gym.create_actor(env, plate_assset, gymapi.Transform(),
                                        "plate", env_i, 0)
            plate_index = self.gym.get_actor_index(env, plate_actor, gymapi.DOMAIN_SIM)

            # Add goal plate
            goal_plate_actor = self.gym.create_actor(env, goal_plate_asset, gymapi.Transform(),
                                        "goal_plate", self.num_envs + env_i, 0)
            goal_plate_index = self.gym.get_actor_index(env, goal_plate_actor, gymapi.DOMAIN_SIM)

            self.gym.end_aggregate(env)

            self.envs.append(env)
            # Appended to list, for the ease of indexing
            self.robot_indices.append(robot_index)
            self.boundary_indices.append(boundary_index)
            self.table_indices.append(table_index)
            self.plate_indices.append(plate_index)
            self.goal_plate_indices.append(goal_plate_index)

        # Convert indice lists to tensor
        self.robot_indices = torch.tensor(self.robot_indices, dtype=torch.long, device=self.device)
        self.plate_indices = torch.tensor(self.plate_indices, dtype=torch.long, device=self.device)
        self.goal_plate_indices = torch.tensor(self.goal_plate_indices, dtype=torch.long, device=self.device)
        self.boundary_indices = torch.tensor(self.boundary_indices, dtype=torch.long, device=self.device)
        self.table_indices = torch.tensor(self.table_indices, dtype=torch.long, device=self.device)

        # Update the state tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(self.num_envs, -1, 2)
        self.dof_positions = self.dof_state_tensor[:, :, 0]
        self.dof_velocities = self.dof_state_tensor[:, :, 1]
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_tensor = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.plate_states = self.root_state_tensor[self.plate_indices]
        self.goal_plate_states = self.root_state_tensor[self.goal_plate_indices]

        # Set values for default dof states
        self.default_dof_tensor = torch.clone(gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))).to(self.device)
        # Change values to default joint positions
        self.default_dof_tensor[:,0] = torch.tensor(self.DefaultJointPositions, dtype=torch.float32, device=self.device).repeat(self.num_envs)

    def _robot_asset(self):
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.fix_base_link = True
        robot_asset_options.use_physx_armature = True
        robot_asset_options.thickness = 0.001

        robot_asset_options.vhacd_enabled = True
        robot_asset_options.vhacd_params = gymapi.VhacdParams()
        robot_asset_options.vhacd_params.resolution = 20000
        robot_asset_options.vhacd_params.concavity = 0.0025
        robot_asset_options.vhacd_params.alpha = 0.04
        robot_asset_options.vhacd_params.beta = 1.0
        robot_asset_options.vhacd_params.convex_hull_downsampling = 4
        robot_asset_options.vhacd_params.max_num_vertices_per_ch = 128

        robot_asset = self.gym.load_asset(self.sim, self.AssetFolder,
                                          self.RobotURDF, robot_asset_options)

        robot_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        for p in robot_props:
            p.friction = 1.0
            p.torsion_friction = 1.0
            p.restitution = 0.8
        self.gym.set_asset_rigid_shape_properties(robot_asset, robot_props)   

        for tip_frame in self.FingertipFrameNames:
            self.tip_indices.append(self.gym.find_asset_rigid_body_index(robot_asset, tip_frame))
            if self.tip_indices[-1] == gymapi.INVALID_HANDLE:
                print("Invalid handle for {}".format(tip_frame))

        return robot_asset     

    def _plate_asset(self):
        plate_asset_options = gymapi.AssetOptions()
        plate_asset_options.disable_gravity = False
        plate_asset_options.thickness = 0.001      

        plate_asset = self.gym.load_asset(self.sim, self.AssetFolder,
                                            self.PlateURDF, plate_asset_options)
        
        plate_props = self.gym.get_asset_rigid_shape_properties(plate_asset)
        for p in plate_props:
            p.friction = 1.0
            p.torsion_friction = 0.001
            p.restitution = 0.0
        
        self.gym.set_asset_rigid_shape_properties(plate_asset, plate_props)
        
        return plate_asset        


    def _goal_plate_asset(self):
        goal_plate_asset_options = gymapi.AssetOptions()
        goal_plate_asset_options.disable_gravity = True
        goal_plate_asset_options.fix_base_link = True

        goal_plate_asset = self.gym.load_asset(self.sim, self.AssetFolder,
                                               self.GoalPlateURDF, goal_plate_asset_options)

        return goal_plate_asset

    def _boundary_asset(self):
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

        boundary_asset = self.gym.load_asset(self.sim, self.AssetFolder,
                                           self.BoundaryURDF, boundary_asset_options)
        
        boundary_props = self.gym.get_asset_rigid_shape_properties(boundary_asset)
        for p in boundary_props:
            p.friction = 0.1
            p.torsion_friction = 0.1
        self.gym.set_asset_rigid_shape_properties(boundary_asset, boundary_props) 

        return boundary_asset       

    def _table_asset(self):
        '''Copied from tasks.Trifinger'''
        # define stage asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.thickness = 0.001

        # load stage asset
        table_asset = self.gym.load_asset(self.sim, self.AssetFolder,
                                           self.TableURDF, table_asset_options)
        # set stage properties
        table_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        # iterate over each mesh
        for p in table_props:
            p.friction = 0.1
            p.torsion_friction = 0.1
        self.gym.set_asset_rigid_shape_properties(table_asset, table_props)
        # return the asset
        return table_asset

    def _set_obs_scale(self):
        '''
            Set the scale tensor for observation and states to 
            normalize the tensor
        '''
        joint_angle_limit = SimpleNamespace(
            low=torch.tensor(self.ModuleJointLowerLimits * self.ModuleNum, dtype=torch.float32, device=self.device),
            high=torch.tensor(self.ModuleJointUpperLimits * self.ModuleNum, dtype=torch.float32, device=self.device)
        )

        joint_velocity_limit = SimpleNamespace(
            low=torch.tensor([-1*i for i in self.ModuleMaxJointVelocites] * self.ModuleNum, dtype=torch.float32, device=self.device),
            high=torch.tensor(self.ModuleMaxJointVelocites * self.ModuleNum, dtype=torch.float32, device=self.device)
        )

        tip_position_limit = SimpleNamespace(
            low=torch.tensor(self.TipMinPosition, dtype=torch.float32, device=self.device),
            high=torch.tensor(self.TipMaxPosition, dtype=torch.float32, device=self.device)
        )

        tip_orientation_limit = SimpleNamespace(
            low=-torch.ones(4, dtype=torch.float32, device=self.device),
            high=torch.ones(4, dtype=torch.float32, device=self.device),
        )

        # May need to change this:
        tip_velocity_limit = SimpleNamespace(
            low=torch.tensor([-0.2]*6, dtype=torch.float32, device=self.device),
            high=torch.tensor([0.2]*6, dtype=torch.float32, device=self.device),
        )

        tip_state_limit = SimpleNamespace(
            low=torch.cat((
                tip_position_limit.low,
                tip_orientation_limit.low,
                tip_velocity_limit.low
            ), 0).repeat(self.ModuleNum),
            high=torch.cat((
                tip_position_limit.high,
                tip_orientation_limit.high,
                tip_velocity_limit.high
            ), 0).repeat(self.ModuleNum)
        )

        # x, y, yaw, linear velocity (x,y), angular velocity(yaw)
        plate_state_limit = SimpleNamespace(
            low=torch.tensor([-0.5*self.MaxComDistance, -0.5*self.MaxComDistance, 0, -0.2, -0.2, -0.5*np.math.pi], dtype=torch.float32, device=self.device), 
            high=torch.tensor([0.5*self.MaxComDistance, 0.5*self.MaxComDistance, 2*np.math.pi, 0.2, 0.2, 0.5*np.math.pi], dtype=torch.float32, device=self.device)
        )

        # x, y, yaw_angle
        goal_plate_pose_limit = SimpleNamespace(
            low=torch.tensor([-0.5*self.MaxComDistance, -0.5*self.MaxComDistance, 0], dtype=torch.float32, device=self.device),
            high=torch.tensor([0.5*self.MaxComDistance, 0.5*self.MaxComDistance, 2*np.math.pi], dtype=torch.float32, device=self.device)
        )

        # For observation limit
        self.obs_lower_limit = torch.cat((joint_angle_limit.low,
                                          joint_velocity_limit.low,
                                          tip_state_limit.low,
                                          plate_state_limit.low,
                                          goal_plate_pose_limit.low), 0)

        self.obs_upper_limit = torch.cat((joint_angle_limit.high,
                                          joint_velocity_limit.high,
                                          tip_state_limit.high,
                                          plate_state_limit.high,
                                          goal_plate_pose_limit.high), 0)

        # For state limit
        self.state_lower_limit = torch.cat((self.obs_lower_limit, self.action_lower_limit), 0)
        self.state_upper_limit = torch.cat((self.obs_upper_limit, self.action_upper_limit), 0)

        # To make sure the size is as intended
        assert len(self.obs_lower_limit) == self.cfg["env"]["numObservations"], \
            "Inconsistency of observation size. Tensor size: {}; Supposed to be {}".format(len(self.obs_lower_limit), self.cfg["env"]["numObservations"])

        assert len(self.state_lower_limit) == self.cfg["env"]["numStates"], \
            "Inconsistency of observation size. Tensor size: {}; Supposed to be {}".format(len(self.state_lower_limit), self.cfg["env"]["numStates"])            

        print("Action limits: ")
        print(self.action_lower_limit)
        print(self.action_upper_limit)
        print("Observation limit: ")
        print(self.obs_lower_limit)
        print(self.obs_upper_limit)
        print("State limit: ")
        print(self.state_lower_limit)
        print(self.state_upper_limit)


    def pre_physics_step(self, actions):
        '''
            Conduct actions from the action NN
        '''       
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)

        # Transform the scale of the action
        action_transformed = unscale_transform(self.actions, self.action_lower_limit, self.action_upper_limit)

        # Conduct action
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action_transformed))

    def post_physics_step(self):
        self.progress_buf += 1

        # Compute the observations
        self.computeObservation()
        # Compute reward
        self.computeReward()

        # Check if the envs should be terminated and reset
        self.check_termination()

        #print("Observation: ", self.obs_buf[0, :])
        #print("Reward: ", self.rew_buf[0])

    def computeObservation(self):
        # Refresh tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)     
        self._updateRootStates()

        # Get joint angles
        tip_states = self.rb_state_tensor[:, self.tip_indices]
        self.plate_2d_state = computePlate2DState(self.num_envs, self.plate_states, self.device)
        self.goal_plate_2d_state = computeGoalPlate2DState(self.num_envs, self.goal_plate_states, self.device)

        self.obs_buf[:], self.states_buf[:] = aggregateObservationState(
                                                self.ModuleNum,
                                                self.dof_positions,
                                                self.dof_velocities,
                                                tip_states,
                                                self.plate_2d_state,
                                                self.goal_plate_2d_state,
                                                self.actions)

        if self.cfg["env"]["normalize_obs"]:
            # for normal obs
            self.obs_buf = scale_transform(
                self.obs_buf,
                lower=self.obs_lower_limit,
                upper=self.obs_upper_limit
            )        

    def computeReward(self):
        self.rew_buf[:] = 0.

        # L2 distance
        dis = self.plate_2d_state[:, 0:2] - self.goal_plate_2d_state[:, 0:2]
        dis_l2 = torch.norm(dis,  p=2, dim=-1)

        yaw_diff = self.plate_2d_state[:, 2] - self.goal_plate_2d_state[:, 2]
        # Yaw difference should be no larger than PI
        # Should be min{yaw_diff, 2PI - yaw_diff}
        yaw_diff = torch.abs(yaw_diff)
        two_pi_minus_yaw = 2*torch.pi - yaw_diff
        # Element-wise compare
        min_yaw_diff = torch.minimum(yaw_diff, two_pi_minus_yaw)

        dis_weighted = self.cfg["env"]["reward_weight"]["dis_weight"] * dis_l2
        yaw_weighted = self.cfg["env"]["reward_weight"]["yaw_weight"] * min_yaw_diff
        self.rew_buf =  dis_weighted + yaw_weighted
                          

        update_info = {
            "raw_distance_l2": dis_l2,
            "raw_yaw_diff": yaw_diff,
            "distance_penalty_avg": dis_weighted,
            "yaw_penalty_avg": yaw_weighted,
            "reward_avg": self.rew_buf
        }

        self.extras.update({"env/rewards/"+k: v.mean() for k, v in update_info.items()})

    def check_termination(self):
        # Reaching the maximum episode timesteps
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # Context related termination

    def reset_idx(self, env_ids):
        # Reset buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # Update root_state_tensors to reset object poses
        reset_plate_indices = self.plate_indices[env_ids]
        reset_goal_plate_indices = self.goal_plate_indices[env_ids]
        aggre_plate_indices = torch.cat((reset_plate_indices, reset_goal_plate_indices), 0)
        self._reset_plate_poses(aggre_plate_indices)

        # Reset poses
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                                                    gymtorch.unwrap_tensor(aggre_plate_indices.to(torch.int32)), len(aggre_plate_indices))

        # Reset robot poses
        reset_robot_indices = self.robot_indices[env_ids].to(torch.int32)
        result = self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.default_dof_tensor),
                                                gymtorch.unwrap_tensor(reset_robot_indices), len(reset_robot_indices))

    def _reset_plate_poses(self, aggre_plate_indices):
        '''
            For each env pending to be reset, reset:

            1. Robot to default poses
            2. Plate to a random pose
            3. Goal plate to a random pose
        '''
        num_indices = len(aggre_plate_indices)
        
        # For plate poses and goal plate
        pos_xs, pos_ys = random_xy(num_indices, self.MaxComDistance, self.device)
        pos_zs = torch.tensor(self.BoundaryZoffset + self.SafetyOffset, dtype=torch.float32, device=self.device).repeat(num_indices)
        quaternions = random_yaw_orientation(num_indices, self.device)
        # Update root states for plate object and goal plate
        temp_tensor = torch.zeros((num_indices, 13), dtype=torch.float32, device=self.device)
        temp_tensor[..., 0] = pos_xs
        temp_tensor[..., 1] = pos_ys
        temp_tensor[..., 2] = pos_zs
        temp_tensor[..., 3:7] = quaternions
        self.root_state_tensor[aggre_plate_indices] = temp_tensor

    def _updateRootStates(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.plate_states = self.root_state_tensor[self.plate_indices]
        self.goal_plate_states = self.root_state_tensor[self.goal_plate_indices]         

@torch.jit.script
def aggregateObservationState(module_num: int,
                              joint_angles: torch.Tensor,
                              joint_velocities: torch.Tensor,
                              tip_states: torch.Tensor,
                              plate_2d_states: torch.Tensor,
                              goal_plate_2d_states: torch.Tensor,
                              actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    obs = torch.cat([joint_angles,
                     joint_velocities,
                     tip_states.view(-1, module_num*13),
                     plate_2d_states,
                     goal_plate_2d_states], dim=-1)

    state = torch.cat([obs,
                       actions], -1)

    return obs, state


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