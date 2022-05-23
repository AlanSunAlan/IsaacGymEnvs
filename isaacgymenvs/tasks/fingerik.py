import os, sys
import enum
import numpy as np

from isaacgym import gymtorch
from isaacgym import gymapi

from tasks.base.vec_task import VecTask

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
gym = gymapi.acquire_gym()



class FingerIK(VecTask):
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
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

    def create_sim(self):
        '''
            Create simulation environment
            1. Configure the simulation parameters
            2. Create simulation environments
            3. Create actors from loading the robot&object urdf files
        '''    

        # Configure the simulation parameters
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        # Create environment
        self._create_envs()

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
        robot_dof_props["dirveMode"].fill(self._actuation_mode)
        robot_dof_props["stiffness"].fill(self._stiffness)
        robot_dof_props["damping"].fill(self._stiffness)
        robot_dof_props["lower"] = self._lower_limit
        robot_dof_props["upper"] = self._upper_limit

        self.envs = []

        # Grid size of each env
        spacing = 0.6
        space_lower = gymapi.Vec3(-spacing, 0, -spacing)
        space_upper = gymapi.Vec3(spacing, spacing, spacing)

        num_envs_per_row = int(np.sqrt(self.num_envs))

        # count number of shapes and bodies
        max_agg_bodies = 0
        max_agg_shapes = 0
        max_agg_bodies += self.gym.get_asset_rigid_body_count(robot_asset)
        max_agg_bodies += self.gym.get_asset_rigid_body_count(robot_asset)
        max_agg_shapes += self.gym.get_asset_rigid_shape_count(goal_obj_asset)  
        max_agg_shapes += self.gym.get_asset_rigid_shape_count(goal_obj_asset)                  

        for env_i in range(0, self.num_envs):
            env = self.gym.create_env(self.sim, space_lower, space_upper, num_envs_per_row)

            # Aggregates the robot and goal_object
            self.gym.begin_aggregate(env, max_agg_bodies, max_agg_shapes, True)

            # Create finger actor
            finger_init_pose = gymapi.Transform()
            finger_init_pose.p = gymapi.Vec3(0,0,spacing)
            finger_actor = self.gym.create_actor(env, robot_asset, finger_init_pose, 
                                                "finger", env_i, 0, 1)

            # Create goal object actor
            goal_obj_init_pose = gymapi.Transform()
            goal_obj_init_pose.p = gymapi.Vec3(0, 0, 0.1)
            goal_obj_actor = self.gym.create_actor(env, goal_obj_asset, goal_obj_init_pose,
                                                  "goal_obj", env_i, 1, 0)

            self.gym.end_aggregate(env)

    def pre_physics_step(self, actions):
        '''
            Excute actions from the NN output
        '''
        pass

    def post_physics_step(self):
        pass



