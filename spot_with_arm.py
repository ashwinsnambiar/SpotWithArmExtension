from typing import List, Optional, Callable, Tuple

import carb
import omni
import numpy as np
from omni.isaac.core.utils.prims import get_prim_at_path, get_current_stage
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.articulations import Articulation
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.manipulators.grippers.gripper import Gripper
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.prims.rigid_prim import RigidPrim
from pxr import Gf, UsdShade
from omni.isaac.core.prims import XFormPrim
from omni.isaac.motion_generation.motion_policy_interface import MotionPolicy
from omni.isaac.motion_generation.articulation_motion_policy import ArticulationMotionPolicy
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.physx.scripts import utils
from omni.isaac.core import SimulationContext

import time

class SpotGripper(Gripper):
    def __init__(
        self,
        end_effector_prim_path: str,
        joint_prim_names: List[str],
        joint_opened_positions: np.ndarray,
        joint_closed_positions: np.ndarray,
        action_deltas: np.ndarray = None,
    ) -> None:
        Gripper.__init__(self, end_effector_prim_path=end_effector_prim_path)
        self._joint_prim_names = joint_prim_names
        self._joint_dof_indicies = np.array([None, None])
        self._joint_opened_positions = joint_opened_positions
        self._joint_closed_positions = joint_closed_positions
        self._get_joint_positions_func = None
        self._set_joint_positions_func = None
        self._action_deltas = action_deltas
        self._articulation_num_dofs = None
        return

    # @property
    # def joint_opened_positions(self) -> np.ndarray:
    #     """
    #     Returns:
    #         np.ndarray: joint positions of the left finger joint and the right finger joint respectively when opened.
    #     """
    #     return self._joint_opened_positions

    # @property
    # def joint_closed_positions(self) -> np.ndarray:
    #     """
    #     Returns:
    #         np.ndarray: joint positions of the left finger joint and the right finger joint respectively when closed.
    #     """
    #     return self._joint_closed_positions

    # @property
    # def joint_dof_indicies(self) -> np.ndarray:
    #     """
    #     Returns:
    #         np.ndarray: joint dof indices in the articulation of the left finger joint and the right finger joint respectively.
    #     """
    #     return self._joint_dof_indicies

    # @property
    # def joint_prim_names(self) -> List[str]:
    #     """
    #     Returns:
    #         List[str]: the left finger joint prim name and the right finger joint prim name respectively.
    #     """
    #     return self._joint_prim_names

    def initialize(
        self,
        articulation_apply_action_func: Callable,
        get_joint_positions_func: Callable,
        set_joint_positions_func: Callable,
        dof_names: List,
        physics_sim_view: omni.physics.tensors.SimulationView = None,
    ) -> None:
        """Create a physics simulation view if not passed and creates a rigid prim view using physX tensor api.
            This needs to be called after each hard reset (i.e stop + play on the timeline) before interacting with any
            of the functions of this class.

        Args:
            articulation_apply_action_func (Callable): apply_action function from the Articulation class.
            get_joint_positions_func (Callable): get_joint_positions function from the Articulation class.
            set_joint_positions_func (Callable): set_joint_positions function from the Articulation class.
            dof_names (List): dof names from the Articulation class.
            physics_sim_view (omni.physics.tensors.SimulationView, optional): current physics simulation view. Defaults to None

        Raises:
            Exception: _description_
        """
        Gripper.initialize(self, physics_sim_view=physics_sim_view)
        self._get_joint_positions_func = get_joint_positions_func
        self._articulation_num_dofs = len(dof_names)
        # for index in range(len(dof_names)):
        #     if self._joint_prim_names[0] == dof_names[index]:
        #         self._joint_dof_indicies[0] = index
        #     elif self._joint_prim_names[1] == dof_names[index]:
        #         self._joint_dof_indicies[1] = index
        # # make sure that all gripper dof names were resolved
        # if self._joint_dof_indicies[0] is None or self._joint_dof_indicies[1] is None:
        #     raise Exception("Not all gripper dof names were resolved to dof handles and dof indices.")
        self._articulation_apply_action_func = articulation_apply_action_func
        current_joint_positions = get_joint_positions_func()
        if self._default_state is None:
            self._default_state = np.array(
                [
                    current_joint_positions[self._joint_dof_indicies[0]],
                    current_joint_positions[self._joint_dof_indicies[1]],
                ]
            )
        self._set_joint_positions_func = set_joint_positions_func
        return

    # def open(self) -> None:
    #     """Applies actions to the articulation that opens the gripper (ex: to release an object held)."""
    #     self._articulation_apply_action_func(self.forward(action="open"))
    #     return

    # def close(self) -> None:
    #     """Applies actions to the articulation that closes the gripper (ex: to hold an object)."""
    #     self._articulation_apply_action_func(self.forward(action="close"))
    #     return

    # def set_action_deltas(self, value: np.ndarray) -> None:
    #     """
    #     Args:
    #         value (np.ndarray): deltas to apply for finger joint positions when openning or closing the gripper.
    #                            [left, right]. Defaults to None.
    #     """
    #     self._action_deltas = value
    #     return

    # def get_action_deltas(self) -> np.ndarray:
    #     """
    #     Returns:
    #         np.ndarray: deltas that will be applied for finger joint positions when openning or closing the gripper.
    #                     [left, right]. Defaults to None.
    #     """
    #     return self._action_deltas

    # def set_default_state(self, joint_positions: np.ndarray) -> None:
    #     """Sets the default state of the gripper

    #     Args:
    #         joint_positions (np.ndarray): joint positions of the left finger joint and the right finger joint respectively.
    #     """
    #     self._default_state = joint_positions
    #     return

    # def get_default_state(self) -> np.ndarray:
    #     """Gets the default state of the gripper

    #     Returns:
    #         np.ndarray: joint positions of the left finger joint and the right finger joint respectively.
    #     """
    #     return self._default_state

    # def post_reset(self):
    #     Gripper.post_reset(self)
    #     self._set_joint_positions_func(
    #         positions=self._default_state, joint_indices=[self._joint_dof_indicies[0], self._joint_dof_indicies[1]]
    #     )
    #     return

    # def set_joint_positions(self, positions: np.ndarray) -> None:
    #     """
    #     Args:
    #         positions (np.ndarray): joint positions of the left finger joint and the right finger joint respectively.
    #     """
    #     self._set_joint_positions_func(
    #         positions=positions, joint_indices=[self._joint_dof_indicies[0], self._joint_dof_indicies[1]]
    #     )
    #     return

    # def get_joint_positions(self) -> np.ndarray:
    #     """
    #     Returns:
    #         np.ndarray: joint positions of the left finger joint and the right finger joint respectively.
    #     """
    #     return self._get_joint_positions_func(joint_indices=[self._joint_dof_indicies[0], self._joint_dof_indicies[1]])

    # def forward(self, action: str) -> ArticulationAction:
    #     """calculates the ArticulationAction for all of the articulation joints that corresponds to "open"
    #        or "close" actions.

    #     Args:
    #         action (str): "open" or "close" as an abstract action.

    #     Raises:
    #         Exception: _description_

    #     Returns:
    #         ArticulationAction: articulation action to be passed to the articulation itself
    #                             (includes all joints of the articulation).
    #     """
    #     if action == "open":
    #         target_joint_positions = [None] * self._articulation_num_dofs
    #         if self._action_deltas is None:
    #             target_joint_positions[self._joint_dof_indicies[0]] = self._joint_opened_positions[0]
    #             target_joint_positions[self._joint_dof_indicies[1]] = self._joint_opened_positions[1]
    #         else:
    #             current_joint_positions = self._get_joint_positions_func()
    #             current_left_finger_position = current_joint_positions[self._joint_dof_indicies[0]]
    #             current_right_finger_position = current_joint_positions[self._joint_dof_indicies[1]]
    #             target_joint_positions[self._joint_dof_indicies[0]] = (
    #                 current_left_finger_position + self._action_deltas[0]
    #             )
    #             target_joint_positions[self._joint_dof_indicies[1]] = (
    #                 current_right_finger_position + self._action_deltas[1]
    #             )
    #     elif action == "close":
    #         target_joint_positions = [None] * self._articulation_num_dofs
    #         if self._action_deltas is None:
    #             target_joint_positions[self._joint_dof_indicies[0]] = self._joint_closed_positions[0]
    #             target_joint_positions[self._joint_dof_indicies[1]] = self._joint_closed_positions[1]
    #         else:
    #             current_joint_positions = self._get_joint_positions_func()
    #             current_left_finger_position = current_joint_positions[self._joint_dof_indicies[0]]
    #             current_right_finger_position = current_joint_positions[self._joint_dof_indicies[1]]
    #             target_joint_positions[self._joint_dof_indicies[0]] = (
    #                 current_left_finger_position - self._action_deltas[0]
    #             )
    #             target_joint_positions[self._joint_dof_indicies[1]] = (
    #                 current_right_finger_position - self._action_deltas[1]
    #             )
    #     else:
    #         raise Exception("action {} is not defined for ParallelGripper".format(action))
    #     return ArticulationAction(joint_positions=target_joint_positions)

    # def apply_action(self, control_actions: ArticulationAction) -> None:
    #     """Applies actions to all the joints of an articulation that corresponds to the ArticulationAction of the finger joints only.

    #     Args:
    #         control_actions (ArticulationAction): ArticulationAction for the left finger joint and the right finger joint respectively.
    #     """
    #     joint_actions = ArticulationAction()
    #     if control_actions.joint_positions is not None:
    #         joint_actions.joint_positions = [None] * self._articulation_num_dofs
    #         joint_actions.joint_positions[self._joint_dof_indicies[0]] = control_actions.joint_positions[0]
    #         joint_actions.joint_positions[self._joint_dof_indicies[1]] = control_actions.joint_positions[1]
    #     if control_actions.joint_velocities is not None:
    #         joint_actions.joint_velocities = [None] * self._articulation_num_dofs
    #         joint_actions.joint_velocities[self._joint_dof_indicies[0]] = control_actions.joint_velocities[0]
    #         joint_actions.joint_velocities[self._joint_dof_indicies[1]] = control_actions.joint_velocities[1]
    #     if control_actions.joint_efforts is not None:
    #         joint_actions.joint_efforts = [None] * self._articulation_num_dofs
    #         joint_actions.joint_efforts[self._joint_dof_indicies[0]] = control_actions.joint_efforts[0]
    #         joint_actions.joint_efforts[self._joint_dof_indicies[1]] = control_actions.joint_efforts[1]
    #     self._articulation_apply_action_func(control_actions=joint_actions)
    #     return

class Spot(Robot):
    def __init__(
            self,
        prim_path: str,
        name: str = "spot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        gripper_dof_names: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        deltas: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                usd_path = assets_root_path + "/Isaac/Robots/BostonDynamics/spot/spot_with_arm.usd"
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/arm0_link_wr1"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
            if gripper_dof_names is None:
                gripper_dof_names = ["arm0_f1x"]
            # if gripper_open_position is None:
            #     gripper_open_position = np.array([0.05, 0.05]) / get_stage_units()
            # if gripper_closed_position is None:
            #     gripper_closed_position = np.array([0.0, 0.0])
        else:
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/arm0_link_wr1"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
            if gripper_dof_names is None:
                gripper_dof_names = ["arm0_f1x"]
            # if gripper_open_position is None:
            #     gripper_open_position = np.array([0.05, 0.05]) / get_stage_units()
            # if gripper_closed_position is None:
            #     gripper_closed_position = np.array([0.0, 0.0])
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        if gripper_dof_names is not None:
            if deltas is None:
                deltas = np.array([0.05, 0.05]) / get_stage_units()
            self._gripper = SpotGripper(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=gripper_dof_names,
                joint_opened_positions=gripper_open_position,
                joint_closed_positions=gripper_closed_position,
                action_deltas=deltas,
            )

        
        new_position = Gf.Vec3f(list(position))
        prim = XFormPrim("/World/spot_with_arm")
        prim.set_world_pose(position=new_position)

        self.arm_joints = ["arm0_sh1", "arm0_el0", "arm0_el1", "arm0_wr0", "arm0_wr1", "arm0_f1x"]

        return

    def find_prim_path_by_name(self, name):
        stage = get_current_stage()  # Update with your USD stage file path
        for prim in stage.Traverse():
            if prim.GetName() == name:
                return prim.GetPath().pathString
        return None

    def move_arm_joints(self, target_positions):
        for i, joint_path in enumerate(self.arm_joints):
            # Get the prim for each joint
            joint_path = self.find_prim_path_by_name(joint_path)
            joint_prim = get_prim_at_path(joint_path)
            # Set the joint position
            joint_prim.GetAttribute("drive:angular:physics:targetPosition").Set(target_positions[i])

    def get_current_arm_positions(self):
        current_positions = np.zeros((len(self.arm_joints)))
        for i, joint_path in enumerate(self.arm_joints):
            # Get the prim for each joint
            joint_path = self.find_prim_path_by_name(joint_path)
            joint_prim = get_prim_at_path(joint_path)
            current_positions[i] = joint_prim.GetAttribute("drive:angular:physics:targetPosition").Get()
        return current_positions

    # @property
    # def end_effector(self) -> RigidPrim:
    #     """[summary]

    #     Returns:
    #         RigidPrim: [description]
    #     """
    #     return self._end_effector

    # @property
    # def gripper(self) -> SpotGripper:
    #     """[summary]

    #     Returns:
    #         ParallelGripper: [description]
    #     """
    #     return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names
        )
        return

    # def post_reset(self) -> None:
    #     """[summary]"""
    #     super().post_reset()
    #     self._gripper.post_reset()
    #     self._articulation_controller.switch_dof_control_mode(
    #         dof_index=self.gripper.joint_dof_indicies[0], mode="position"
    #     )
    #     self._articulation_controller.switch_dof_control_mode(
    #         dof_index=self.gripper.joint_dof_indicies[1], mode="position"
    #     )
    #     return





class SpotMotionPolicy(MotionPolicy):
    def __init__(self) -> None:
        super().__init__()

    def compute_joint_targets(
        self,
        active_joint_positions: np.array,
        active_joint_velocities: np.array,
        watched_joint_positions: np.array,
        watched_joint_velocities: np.array,
        frame_duration: float,
    ) -> Tuple[np.array, np.array]:
        """Compute position and velocity targets for the next frame given the current robot state.
        Position and velocity targets are used in Isaac Sim to generate forces using the PD equation
        kp*(joint_position_targets-joint_positions) + kd*(joint_velocity_targets-joint_velocities).

        Args:
            active_joint_positions (np.array): current positions of joints specified by get_active_joints()
            active_joint_velocities (np.array): current velocities of joints specified by get_active_joints()
            watched_joint_positions (np.array): current positions of joints specified by get_watched_joints()
            watched_joint_velocities (np.array): current velocities of joints specified by get_watched_joints()
            frame_duration (float): duration of the physics frame

        Returns:
            Tuple[np.array,np.array]:
            joint position targets for the active robot joints for the next frame \n
            joint velocity targets for the active robot joints for the next frame
        """

        active_joint_positions = np.array([0,0,0,0,0,10])

        return active_joint_positions, np.zeros_like(active_joint_velocities)

    def get_active_joints(self) -> List[str]:
        return ["arm0_sh1", "arm0_el0", "arm0_el1", "arm0_wr0", "arm0_wr1", "arm0_f1x"]
    
    def get_watched_joints(self) -> List[str]:
        return ["fl_hx", "fr_hx", "hl_hx", "hr_hx", "fl_hy", "fl_kn", "fr_hy", "fr_kn", "hl_hy", "hl_kn", "hr_hy", "hr_kn"]
    
    def set_end_effector_target(self, target_translation=None, target_orientation=None) -> None:
        """Set end effector target.

        Args:
            target_translation (nd.array): Translation vector (3x1) for robot end effector.
                Target translation should be specified in the same units as the USD stage, relative to the stage origin.
            target_orientation (nd.array): Quaternion of desired rotation for robot end effector relative to USD stage global frame

        Returns:
            None
        """
        
    

class SpotArticulationPolicy(ArticulationMotionPolicy):
    def __init__(self, robot_articulation: Articulation, motion_policy: MotionPolicy, default_physics_dt: float = 1 / 60) -> None:
        super().__init__(robot_articulation, motion_policy, default_physics_dt)




class SpotWithArm(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 500.0
        self._world_settings["rendering_dt"] = 10.0 / 500.0
        self._base_command = [0.0, 0.0, 0.0]

    def setup_scene(self) -> None:
        self._world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
        )
        self.spot = Spot(prim_path="/World/spot_with_arm", usd_path="/home/ashwin/coding/project_sam/sam_v3.usd", name="spot", position=np.array([0,0,0.69])) 

        wrist_joint = get_prim_at_path("/World/spot_with_arm/arm0_link_wr1")
        finger_joint = get_prim_at_path("/World/spot_with_arm/arm0_link_fngr")

        rough_material = PhysicsMaterial(prim_path="/World/Physics_Materials/rough_material", static_friction=1.0, dynamic_friction=0.8)

        # mat_shade = UsdShade.Material("/World/Physics_Materials/rough_material")
        UsdShade.MaterialBindingAPI(wrist_joint).Bind(material=rough_material.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        UsdShade.MaterialBindingAPI(finger_joint).Bind(material=rough_material.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

        utils.setRigidBody(wrist_joint, approximationShape="convexDecomposition", kinematic=False)
        utils.setRigidBody(finger_joint, approximationShape="convexDecomposition", kinematic=False)

        self._cube = self._world.scene.add(DynamicSphere(prim_path="/World/sphere",
                                                        name="sphere",
                                                        position=np.array([1.1117, 0.0, 1.3301]),
                                                        scale=np.array([0.035, 0.035, 0.035]),
                                                        color=np.array([0, 0, 1.0]),
                                                        physics_material=rough_material))
        
        return

    async def setup_post_load(self):
        self.spot.initialize()
        self._world = self.get_world()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        self.sim_context = SimulationContext.instance()
        self.start_time = self.sim_context.current_time

        # print(self.spot.get_current_arm_positions())
        # # target_pos = [0,0,0,0,0,0]
        # # self.spot.move_arm_joints(target_positions=target_pos)
        # print(self.spot.get_current_arm_positions())
        # motion = SpotMotionPolicy()
        # spot_motion = SpotArticulationPolicy(robot_articulation=self.spot,
        #                                      motion_policy=motion,)
        
        
        target_pos = [10,0,0,0,0,0]
        self.spot.move_arm_joints(target_positions=target_pos)

        return
    
    async def setup_post_reset(self):
        self.start_time = self.sim_context.current_time
        target_pos = [10,0,0,0,0,0]
        self.spot.move_arm_joints(target_positions=target_pos)
        return await super().setup_post_reset()
    
    # def move_arm(self, target_positions):
    #     time.sleep(10)
    #     GetSimulationTime()
    #     self.spot.move_arm_joint(target_positions)

    def physics_step(self, step_size):
        # current_observations = self._world.get_observations()
        # if current_observations["task_event"] == 0:
        #     target_pos = [10,0,0,0,0,0]
        #     self.spot.move_arm_joints(target_pos)
        curr_time = self.sim_context.current_time        
        if curr_time - self.start_time >= 2:
            target_pos = [-10,0,0,0,0,0]
            self.spot.move_arm_joints(target_positions=target_pos)
            target_pos = [-10,0,0,0,0,-90]
            self.spot.move_arm_joints(target_positions=target_pos)
        pass