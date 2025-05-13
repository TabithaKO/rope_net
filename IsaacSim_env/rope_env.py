import numpy as np
import os
import torch
from train_logreg import RopeToJointRegressor
from isaacsim import SimulationApp
# simulation_app = SimulationApp({"headless": False}) # Comment out if importing
from pxr import UsdLux, UsdGeom, Sdf, Gf, UsdPhysics, UsdShade, PhysxSchema
import omni.physxdemos as demo
from omni.physx.scripts import physicsUtils
import omni.physx.bindings._physx as physx_bindings
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core import World
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.franka import Franka
from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf, Tf, UsdPhysics
from omni.isaac.motion_generation import LulaKinematicsSolver
from omni.isaac.core.utils.rotations import quat_to_rot_matrix
from pick_place_modified import PickPlaceController
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
from train_rope_ppo_gpu import train_isaac_sim, RopeManipulationRL
from Bimanual_data_3K.train_conditioned_reg import RopeToJointRegressor, RopeToConditionedJointRegressor



class RopeEnv():
    def __init__(self):
        self._target = None
        self.world = World()
        self.create(1, 3) # 4 or 3
        self.create_grippers()
        for _ in range(50):
            self.world.step()
        
        # First, step the simulation to make sure articulations are initialized
        

    def create(self, num_ropes, rope_length):
        self._stage = get_current_stage()
        self.world.scene.add_default_ground_plane(dynamic_friction=0.05, restitution=0.1)
        self._defaultPrimPath = "/World/"
        # configure ropes:
        self._linkHalfLength = 0.05 # 0.05 or 0.03
        self._linkRadius = 0.5 * self._linkHalfLength # 0.5 or 0.6
        self._ropeLength = rope_length
        self._numRopes = num_ropes
        self._ropeSpacing = 0.5
        self._ropeColor = np.array([1.0, 0, 0])
        self._coneAngleLimit = 110
        self._rope_damping = 0.5
        self._rope_stiffness = 0.1

        # # configure collider capsule:
        self._capsuleZ = 0.
        self._capsuleHeight = 0.
        self._capsuleRadius = 0.
        self._capsuleRestOffset = 0.
        self._capsuleColor = np.array([0, 0, 1.0])

        # physics options:
        self._contactOffset = 1
        self._physicsMaterialPath = "/World/PhysicsMaterial"
        UsdShade.Material.Define(self._stage, self._physicsMaterialPath)
        material = UsdPhysics.MaterialAPI.Apply(self._stage.GetPrimAtPath(self._physicsMaterialPath))
        material.CreateStaticFrictionAttr().Set(2.0)
        material.CreateDynamicFrictionAttr().Set(0.5)
        material.CreateRestitutionAttr().Set(0.01)

        self._createRopes()
        
    
    def create_grippers(self):
        robot_prim_path1 = "/panda1"
        robot_prim_path2 = "/panda2"

        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"


        add_reference_to_stage(path_to_robot_usd, robot_prim_path1)
        add_reference_to_stage(path_to_robot_usd, robot_prim_path2)

        self._articulation1 = self.world.scene.add(Franka(prim_path="/panda1",
                                        name="franka1", position=[-5, 0.4, 0]))
        self._articulation2 = self.world.scene.add(Franka(prim_path="/panda2",
                                        name="franka2", position=[0., 0.4, 0]))
        
        self.world.reset()
        self.controller1 = PickPlaceController(name = 'franka1', robot_articulation = self._articulation1, end_effector_initial_height = 0.07, gripper=self._articulation1._gripper)
        self.controller2 = PickPlaceController(name = 'franka2', robot_articulation = self._articulation2, end_effector_initial_height = 0.07, gripper=self._articulation2._gripper)

        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")

        kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        self.ik_solver = LulaKinematicsSolver(
            robot_description_path = kinematics_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path = kinematics_config_dir + "/franka/lula_franka_gen.urdf"
        )

    def move_gripper_with_joints(self, articulation, joint_positions,  speed_factor:float = 0.02):
        
        current_joint_positions = articulation.get_joint_positions()
        gripper_position = joint_positions[7:]
        
        
        joint_diffs = []
        for i in range(len(current_joint_positions)):
            if i < len(joint_positions):
                joint_diffs.append(joint_positions[i] - current_joint_positions[i])
            else:
                joint_diffs.append(gripper_position[i-len(joint_positions)] - current_joint_positions[i])
        
        max_diff = max([abs(diff) for diff in joint_diffs])
        num_steps = max(int(max_diff / speed_factor), 30)  # At least 30 steps for smoothness
        
        # print(f"Moving to target over {num_steps} steps")
        
        for step in range(1, num_steps + 1):
            alpha = step / num_steps
            interpolated_positions = []
            
            for i in range(len(current_joint_positions)):
                if i < len(joint_positions):
                    interpolated_positions.append(current_joint_positions[i] + alpha * joint_diffs[i])
                else:
                    interpolated_positions.append(current_joint_positions[i] + alpha * joint_diffs[i])
            
            action = ArticulationAction(joint_positions=interpolated_positions)
            articulation.apply_action(action)
            
            for _ in range(1):
                self.world.step()

    def move_gripper_with_xyz(self, articulation, target_position, target_orientation, open:bool = True, speed_factor:float = 0.02):
        """
        Move the gripper to a target position and orientation with controlled speed
        
        Args:
            articulation: The robot articulation to control
            target_position: Target position in world coordinates
            target_orientation: Target orientation in world coordinates (quaternion)
            open: Whether the gripper should be open (True) or closed (False)
            speed_factor: Controls movement speed (lower = slower, recommended range: 0.01-0.1)
        
        Returns:
            bool: Whether the movement was successful
        """
        local_target_position, local_target_orientation = self.transform_target_to_robot_frame(
            articulation, target_position, target_orientation
        )

        joint_positions, success = self.ik_solver.compute_inverse_kinematics(
            target_position=local_target_position,
            target_orientation=local_target_orientation,
            frame_name="panda_hand"
        )

        if success:
            current_joint_positions = articulation.get_joint_positions()
            gripper_position = [0.04, 0.04] if open else [0.01, 0.01]
            
            target_joint_positions = list(joint_positions) + gripper_position
            joint_diffs = []
            for i in range(len(current_joint_positions)):
                if i < len(joint_positions):
                    joint_diffs.append(joint_positions[i] - current_joint_positions[i])
                else:
                    joint_diffs.append(gripper_position[i-len(joint_positions)] - current_joint_positions[i])
            
            max_diff = max([abs(diff) for diff in joint_diffs])
            num_steps = max(int(max_diff / speed_factor), 30)  # At least 30 steps for smoothness
            
            # print(f"Moving to target over {num_steps} steps")
            
            for step in range(1, num_steps + 1):
                alpha = step / num_steps
                interpolated_positions = []
                
                for i in range(len(current_joint_positions)):
                    if i < len(joint_positions):
                        interpolated_positions.append(current_joint_positions[i] + alpha * joint_diffs[i])
                    else:
                        interpolated_positions.append(current_joint_positions[i] + alpha * joint_diffs[i])
                
                action = ArticulationAction(joint_positions=interpolated_positions)
                articulation.apply_action(action)
                
                for _ in range(2):
                    self.world.step()
            
            return target_joint_positions
        else:
            print(f"IK failed to converge with position: {target_position}, orientation: {target_orientation}")
            return None

    def _createCapsule(self, path: Sdf.Path):
        capsuleGeom = UsdGeom.Capsule.Define(self._stage, path)
        capsuleGeom.CreateHeightAttr(self._linkHalfLength)
        capsuleGeom.CreateRadiusAttr(self._linkRadius)
        capsuleGeom.CreateAxisAttr("X")
        capsuleGeom.CreateDisplayColorAttr().Set([self._ropeColor])

        UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
        massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
        massAPI.CreateDensityAttr().Set(0.00005)
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(capsuleGeom.GetPrim())
        physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
        physxCollisionAPI.CreateContactOffsetAttr().Set(self._contactOffset)
        physicsUtils.add_physics_material_to_prim(self._stage, capsuleGeom.GetPrim(), self._physicsMaterialPath)

    def _createJoint(self, jointPath):        
        joint = UsdPhysics.Joint.Define(self._stage, jointPath)

        # locked DOF (lock - low is greater than high)
        d6Prim = joint.GetPrim()
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transX")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transY")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transZ")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "rotX")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)

        # Moving DOF:
        dofs = ["rotY", "rotZ"]
        for d in dofs:
            limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, d)
            limitAPI.CreateLowAttr(-self._coneAngleLimit)
            limitAPI.CreateHighAttr(self._coneAngleLimit)

            # joint drives for rope dynamics:
            driveAPI = UsdPhysics.DriveAPI.Apply(d6Prim, d)
            driveAPI.CreateTypeAttr("force")
            driveAPI.CreateDampingAttr(self._rope_damping)
            driveAPI.CreateStiffnessAttr(self._rope_stiffness)

    def _createColliderCapsule(self):
        capsulePath = "/World/CapsuleCollider"
        capsuleGeom = UsdGeom.Capsule.Define(self._stage, capsulePath)
        capsuleGeom.CreateHeightAttr(self._capsuleHeight)
        capsuleGeom.CreateRadiusAttr(self._capsuleRadius)
        capsuleGeom.CreateAxisAttr("Y")
        capsuleGeom.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, self._capsuleZ))
        capsuleGeom.CreateDisplayColorAttr().Set([self._capsuleColor])

        # make the capsule high-quality render
        capsulePrim = capsuleGeom.GetPrim()
        capsulePrim.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool, True).Set(True)
        capsulePrim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int, True).Set(2)

        UsdPhysics.CollisionAPI.Apply(capsulePrim)
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(capsulePrim)
        physxCollisionAPI.CreateRestOffsetAttr().Set(self._capsuleRestOffset)
        physxCollisionAPI.CreateContactOffsetAttr().Set(self._contactOffset)
        physicsUtils.add_physics_material_to_prim(self._stage, capsulePrim, self._physicsMaterialPath)

    def _createRopes(self):
        linkLength = 2.0 * self._linkHalfLength - self._linkRadius
        numLinks = int(self._ropeLength / linkLength)
        xStart = -numLinks * linkLength * 0.5
        yStart = -(self._numRopes // 2) * self._ropeSpacing

        for ropeInd in range(self._numRopes):
            scopePath = f"/World/Rope{ropeInd}"
            UsdGeom.Scope.Define(self._stage, scopePath)
            
            # capsule instancer
            instancerPath = f"/World/Rope{ropeInd}/rigidBodyInstancer"
            rboInstancer = UsdGeom.PointInstancer.Define(self._stage, instancerPath)
            
            capsulePath = f"/World/Rope{ropeInd}/rigidBodyInstancer/capsule"
            self._createCapsule(capsulePath)
            
            meshIndices = []
            positions = []
            orientations = []
            
            y = yStart + ropeInd * self._ropeSpacing
            z = self._capsuleZ + self._capsuleRadius + self._linkRadius * 1.4            

            for linkInd in range(numLinks):
                meshIndices.append(0)
                x = xStart + linkInd * linkLength
                positions.append(Gf.Vec3f(x, y, z))
                orientations.append(Gf.Quath(1.0))

            meshList = rboInstancer.GetPrototypesRel()
            # add mesh reference to point instancer
            meshList.AddTarget(capsulePath)

            rboInstancer.GetProtoIndicesAttr().Set(meshIndices)
            rboInstancer.GetPositionsAttr().Set(positions)
            rboInstancer.GetOrientationsAttr().Set(orientations)
            
            # joint instancer
            jointInstancerPath = scopePath + "/jointInstancer"
            jointInstancer = PhysxSchema.PhysxPhysicsJointInstancer.Define(self._stage, jointInstancerPath)
            
            jointPath = jointInstancerPath + "/joint"
            self._createJoint(jointPath)
            
            meshIndices = []
            body0s = []
            body0indices = []
            localPos0 = []
            localRot0 = []
            body1s = []
            body1indices = []
            localPos1 = []
            localRot1 = []      
            body0s.append(instancerPath)
            body1s.append(instancerPath)

            jointX = self._linkHalfLength - 0.5 * self._linkRadius
            for linkInd in range(numLinks - 1):
                meshIndices.append(0)
                
                body0indices.append(linkInd)
                body1indices.append(linkInd + 1)
                         
                localPos0.append(Gf.Vec3f(jointX, 0, 0)) 
                localPos1.append(Gf.Vec3f(-jointX, 0, 0)) 
                localRot0.append(Gf.Quath(1.0))
                localRot1.append(Gf.Quath(1.0))

            meshList = jointInstancer.GetPhysicsPrototypesRel()
            meshList.AddTarget(jointPath)

            jointInstancer.GetPhysicsProtoIndicesAttr().Set(meshIndices)

            jointInstancer.GetPhysicsBody0sRel().SetTargets(body0s)
            jointInstancer.GetPhysicsBody0IndicesAttr().Set(body0indices)
            jointInstancer.GetPhysicsLocalPos0sAttr().Set(localPos0)
            jointInstancer.GetPhysicsLocalRot0sAttr().Set(localRot0)

            jointInstancer.GetPhysicsBody1sRel().SetTargets(body1s)
            jointInstancer.GetPhysicsBody1IndicesAttr().Set(body1indices)
            jointInstancer.GetPhysicsLocalPos1sAttr().Set(localPos1)
            jointInstancer.GetPhysicsLocalRot1sAttr().Set(localRot1)
    
    
        
    def get_rope_positions(self):
        """
        Get positions of all rope capsules in the environment.
        
        Returns:
            numpy.ndarray: Array of shape (num_ropes, num_links_per_rope, 3) containing 
                        the 3D positions of all rope capsules.
        """
        all_rope_positions = []
        
        # Get the current stage
        stage = self._stage
        
        # For each rope in the environment
        for rope_idx in range(self._numRopes):
            rope_positions = []
            
            # Path to the point instancer for this rope
            instancer_path = f"/World/Rope{rope_idx}/rigidBodyInstancer"
            instancer_prim = stage.GetPrimAtPath(instancer_path)
            
            if not instancer_prim.IsValid():
                continue
                
            # Get the point instancer
            point_instancer = UsdGeom.PointInstancer(instancer_prim)
            
            # Get positions at current time
            # Use Usd.TimeCode.Default() which represents the current state
            current_time = Usd.TimeCode.Default()
            
            # Get position attributes directly
            positions_attr = point_instancer.GetPositionsAttr()
            if positions_attr:
                positions = positions_attr.Get(current_time)
                
                # Convert positions to numpy arrays
                for position in positions:
                    # Gf.Vec3f to numpy array
                    pos_array = np.array([position[0], position[1], position[2]])
                    rope_positions.append(pos_array)
                    
            all_rope_positions.append(rope_positions)
        
        return np.array(all_rope_positions)
    
    def get_rope_point_cloud(self):
        """
        Get a flattened point cloud of all rope positions.
        
        Returns:
            numpy.ndarray: Array of shape (total_points, 3) containing all rope positions
                        in a single point cloud.
        """
        rope_positions = self.get_rope_positions()
        
        # Flatten the array to get a single point cloud
        # This converts from [num_ropes, links_per_rope, 3] to [num_ropes*links_per_rope, 3]
        point_cloud = rope_positions.reshape(-1, 3)
        
        return point_cloud
    
    def transform_target_to_robot_frame(self, robot, global_position, global_orientation):
        # Get the robot's current pose
        robot_position, robot_orientation = robot.get_world_pose()
        
        # Convert to transformation matrices
        robot_rotation = quat_to_rot_matrix(robot_orientation)
        robot_transform = np.eye(4)
        robot_transform[:3, :3] = robot_rotation
        robot_transform[:3, 3] = robot_position
        
        # Create target transform
        target_rotation = quat_to_rot_matrix(global_orientation)
        target_transform = np.eye(4)
        target_transform[:3, :3] = target_rotation
        target_transform[:3, 3] = global_position
        
        # Calculate relative transform
        robot_inv = np.linalg.inv(robot_transform)
        relative_transform = np.matmul(robot_inv, target_transform)
        
        # Extract position and orientation
        relative_position = relative_transform[:3, 3]
        
        # For simplicity, we'll use the global orientation for now
        # In a more advanced implementation, you'd convert the rotation matrix
        # back to a quaternion
        
        return relative_position, global_orientation
    
    def test_arms(self):
        print("Initializing simulation...")
        print(self.get_rope_positions())

        gripper_1 = [
            2.0,   # Joint 1 (base rotation)
            -0.8,   # Joint 2 (shoulder down)
            0.0,   # Joint 3 (shoulder sideways)
            -3.5,   # Joint 4 (elbow bend)
            0.0,   # Joint 5 (wrist rotate)
            2.0,   # Joint 6 (wrist bend)
            0.8,   # Joint 7 (end-effector rotate)import omni.timeline.scripts.python._timeline as tl
            0.04,  # Gripper left finger open
            0.04   # Gripper right finger open
        ]
        self.move_gripper_with_joints(self._articulation1, gripper_1)
        for _ in range(500):
            self.world.step()
        print("Close gripper 1")
        self.move_gripper_with_joints(self._articulation1, gripper_1)

        target_position = np.array([-0.3, 0.0, 0.13])  
        target_orientation = np.array([0.0, 1.0, 0.0, 0.0])
        self.move_gripper_with_xyz(self._articulation2, target_position, target_orientation)
        for _ in range(500):
            self.world.step()
        target_position = np.array([-0.3, 0.0, 0.13])
        self.move_gripper_with_xyz(self._articulation2, target_position, target_orientation, False)
        for _ in range(500):
            self.world.step()
        self.move_gripper_with_xyz(self._articulation2, target_position, target_orientation, False)
        for _ in range(500):
            self.world.step()
        target_position = np.array([0.5, 0.0, 0.15])  
        self.move_gripper_with_xyz(self._articulation2, target_position, target_orientation, False)
        for _ in range(500):
            self.world.step()
        target_position = np.array([-0.5, 0.0, 0.15])  
        self.move_gripper_with_xyz(self._articulation2, target_position, target_orientation, False)
        for _ in range(500):
            self.world.step()
        self.move_gripper_with_xyz(self._articulation2, target_position, target_orientation, True)
        for _ in range(500):
            self.world.step()
        target_position = np.array([-0.3, 0.0, 0.5])  
        self.move_gripper_with_xyz(self._articulation2, target_position, target_orientation, True)
        for _ in range(500):
            self.world.step()
        print(self.get_rope_positions())
        print("Simulation running, press Ctrl+C to exit")
        print(self._articulation1.get_joint_positions())
        # try:
        #     while True:
        #         self.world.step()
        # except KeyboardInterrupt:
        #     print("Shutting down simulation")
        self.close_sim()
    
    def sample_(self, num_samples, save_number):
        y_max = 0.1
        y_min = -0.3
        x_max = 0.6
        x_min = -0.6
        z_ground = 0.116
        z_air = 0.3
        y_start = 0
        target_orientation = np.array([0.0, 1.0, 0.0, 0.0])
        rope_positions = []
        joint_positions = []
        grasping_point = []
        for i in range(num_samples):
            print("Run {} of {}.".format(i, num_samples))
            current_rope = []
            current_joints = []
            current_rope.append(self.get_rope_point_cloud())
            current_joints.append(self._articulation2.get_joint_positions())
            if np.random.rand() < 0.5:
                x_min = 0.1
            else:
                x_max = -0.1
            target_position = np.array([np.random.uniform(x_min, x_max), y_start, z_air])
            next_joints = self.move_gripper_with_xyz(self._articulation2, target_position, target_orientation)
            while next_joints == None:
                target_position = np.array([np.random.uniform(x_min, x_max), y_start, z_air])
                next_joints = self.move_gripper_with_xyz(self._articulation2, target_position, target_orientation)
            current_joints.append(next_joints)
            for _ in range(10):
                self.world.step()
            target_position[2] = z_ground
            next_joints = self.move_gripper_with_xyz(self._articulation2, target_position, target_orientation)
            if next_joints == None:
                print("encountered a mistake continuing")
                continue
            for _ in range(20):
                self.world.step()
            current_joints.append(next_joints)
            next_joints[7] = 0.01
            next_joints[8] = 0.01
            current_joints.append(next_joints)
            grasping_point.append(target_position)
            _ = self.move_gripper_with_joints(self._articulation2, next_joints)
            for _ in range(50):
                self.world.step()
            x_max = 0.55
            x_min = -0.55
            target_position = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), z_ground +0.1])
            next_joints = self.move_gripper_with_xyz(self._articulation2, target_position, target_orientation, False)
            while next_joints == None:
                target_position = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), z_ground+0.1])
                next_joints = self.move_gripper_with_xyz(self._articulation2, target_position, target_orientation, False)
            current_joints.append(next_joints)
            next_joints[7] = 0.04
            next_joints[8] = 0.04
            current_joints.append(next_joints)
            _ = self.move_gripper_with_joints(self._articulation2, next_joints)
            
            for _ in range(50):
                self.world.step()
            target_position[2] = z_air
            next_joints = self.move_gripper_with_xyz(self._articulation2, target_position, target_orientation)
            current_rope.append(self.get_rope_point_cloud())
            rope_positions.append(current_rope)
            joint_positions.append(current_joints)
            self.world.reset()
            if i%save_number == 0:
                print("saving run: ", i)
                combined_rope = np.stack(rope_positions)
                np.save("rope_positions_{}.npy".format(i), combined_rope)
                combined_joint = np.stack(joint_positions)
                np.save("joint_positions_{}.npy".format(i), combined_joint)
                combined_grasp = np.stack(grasping_point)
                np.save("grasp_point_{}.npy".format(i), combined_grasp)

    def test_controller(self):
        
        picking_position = np.array([-0.11250278353691101, 0.0, 0.01])
        placing_position = np.array([-0.5, 0.0, 0.05])
        finished = False
        self._articulation2.gripper.set_joint_positions([0.04, 0.04])
        num_loops = 0
        target_positions = []
        prev_event = self.controller2._event
        while finished == False:
            num_loops += 1
            next_action = self.controller2.forward(picking_position=picking_position,placing_position=placing_position, current_joint_positions = self._articulation2.get_joint_positions())
            if prev_event != self.controller2._event:
                target_positions.append(self._articulation2.get_joint_positions())
                prev_event = self.controller2._event
            if self.controller2._event == 3:
                for _ in range(3):
                    self.world.step()
            self._articulation2.apply_action(next_action)
            finished = self.controller2.is_done()
            self.world.step()
            

        self.world.reset()
        for i in range(len(target_positions)):
            self.move_gripper_with_joints(self._articulation2,target_positions[i], 0.005)
            for _ in range(5):
                self.world.step()

    def sample_controllable(self, max_runs, save_after):
        y_max = 0.1
        y_min = -0.3
        x_max = 0.6
        x_min = -0.6
        z_ground = 0.003
        z_end = 0.05
        y_start = 0
        rope_positions = []
        joint_positions = []
        grasping_point = []
        for i in range(max_runs):
            print("Run {} of {}".format(i, max_runs))
            grasp_position = np.array([np.random.uniform(x_min, x_max), y_start, z_ground])
            target_position = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), z_end])
            grasping_point.append(grasp_position)
            self._articulation2.gripper.set_joint_positions([0.04, 0.04])
            prev_event = self.controller2._event
            rope_points = [self.get_rope_point_cloud()]
            target_positions = []
            finished = False
            while finished == False:
                next_action = self.controller2.forward(picking_position=grasp_position,placing_position=target_position, current_joint_positions = self._articulation2.get_joint_positions())
                if prev_event != self.controller2._event:
                    target_positions.append(self._articulation2.get_joint_positions())
                    
                    prev_event = self.controller2._event
                if self.controller2._event == 3 or self.controller2._event == 2:
                    for _ in range(20):
                        self.world.step()
                self._articulation2.apply_action(next_action)
                finished = self.controller2.is_done()
                self.world.step()
            rope_points.append(self.get_rope_point_cloud())
            self.world.reset()
            self.controller2.reset(0.07)
            rope_positions.append(rope_points)
            joint_positions.append(target_positions)
            if i % save_after == 0:
                print("saving run: ", i)
                combined_rope = np.stack(rope_positions)
                np.save("rope_positions_{}.npy".format(i), combined_rope)
                combined_joint = np.stack(joint_positions)
                np.save("joint_positions_{}.npy".format(i), combined_joint)
                combined_grasp = np.stack(grasping_point)
                np.save("grasp_point_{}.npy".format(i), combined_grasp)
                print(combined_grasp.shape)
                print(combined_joint.shape)
                print(combined_rope.shape)

    def wait_to_close(self):
        try:
            while True:
                self.world.step()
        except KeyboardInterrupt:
            print("Shutting down simulation")
        

