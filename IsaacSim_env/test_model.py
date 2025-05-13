from rope_env import RopeEnv
import numpy as np

ropes = RopeEnv()
pred_joints = np.load('infer_conditioned_9.npy')
grasp_pred = np.load('grasp_predictions_conditioned.npy')
true_joints = np.load('joint_positions_9.npy')
true_grasp = np.load('grasp_point_9.npy')            
for k in range(pred_joints.shape[0]):
    for i in range(400):
        ropes.world.step()
    for i in range(pred_joints.shape[1]):
        print(i, " I")
        if i == 2 or i == 4:
            move = pred_joints[1,i,:]
            if i == 2:
                move[-1] = 0.04
                move[-2] = 0.04
            else:
                move[-1] = 0.01
                move[-2] = 0.01
            ropes.move_gripper_with_joints(ropes._articulation2,move)
            for j in range(50):
                ropes.world.step()
        pred_joints[1,i,-1] = max(pred_joints[1,i,-1], 0.01)
        pred_joints[1,i,-2] = max(pred_joints[1,i,-2], 0.01)
        ropes.move_gripper_with_joints(ropes._articulation2,pred_joints[1,i,:])
        for j in range(50):
            ropes.world.step()
    ropes.world.reset()
    for i in range(pred_joints.shape[1]):
        print(i)
        if i == 2 or i == 4 or i ==3:
            move = true_joints[1,i,:]
            if i == 2:
                move[-1] = 0.04
                move[-2] = 0.04
            else:
                move[-1] = 0.01
                move[-2] = 0.01
            ropes.move_gripper_with_joints(ropes._articulation2,move)
            for j in range(50):
                ropes.world.step()
        true_joints[1,i,-1] = max(true_joints[1,i,-1], 0.01)
        true_joints[1,i,-2] = max(true_joints[1,i,-2], 0.01)
        ropes.move_gripper_with_joints(ropes._articulation2,true_joints[1,i,:])
        for j in range(50):
            ropes.world.step()
    ropes.world.reset()
