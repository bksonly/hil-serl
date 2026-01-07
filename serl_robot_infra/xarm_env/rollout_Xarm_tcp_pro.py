from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
import os
import sys
import cv2
import torch
import pickle
import argparse
import time
from datetime import date
import datetime
from model.utils import *
from matplotlib import pyplot as plt
# import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R


sys.path.append('/home/onestar/zzy/BestMan_Xarm/RoboticsToolBox/')
from Bestman_real_xarm6 import Bestman_Real_Xarm6

# parse the task name via command line
parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='pick1')
args = parser.parse_args()
task = args.task

# config
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ['DEVICE']

# def calculate_new_pose(x, y, z, quaternion, distance):
#     """
#     基于给定的6D位姿 (x, y, z, 四元数), 计算沿着 z 轴“负方向”平移 distance 后的新位姿。
#     """
#     rotation = R.from_quat(quaternion)
#     rotation_matrix = rotation.as_matrix()
#     z_axis = rotation_matrix[:, 2]        # 取出姿态矩阵的 z 轴 (第三列)
#     new_position = np.array([x, y, z]) - distance * z_axis
#     return new_position[0], new_position[1], new_position[2]


# def get_rs2_cam_rgb(pipeline):
#     frames = pipeline.wait_for_frames()
#     color_frame = frames.get_color_frame()
#     color = np.asanyarray(color_frame.get_data())
#     return color

if __name__ == "__main__":
    # init xarm
    bestman = Bestman_Real_Xarm6('192.168.1.240', None, None)

    # bestman.go_home(100) # parameter is distance

    # load the policy
    ckpt_path = os.path.join(train_cfg['checkpoint_dir'], task, train_cfg['eval_ckpt_name'])
    
    print(f'Loaded: {ckpt_path}')
    policy = make_policy(policy_config['policy_class'], policy_config)
    # loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device))["actor"])
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    print(loading_status)
    policy.to(device)
    policy.eval()
    stats_path = os.path.join(train_cfg['checkpoint_dir'], task, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    num_queries = policy_config['num_queries']


    n_rollouts = 1 # 10 trails
    extra_time = 1500 # total time = episode_len + extra_time
    for i in range(n_rollouts):

        if policy_config['temporal_agg']:
            all_time_actions = torch.zeros([cfg['episode_len']+extra_time, cfg['episode_len']+num_queries+extra_time, cfg['state_dim']]).to(device)
      
        with torch.inference_mode():
            # init buffers
            cam = cv2.VideoCapture(cfg['camera_port'])
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Warm-up frames
            for _ in range(10):
                cam.grab()
            _, image = cam.retrieve()
            cv2.imwrite("cam.png", image)
            bestman.move_end_effector_to_goal_pose([0.158, 0.28, 0.145, 180, -90, 0])
            bestman.gripper_goto_robotiq(0)
            time.sleep(0.3)
            # exit()
            print("Warm-up done.")

            print('#'*50)
            print("READY TP START!")
            print('#'*50)

            for t in range(cfg['episode_len'] + extra_time): # TODO remove redunt actions 
                # start = time.time()
                # get real time obs
                pose = bestman.get_current_end_effector_pose()
                x, y, z, roll, pitch, yaw = pose
                r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
                qx, qy, qz, qw = r.as_quat() 
                qpos = [x, y+0.03, z, qx, qy, qz, qw]
                gripper_open_width = 1 - bestman.get_gripper_position_robotiq() / 255.0
                qpos = qpos + [gripper_open_width]

                pose_deg=[x,y,z,roll/np.pi*180.0,pitch/np.pi*180.0,yaw/np.pi*180.0]
                pose_str = ', '.join([f'{x:.3f}' for x in pose_deg])
                print(f"位姿： [{pose_str}], gripper: {gripper_open_width:.3f}")

                # try:
                cam.grab()
                cam.grab()
                _, image_xv = cam.retrieve()
                cv2.imshow('RealSense', image_xv)
                cv2.waitKey(1)
                # cv2.imwrite("cam.png", image_xv)
                # time.sleep(1)
                qpos_numpy = np.array(qpos)
                qpos = pre_process(qpos_numpy)
             
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                curr_image = get_image({cn: image_xv for cn in cfg['camera_names']}, cfg['camera_names'], device)

                # if t % num_queries == 0:
                #     all_actions = policy(qpos, curr_image) 
                if policy_config['temporal_agg']:
                    if t % 1 == 0:
                        print("推理",t)
                        all_actions = policy(qpos, curr_image) 
                        all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.26 # 0.001-0.1 no difference
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)    
                else:
                    if t % 50 == 0:
                        all_actions = policy(qpos, curr_image) 
                    raw_action = all_actions[:, t % 50]
                    
                # post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
 
# ############################################################
                q_data = action[3:7]

                a = int((1-action[7]) * 255 * 1.4 + 10)
                a = np.clip(a, 0, 255)
                rotation = R.from_quat(q_data)
                euler_angles_data = rotation.as_euler('xyz', degrees=False)
                goal_pose = [action[0], action[1]-0.03, action[2], euler_angles_data[0], euler_angles_data[1], euler_angles_data[2]]
                goal_pose_str = ', '.join([f'{x:.3f}' for x in goal_pose])
                print(f"动作： [{goal_pose_str}], gripper_cmd: {action[7]:.3f}")
           
                # bestman.robot.set_state(0)
                # start_time = time.time()
                bestman.move_end_effector_to_goal_pose(goal_pose, is_radian=True)
                # print(f"机械臂耗时: {time.time() - start_time:.6f} 秒")
                # start_time = time.time()
                bestman.gripper_goto_robotiq(a, wait_motion=False)
                # print(f"夹爪耗时: {time.time() - start_time:.6f} 秒")

                time.sleep(0.4)
                # end = time.time()
                # print(f"总耗时: {end - start:.6f} 秒")
        print('#'*50)
        print("STOP!")
        print('#'*50)
