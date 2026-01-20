#!/usr/bin/env python3
"""
从HDF5文件标注reward数据，用于训练reward分类器。

功能：
1. 从hdf5文件夹随机抽取N个.hdf5文件
2. 按顺序显示图片，人类识别到完成时按空格键
3. 空格键之前的帧标记为failure，之后的标记为success
4. 按照train_reward_classifier.py期望的格式存入pkl

数据格式：
- observations: {"image": np.ndarray (H, W, 3), uint8}
- actions: 会被train_reward_classifier.py替换，这里可以放dummy值
- next_observations: 同observations格式
- rewards, masks, dones: 标准字段
"""

import os
import glob
import random
import h5py
import pickle as pkl
import numpy as np
import cv2
import datetime
from absl import app, flags
from typing import List, Dict

FLAGS = flags.FLAGS
flags.DEFINE_string("hdf5_dir", "/home/ubuntu/Desktop/hil-serl/serl_robot_infra/xarm_env/umi/hdf5", "Directory containing HDF5 files.")
flags.DEFINE_string("output_dir", "./classifier_data", "Output directory for pkl files.")
flags.DEFINE_integer("num_episodes", 5, "Number of HDF5 episodes to annotate (randomly sampled).")
flags.DEFINE_integer("image_size", 128, "Target image size for resizing.")
flags.DEFINE_integer("frame_delay_ms", 60, "Delay between frames in milliseconds (for playback speed).")

def resize_image(img: np.ndarray, target_size: int = 128) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        img: (H, W, 3) image (RGB)
        target_size: target size
    
    Returns:
        Resized RGB image (target_size, target_size, 3), uint8
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    if img.shape[:2] != (target_size, target_size):
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return img.astype(np.uint8)

def load_hdf5_episode(hdf5_path: str, image_size: int = 128) -> List[Dict]:
    """
    从HDF5文件加载所有帧的数据。
    
    Args:
        hdf5_path: HDF5文件路径
        image_size: 目标图像尺寸
    
    Returns:
        List of frames, each frame is a dict with:
        - "image": (image_size, image_size, 3) uint8 RGB image（保存用）
        - "image_raw": (1280, 1280, 3) uint8 RGB image（展示用，原画质）
        - "frame_idx": int
    """
    frames = []
    with h5py.File(hdf5_path, 'r') as f:
        images = f['observations/front/images'][:]  # (T, 1280, 1280, 3)
        T = len(images)
        
        for t in range(T):
            img_raw = images[t]  # (1280, 1280, 3)
            if img_raw.dtype != np.uint8:
                img_raw = np.clip(img_raw, 0, 255).astype(np.uint8)
            img_resized = resize_image(img_raw, image_size)
            frames.append({
                "image": img_resized,
                "image_raw": img_raw,
                "frame_idx": t,
            })
    
    return frames

def annotate_episode(hdf5_path: str, image_size: int = 128, frame_delay_ms: int = 100) -> tuple:
    """
    标注单个episode：显示图片，等待空格键，返回success和failure的transitions。
    
    Args:
        hdf5_path: HDF5文件路径
        image_size: 目标图像尺寸
        display_scale: 显示缩放比例
        frame_delay_ms: 帧间延迟（毫秒）
    
    Returns:
        (success_transitions, failure_transitions) 两个列表
    """
    # 加载数据
    frames = load_hdf5_episode(hdf5_path, image_size)
    if len(frames) == 0:
        return [], []
    
    print(f"\n标注文件: {os.path.basename(hdf5_path)}")
    print(f"总帧数: {len(frames)}")
    print("操作说明:")
    print("  - 空格键: 标记当前帧及之后为success")
    print("  - 窗口关闭: 退出程序")
    
    success_transitions = []
    failure_transitions = []
    success_frame_idx = None  # None表示还没有按空格
    
    # 显示窗口
    window_name = "Reward Annotation - Press SPACE when task succeeds"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    for i, frame_data in enumerate(frames):
        
        # 准备显示图像：使用原始分辨率，保证清晰度
        img = frame_data["image_raw"]  # (1280, 1280, 3), 存储为RGB
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV 期望 BGR
        display_img = img_bgr
        
        # 添加文字标注
        frame_text = f"Frame {i}/{len(frames)-1}"
        if success_frame_idx is not None:
            if i >= success_frame_idx:
                label_text = "SUCCESS"
                color = (0, 255, 0)  # 绿色
            else:
                label_text = "FAILURE"
                color = (0, 0, 255)  # 红色
        else:
            label_text = "FAILURE (press SPACE when succeeds)"
            color = (0, 0, 255)  # 红色
        
        font_scale = 2.0
        thickness = 4
        y1, y2 = 60, 120
        cv2.putText(display_img, frame_text, (10, y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        cv2.putText(display_img, label_text, (10, y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        cv2.imshow(window_name, display_img)
        
        # 等待按键（空格标记成功）
        key = cv2.waitKey(frame_delay_ms) & 0xFF
        if key == ord(' '):
            if success_frame_idx is None:
                success_frame_idx = i
                print(f"✓ 在第 {i} 帧标记为success起点")
        
        # 创建transition（用于分类器训练）
        # 注意：train_reward_classifier.py会替换actions，所以这里可以放dummy值
        # 但需要确保格式正确
        obs = {
            "image": frame_data["image"].copy(),  # (H, W, 3) uint8
        }
        
        # 如果有下一帧，使用下一帧作为next_observation，否则使用当前帧
        if i < len(frames) - 1:
            next_obs = {
                "image": frames[i + 1]["image"].copy(),
            }
        else:
            next_obs = obs.copy()
        
        transition = {
            "observations": obs,
            "actions": np.zeros(7, dtype=np.float32),  # Dummy action，会被train_reward_classifier.py替换
            "next_observations": next_obs,
            "rewards": 1.0 if (success_frame_idx is not None and i >= success_frame_idx) else 0.0,
            "masks": 1.0 if i < len(frames) - 1 else 0.0,  # 最后一帧mask=0
            "dones": (i == len(frames) - 1),
        }
        
        # 根据success_frame_idx分类
        if success_frame_idx is not None and i >= success_frame_idx:
            success_transitions.append(transition)
        else:
            failure_transitions.append(transition)
    
    cv2.destroyAllWindows()
    
    print(f"标注完成: {len(failure_transitions)} failure, {len(success_transitions)} success")
    
    return success_transitions, failure_transitions

def main(_):
    global space_pressed, pause_playback, skip_episode
    
    # 检查输入目录
    if not os.path.exists(FLAGS.hdf5_dir):
        raise ValueError(f"HDF5目录不存在: {FLAGS.hdf5_dir}")
    
    # 查找所有HDF5文件
    hdf5_files = sorted(glob.glob(os.path.join(FLAGS.hdf5_dir, "*.hdf5")))
    if len(hdf5_files) == 0:
        raise ValueError(f"在 {FLAGS.hdf5_dir} 中未找到HDF5文件")
    
    # 随机抽取
    num_episodes = min(FLAGS.num_episodes, len(hdf5_files))
    selected_files = random.sample(hdf5_files, num_episodes)
    print(f"从 {len(hdf5_files)} 个文件中随机抽取 {num_episodes} 个进行标注")
    
    # 收集所有标注数据
    all_success_transitions = []
    all_failure_transitions = []
    
    try:
        for i, hdf5_path in enumerate(selected_files):
            print(f"\n{'='*60}")
            print(f"Episode {i+1}/{num_episodes}")
            print(f"{'='*60}")
            
            success_trans, failure_trans = annotate_episode(
                hdf5_path,
                image_size=FLAGS.image_size,
                frame_delay_ms=FLAGS.frame_delay_ms,
            )
            
            if success_trans is None:  # 用户关闭窗口
                print("用户中断标注")
                break
            
            all_success_transitions.extend(success_trans)
            all_failure_transitions.extend(failure_trans)
            
            print(f"累计: {len(all_failure_transitions)} failure, {len(all_success_transitions)} success")
    
    except KeyboardInterrupt:
        print("\n用户中断标注")
    finally:
        cv2.destroyAllWindows()
    
    # 保存数据
    if len(all_success_transitions) > 0 or len(all_failure_transitions) > 0:
        # 创建输出目录
        os.makedirs(FLAGS.output_dir, exist_ok=True)
        
        # 生成文件名
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = os.path.basename(FLAGS.hdf5_dir.rstrip('/'))
        
        if len(all_success_transitions) > 0:
            success_file = os.path.join(FLAGS.output_dir, f"{exp_name}_success_images_{uuid}.pkl")
            with open(success_file, "wb") as f:
                pkl.dump(all_success_transitions, f)
            print(f"\n✓ 保存 {len(all_success_transitions)} 个success transitions到: {success_file}")
        
        if len(all_failure_transitions) > 0:
            failure_file = os.path.join(FLAGS.output_dir, f"{exp_name}_failure_images_{uuid}.pkl")
            with open(failure_file, "wb") as f:
                pkl.dump(all_failure_transitions, f)
            print(f"✓ 保存 {len(all_failure_transitions)} 个failure transitions到: {failure_file}")
        
        print(f"\n总计:")
        print(f"  Success: {len(all_success_transitions)}")
        print(f"  Failure: {len(all_failure_transitions)}")
    else:
        print("\n没有数据需要保存")

if __name__ == "__main__":
    app.run(main)
