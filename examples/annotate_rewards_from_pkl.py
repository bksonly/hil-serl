#!/usr/bin/env python3
"""
从PKL文件标注reward数据，用于训练reward分类器。

功能：
1. 从pkl文件加载transitions，识别出连续的轨迹段（episodes）
2. 随机抽取N个完整的轨迹段
3. 按顺序播放每个轨迹段的图片，人类识别到完成时按空格键
4. 空格键之前的帧标记为failure，之后的标记为success
5. 按照train_reward_classifier.py期望的格式存入pkl

数据格式：
- PKL文件：列表，每个元素是一个transition dict
- transition: {
    "observations": {"image": np.ndarray (128, 128, 3), uint8},
    "actions": ...,
    "next_observations": {"image": ...},
    "rewards", "masks", "dones": 标准字段
  }
"""

import os
import random
import pickle as pkl
import numpy as np
import cv2
import datetime
from absl import app, flags
from typing import List, Dict

FLAGS = flags.FLAGS
flags.DEFINE_string("pkl_path", "/home/ubuntu/Desktop/hil-serl/demo_data/unplug.pkl", "Path to input PKL file containing transitions.")
flags.DEFINE_string("output_dir", "./classifier_data", "Output directory for pkl files.")
flags.DEFINE_integer("num_episodes", 5, "Number of trajectory episodes to annotate (randomly sampled).")
flags.DEFINE_integer("display_size", 512, "Display size for images (will resize from 128x128).")
flags.DEFINE_integer("frame_delay_ms", 60, "Delay between frames in milliseconds (for playback speed).")


def resize_image_for_display(img: np.ndarray, target_size: int = 512) -> np.ndarray:
    """
    Resize image for display (from 128x128 to larger size for visibility).
    
    Args:
        img: (H, W, 3) image (RGB), typically 128x128
        target_size: target display size
    
    Returns:
        Resized RGB image (target_size, target_size, 3), uint8
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    if img.shape[:2] != (target_size, target_size):
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return img.astype(np.uint8)


def load_pkl_transitions(pkl_path: str) -> List[Dict]:
    """
    从PKL文件加载所有transitions。
    
    Args:
        pkl_path: PKL文件路径
    
    Returns:
        List of transition dictionaries
    """
    with open(pkl_path, 'rb') as f:
        transitions = pkl.load(f)
    
    if not isinstance(transitions, list):
        raise ValueError(f"PKL文件应该包含一个列表，但得到了 {type(transitions)}")
    
    if len(transitions) == 0:
        raise ValueError("PKL文件为空")
    
    # 验证格式
    sample = transitions[0]
    if "observations" not in sample or "image" not in sample["observations"]:
        raise ValueError("PKL文件格式错误：每个transition应包含 observations['image']")
    
    print(f"从 {pkl_path} 加载了 {len(transitions)} 个transitions")
    return transitions


def split_into_episodes(transitions: List[Dict]) -> List[List[Dict]]:
    """
    将transitions列表分割成连续的轨迹段（episodes）。
    使用dones标志识别轨迹边界：dones=True表示轨迹结束。
    
    Args:
        transitions: List of all transitions
    
    Returns:
        List of episodes, each episode is a list of transitions
    """
    episodes = []
    current_episode = []
    
    for i, transition in enumerate(transitions):
        current_episode.append(transition)
        
        # 检查是否是轨迹结束
        is_done = transition.get("dones", False)
        if is_done or i == len(transitions) - 1:
            # 轨迹结束，保存当前episode
            if len(current_episode) > 0:
                episodes.append(current_episode)
                current_episode = []
    
    # 如果最后一个transition不是done，也要保存最后一个episode
    if len(current_episode) > 0:
        episodes.append(current_episode)
    
    print(f"识别出 {len(episodes)} 个轨迹段（episodes）")
    return episodes


def annotate_episode(
    transitions: List[Dict],
    display_size: int = 512,
    frame_delay_ms: int = 60
) -> tuple:
    """
    标注单个轨迹段（episode）：显示图片，等待空格键，返回success和failure的transitions。
    
    Args:
        transitions: List of transition dictionaries for one episode
        display_size: 显示图像尺寸
        frame_delay_ms: 帧间延迟（毫秒）
    
    Returns:
        (success_transitions, failure_transitions) 两个列表
    """
    if len(transitions) == 0:
        return [], []
    
    print(f"\n标注轨迹段")
    print(f"总transition数: {len(transitions)}")
    print("操作说明:")
    print("  - 空格键: 标记当前帧及之后为success")
    print("  - 窗口关闭: 退出程序")
    
    success_transitions = []
    failure_transitions = []
    success_frame_idx = None  # None表示还没有按空格
    
    # 显示窗口
    window_name = "Reward Annotation - Press SPACE when task succeeds"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    for i, transition in enumerate(transitions):
        # 提取图像（PKL中图像是128x128）
        img_rgb = transition["observations"]["image"]  # (128, 128, 3), RGB
        if img_rgb.dtype != np.uint8:
            img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
        
        # 放大用于显示
        img_display = resize_image_for_display(img_rgb, display_size)
        img_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)  # OpenCV 期望 BGR
        
        # 添加文字标注
        frame_text = f"Transition {i}/{len(transitions)-1}"
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
        
        font_scale = 1.5
        thickness = 3
        y1, y2 = 40, 80
        cv2.putText(img_bgr, frame_text, (10, y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        cv2.putText(img_bgr, label_text, (10, y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        cv2.imshow(window_name, img_bgr)
        
        # 等待按键（空格标记成功）
        key = cv2.waitKey(frame_delay_ms) & 0xFF
        if key == ord(' '):
            if success_frame_idx is None:
                success_frame_idx = i
                print(f"✓ 在第 {i} 个transition标记为success起点")
        
        # 创建标注后的transition（用于分类器训练）
        # 使用原始图像尺寸（128x128），保持与训练数据一致
        obs = {
            "image": transition["observations"]["image"].copy(),  # (128, 128, 3) uint8
        }
        
        # next_observations 使用transition中已有的
        next_obs = {
            "image": transition["next_observations"]["image"].copy(),
        }
        
        annotated_transition = {
            "observations": obs,
            "actions": transition["actions"].copy(),  # 保留原始actions
            "next_observations": next_obs,
            "rewards": 1.0 if (success_frame_idx is not None and i >= success_frame_idx) else 0.0,
            "masks": transition.get("masks", 1.0),  # 使用原始mask，或默认1.0
            "dones": transition.get("dones", False),  # 使用原始done标志
        }
        
        # 根据success_frame_idx分类
        if success_frame_idx is not None and i >= success_frame_idx:
            success_transitions.append(annotated_transition)
        else:
            failure_transitions.append(annotated_transition)
    
    cv2.destroyAllWindows()
    
    print(f"标注完成: {len(failure_transitions)} failure, {len(success_transitions)} success")
    
    return success_transitions, failure_transitions


def main(_):
    # 检查输入文件
    if not os.path.exists(FLAGS.pkl_path):
        raise ValueError(f"PKL文件不存在: {FLAGS.pkl_path}")
    
    # 加载所有transitions
    all_transitions = load_pkl_transitions(FLAGS.pkl_path)
    
    # 分割成轨迹段
    episodes = split_into_episodes(all_transitions)
    if len(episodes) == 0:
        raise ValueError("未识别出任何轨迹段")
    
    # 随机抽取轨迹段
    num_episodes = min(FLAGS.num_episodes, len(episodes))
    selected_episodes = random.sample(episodes, num_episodes)
    print(f"从 {len(episodes)} 个轨迹段中随机抽取 {num_episodes} 个进行标注")
    
    # 收集所有标注数据
    all_success_transitions = []
    all_failure_transitions = []
    
    try:
        for i, episode_transitions in enumerate(selected_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {i+1}/{num_episodes}")
            print(f"{'='*60}")
            
            success_trans, failure_trans = annotate_episode(
                episode_transitions,
                display_size=FLAGS.display_size,
                frame_delay_ms=FLAGS.frame_delay_ms,
            )
            
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
        pkl_basename = os.path.splitext(os.path.basename(FLAGS.pkl_path))[0]
        
        if len(all_success_transitions) > 0:
            success_file = os.path.join(FLAGS.output_dir, f"{pkl_basename}_success_images_{uuid}.pkl")
            with open(success_file, "wb") as f:
                pkl.dump(all_success_transitions, f)
            print(f"\n✓ 保存 {len(all_success_transitions)} 个success transitions到: {success_file}")
        
        if len(all_failure_transitions) > 0:
            failure_file = os.path.join(FLAGS.output_dir, f"{pkl_basename}_failure_images_{uuid}.pkl")
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
