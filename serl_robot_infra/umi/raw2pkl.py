#!/usr/bin/env python3
"""
Convert raw UMI single-arm sessions directly to a unified PKL dataset.

Design goals
------------
- Input interface mirrors the previous raw2hdf5.py: accept one or more
  multi_sessions directories and process each contained session.
- Output format mirrors convert_hdf5_to_pkl.py: a single PKL file storing a
  Python list of transition dicts (one list across all sessions).
- Single-arm only. If both left/right dirs exist, the left-hand directory is
  used by default; right-hand is used otherwise; session root is a fallback.
- No imports from the deprecated raw2hdf5/raw_reader/convert_hdf5_to_pkl code.
"""

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from absl import app, flags
from scipy.spatial.transform import Rotation as R


POSE_SOURCE_MAP = {
    "merged": ("Merged_Trajectory", "merged_trajectory.txt"),
    "slam": ("SLAM_Poses", "slam_raw.txt"),
    "vive": ("Vive_Poses", "vive_data_tum.txt"),
}

# 在脚本中直接修改输入输出地址
INPUT_DIRS = [
    "/home/ubuntu/Desktop/hil-serl/serl_robot_infra/umi/data_collector_opt/DATA",
]
OUTPUT_PKL = "/home/ubuntu/Desktop/hil-serl/demo_data/unplug.pkl"

FLAGS = flags.FLAGS
flags.DEFINE_string("pose_source", "merged", "选择位姿来源 (merged/slam/vive)")
flags.DEFINE_string("frame_source", "mp4", "RGB 数据来源 (mp4/mkv)")
flags.DEFINE_integer("downsample_stride", 3, "对 timestamps.csv 进行下采样的步长")
flags.DEFINE_integer("image_size", 128, "输出图像尺寸，保持与原 pkl 一致")
flags.DEFINE_list("action_scale", [0.015, 0.1, 1.0], "动作归一化尺度 [xyz_m, rot_rad, gripper]")
flags.DEFINE_integer("num_workers", 8, "并行处理 session 的进程数")


@dataclass
class ReaderConfig:
    pose_source: str = "merged"
    downsample_stride: int = 3
    frame_source: str = "mp4"  # mp4 | mkv


def _ensure_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少文件 {label}: {path}")


def _read_pose_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = [
        "timestamp",
        "Pos X",
        "Pos Y",
        "Pos Z",
        "Q_X",
        "Q_Y",
        "Q_Z",
        "Q_W",
    ]
    return df


def _read_clamp_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = ["timestamp", "clamp"]
    return df


def _detect_single_arm_dir(session_path: str) -> str:
    """
    Prefer left_hand*, then right_hand*, otherwise return session_path.
    """
    left_dir = None
    right_dir = None
    for item in os.listdir(session_path):
        full = os.path.join(session_path, item)
        if not os.path.isdir(full):
            continue
        if item.startswith("left_hand"):
            left_dir = full
        elif item.startswith("right_hand"):
            right_dir = full
    if left_dir:
        return left_dir
    if right_dir:
        return right_dir
    return session_path


def _load_single_arm_data(
    session_path: str,
    cfg: ReaderConfig,
) -> Tuple[List[List[float]], List[np.ndarray]]:
    """
    Load images and qpos from a single-arm session directory.
    Returns (qpos_list, images_list) with matching lengths.
    """
    if cfg.pose_source not in POSE_SOURCE_MAP:
        raise ValueError(f"pose_source 必须是 {list(POSE_SOURCE_MAP.keys())} 之一")
    frame_source = cfg.frame_source.lower()
    if frame_source not in {"mp4", "mkv"}:
        raise ValueError("frame_source 仅支持 'mp4' 或 'mkv'")

    arm_root = _detect_single_arm_dir(session_path)
    pose_subdir, pose_file = POSE_SOURCE_MAP[cfg.pose_source]

    video_filename = f"video.{frame_source}"
    video_path = os.path.join(arm_root, "RGB_Images", video_filename)
    _ensure_file(video_path, "video")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频 {video_path}")

    timestamps_path = os.path.join(arm_root, "RGB_Images", "timestamps.csv")
    pose_path = os.path.join(arm_root, pose_subdir, pose_file)
    clamp_path = os.path.join(arm_root, "Clamp_Data", "clamp_data_tum.txt")

    _ensure_file(timestamps_path, "timestamps")
    _ensure_file(pose_path, "pose")
    _ensure_file(clamp_path, "clamp")

    timestamps_df = pd.read_csv(timestamps_path)
    if timestamps_df.empty:
        raise RuntimeError(f"{arm_root} timestamps.csv 为空")

    stride = max(1, int(cfg.downsample_stride))
    timestamps_df = timestamps_df.iloc[::stride].reset_index(drop=True)

    pose_df = _read_pose_file(pose_path)
    clamp_df = _read_clamp_file(clamp_path)

    pose_ts = pose_df["timestamp"].to_numpy()
    clamp_ts = clamp_df["timestamp"].to_numpy() if not clamp_df.empty else np.array([])

    images: List[np.ndarray] = []
    qpos: List[List[float]] = []

    try:
        for _, row in timestamps_df.iterrows():
            frame_idx = int(row["frame_index"])
            target_ts = float(row["header_stamp"])

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                raise RuntimeError(f"读取帧 {frame_idx} 失败: {video_path}")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(rgb_frame)

            pose_idx = int(np.argmin(np.abs(pose_ts - target_ts)))
            pose_row = pose_df.iloc[pose_idx]

            if clamp_ts.size == 0:
                clamp_val = 0.0
            else:
                clamp_idx = int(np.argmin(np.abs(clamp_ts - target_ts)))
                clamp_val = float(clamp_df.iloc[clamp_idx]["clamp"])

            state = [
                float(pose_row["Pos X"]),
                float(pose_row["Pos Y"]),
                float(pose_row["Pos Z"]),
                float(pose_row["Q_X"]),
                float(pose_row["Q_Y"]),
                float(pose_row["Q_Z"]),
                float(pose_row["Q_W"]),
                clamp_val,
            ]
            qpos.append(state)
    finally:
        if cap is not None:
            cap.release()

    return qpos, images


def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local):
    rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T_local = np.eye(4)
    T_local[:3, :3] = rotation_local
    T_local[:3, 3] = [x, y, z]

    T_base_r = np.dot(T_local[:3, :3], T_base_to_local[:3, :3])

    x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]
    rotation_base = R.from_matrix(T_base_r)
    roll_base, pitch_base, yaw_base = rotation_base.as_euler("xyz", degrees=False)
    qx_base, qy_base, qz_base, qw_base = rotation_base.as_quat()
    return x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, roll_base, pitch_base, yaw_base


def get_base_transformation():
    base_x, base_y, base_z = 0.158, 0.28, 0.145
    base_roll, base_pitch, base_yaw = np.deg2rad([179.94725, -89.999981, 0.0])

    rotation_base_to_local = R.from_euler("xyz", [base_roll, base_pitch, base_yaw]).as_matrix()
    T_base_to_local = np.eye(4)
    T_base_to_local[:3, :3] = rotation_base_to_local
    T_base_to_local[:3, 3] = [base_x, base_y, base_z]
    return T_base_to_local


def qpos_to_state(qpos: np.ndarray) -> Dict[str, np.ndarray]:
    T_base_to_local = get_base_transformation()

    x, y, z, qx, qy, qz, qw = qpos[:7]
    x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, _, _, _ = transform_to_base_quat(
        x, y, z, qx, qy, qz, qw, T_base_to_local
    )

    tcp_pose = np.array([x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base], dtype=np.float32)
    gripper_width_mm = qpos[7] if len(qpos) > 7 else 0.0
    gripper_pose = np.array([(gripper_width_mm / 88.0) * 2.0 - 1.0], dtype=np.float32)

    return {
        "tcp_pose": tcp_pose,
        "gripper_pose": gripper_pose,
    }


def resize_image(img: np.ndarray, target_size: int) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[:2] != (target_size, target_size):
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.uint8)


def flatten_state_dict(state_dict: Dict[str, np.ndarray], proprio_keys: Optional[List[str]] = None) -> np.ndarray:
    if proprio_keys is None:
        proprio_keys = ["tcp_pose", "gripper_pose"]
    parts = []
    for key in proprio_keys:
        if key not in state_dict:
            raise KeyError(f"缺少键 {key} (已有: {list(state_dict.keys())})")
        arr = state_dict[key]
        if arr.ndim > 1:
            arr = arr.flatten()
        parts.append(arr.astype(np.float32))
    return np.concatenate(parts, axis=0).astype(np.float32)


def compute_action_delta(
    curr_tcp_pose: np.ndarray,
    next_tcp_pose: np.ndarray,
    curr_gripper: float,
    next_gripper: float,
    action_scale: np.ndarray,
) -> np.ndarray:
    xyz_delta = next_tcp_pose[:3] - curr_tcp_pose[:3]

    curr_rot = R.from_quat(curr_tcp_pose[3:])
    next_rot = R.from_quat(next_tcp_pose[3:])
    delta_rot = curr_rot.inv() * next_rot
    rotvec_delta = delta_rot.as_rotvec()

    gripper_delta = next_gripper - curr_gripper

    xyz_normalized = np.clip(xyz_delta / action_scale[0], -1.0, 1.0)
    rotvec_normalized = np.clip(rotvec_delta / action_scale[1], -1.0, 1.0)
    gripper_normalized = np.clip(gripper_delta / action_scale[2], -1.0, 1.0)

    action = np.concatenate([xyz_normalized, rotvec_normalized, [gripper_normalized]])
    return action.astype(np.float32)


def session_to_transitions(
    session_path: str,
    reader_cfg: ReaderConfig,
    image_size: int,
    action_scale: np.ndarray,
) -> List[Dict]:
    qpos_list, images = _load_single_arm_data(session_path, reader_cfg)
    if len(qpos_list) < 2:
        raise RuntimeError(f"{session_path} 有效帧数不足以形成转移 (T={len(qpos_list)})")

    transitions: List[Dict] = []
    T = len(qpos_list)
    for t in range(T - 1):
        curr_state_dict = qpos_to_state(np.array(qpos_list[t]))
        next_state_dict = qpos_to_state(np.array(qpos_list[t + 1]))

        curr_image = resize_image(images[t], image_size)
        next_image = resize_image(images[t + 1], image_size)

        curr_state_flat = flatten_state_dict(curr_state_dict)
        next_state_flat = flatten_state_dict(next_state_dict)

        action = compute_action_delta(
            curr_state_dict["tcp_pose"],
            next_state_dict["tcp_pose"],
            curr_state_dict["gripper_pose"][0],
            next_state_dict["gripper_pose"][0],
            action_scale,
        )

        reward = 1.0 if (t == T - 2) else 0.0
        done = t == T - 2

        transitions.append(
            {
                "observations": {
                    "state": curr_state_flat,
                    "image": curr_image,
                },
                "actions": action,
                "next_observations": {
                    "state": next_state_flat,
                    "image": next_image,
                },
                "rewards": float(reward),
                "masks": 1.0 - float(done),
                "dones": bool(done),
                "infos": {},
            }
        )

    return transitions


def discover_sessions(input_dir: str) -> List[str]:
    return [
        os.path.join(input_dir, d)
        for d in sorted(os.listdir(input_dir))
        if os.path.isdir(os.path.join(input_dir, d)) and d.startswith("session")
    ]


def convert_all_to_pkl(
    input_dirs: List[str],
    output_pkl: str,
    reader_cfg: ReaderConfig,
    image_size: int,
    action_scale: np.ndarray,
    num_workers: int,
) -> None:
    sessions: List[str] = []
    for input_dir in input_dirs:
        found = discover_sessions(input_dir)
        if not found:
            print(f"[WARN] {input_dir} 未找到任何 session 目录，跳过")
            continue
        sessions.extend(found)

    if not sessions:
        raise FileNotFoundError("在给定的输入目录中未找到任何 session 目录")

    print(f"[INFO] 共 {len(sessions)} 个 session，开始并行处理 (workers={num_workers})")

    all_transitions: List[Dict] = []
    num_workers = max(1, int(num_workers))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_session = {
            executor.submit(session_to_transitions, session_path, reader_cfg, image_size, action_scale): session_path
            for session_path in sessions
        }
        for future in as_completed(future_to_session):
            session_path = future_to_session[future]
            try:
                transitions = future.result()
                all_transitions.extend(transitions)
                print(f"[OK] {session_path} -> {len(transitions)} transitions")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[FAIL] {session_path}: {exc}")

    output_dir = os.path.dirname(output_pkl)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[INFO] 共计 {len(all_transitions)} transitions，写入 {output_pkl}")
    with open(output_pkl, "wb") as f:
        pickle.dump(all_transitions, f)

    print("[INFO] 转换完成")
    if all_transitions:
        sample = all_transitions[0]
        print("Sample transition keys:", sample.keys())
        print("observations keys:", sample["observations"].keys())
        print("state shape:", sample["observations"]["state"].shape)
        print("image shape:", sample["observations"]["image"].shape)
        print("actions shape:", sample["actions"].shape)


def main(_):
    if FLAGS.pose_source not in POSE_SOURCE_MAP:
        raise ValueError(f"pose_source 必须是 {list(POSE_SOURCE_MAP.keys())} 之一，当前为 {FLAGS.pose_source}")
    if FLAGS.frame_source not in {"mp4", "mkv"}:
        raise ValueError(f"frame_source 仅支持 'mp4' 或 'mkv'，当前为 {FLAGS.frame_source}")
    
    reader_cfg = ReaderConfig(
        pose_source=FLAGS.pose_source,
        downsample_stride=FLAGS.downsample_stride,
        frame_source=FLAGS.frame_source,
    )
    
    action_scale = np.array([float(x) for x in FLAGS.action_scale], dtype=np.float32)
    if len(action_scale) != 3:
        raise ValueError("--action_scale 必须为 3 个值: [xyz_m, rot_rad, gripper]")
    
    convert_all_to_pkl(
        input_dirs=INPUT_DIRS,
        output_pkl=OUTPUT_PKL,
        reader_cfg=reader_cfg,
        image_size=FLAGS.image_size,
        action_scale=action_scale,
        num_workers=FLAGS.num_workers,
    )


if __name__ == "__main__":
    app.run(main)
