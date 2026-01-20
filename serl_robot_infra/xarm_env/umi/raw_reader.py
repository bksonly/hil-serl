import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


POSE_SOURCE_MAP = {
    "merged": ("Merged_Trajectory", "merged_trajectory.txt"),
    "slam": ("SLAM_Poses", "slam_raw.txt"),
    "vive": ("Vive_Poses", "vive_data_tum.txt"),
}


@dataclass
class ReaderConfig:
    pose_source: str = "merged"
    downsample_stride: int = 3
    frame_source: str = "mp4"


@dataclass
class ArmData:
    qpos: List[List[float]]
    action: List[List[float]]
    images: List[np.ndarray]
    timestamps: List[float]


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


def _ensure_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少文件 {label}: {path}")


def collect_single_arm_data(arm_root: str, cfg: ReaderConfig) -> ArmData:
    pose_subdir, pose_file = POSE_SOURCE_MAP[cfg.pose_source]
    frame_source = cfg.frame_source.lower()
    if frame_source not in {"mp4", "bgr8", "mkv"}:
        raise ValueError(f"frame_source 仅支持 'mp4'、'bgr8' 或 'mkv'，当前为 {cfg.frame_source!r}")

    cap = None
    bgr8_fp = None
    bgr8_meta = None
    if frame_source == "bgr8":
        rgb_root = os.path.join(arm_root, "RGB_Images")
        bgr8_path = os.path.join(rgb_root, "frames.bgr8")
        meta_path = os.path.join(rgb_root, "raw_meta.json")
        _ensure_file(bgr8_path, "frames.bgr8")
        _ensure_file(meta_path, "raw_meta")
        with open(meta_path, "r", encoding="utf-8") as f:
            bgr8_meta = json.load(f)
        width = int(bgr8_meta["width"])
        height = int(bgr8_meta["height"])
        channels = int(bgr8_meta.get("channels", 3))
        dtype = np.dtype(bgr8_meta.get("dtype", "uint8"))
        stride_bytes = int(
            bgr8_meta.get("stride_bytes", width * height * channels * dtype.itemsize)
        )
        bgr8_meta.update(
            {
                "width": width,
                "height": height,
                "channels": channels,
                "dtype": dtype,
                "stride_bytes": stride_bytes,
            }
        )
        file_size = os.path.getsize(bgr8_path)
        if file_size % stride_bytes != 0:
            raise RuntimeError(
                f"frames.bgr8 文件大小 {file_size} 不能被 stride_bytes={stride_bytes} 整除"
            )
        bgr8_meta["num_frames"] = file_size // stride_bytes
        bgr8_fp = open(bgr8_path, "rb")
    else:
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

    timestamps = pd.read_csv(timestamps_path)
    if timestamps.empty:
        raise RuntimeError(f"{arm_root} timestamps.csv 为空")
    stride = max(1, int(cfg.downsample_stride))
    timestamps = timestamps.iloc[::stride].reset_index(drop=True)

    pose_df = _read_pose_file(pose_path)
    clamp_df  = _read_clamp_file(clamp_path)

    pose_ts = pose_df["timestamp"].to_numpy()
    clamp_ts = clamp_df["timestamp"].to_numpy() if not clamp_df.empty else np.array([])



    images: List[np.ndarray] = []
    qpos: List[List[float]] = []
    actions: List[List[float]] = []
    record_timestamps: List[float] = []

    try:
        for _, row in timestamps.iterrows():
            frame_idx = int(row["frame_index"])
            target_ts = float(row["header_stamp"])

            if frame_source == "bgr8":
                if frame_idx < 0 or frame_idx >= bgr8_meta["num_frames"]:
                    raise IndexError(
                        f"frame_index {frame_idx} 超出 bgr8 帧数范围 [0, {bgr8_meta['num_frames']})"
                    )
                offset = frame_idx * bgr8_meta["stride_bytes"]
                bgr8_fp.seek(offset)
                raw = bgr8_fp.read(bgr8_meta["stride_bytes"])
                if len(raw) != bgr8_meta["stride_bytes"]:
                    raise RuntimeError(f"读取 bgr8 帧 {frame_idx} 失败")
                frame = np.frombuffer(raw, dtype=bgr8_meta["dtype"])
                frame = frame.reshape(
                    (bgr8_meta["height"], bgr8_meta["width"], bgr8_meta["channels"])
                )
            else:
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
# 写入三个时间戳供后续使用
            rocord_timestamp =  [
                float(target_ts), #image timestamp
                float(pose_row["timestamp"]), #pose timestamp
                clamp_ts[clamp_idx] if clamp_ts.size > 0 else -1.0,#clamp timestamp
            ]
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
            actions.append(state.copy())
            record_timestamps.append(rocord_timestamp)
    finally:
        if cap is not None:
            cap.release()
        if bgr8_fp is not None:
            bgr8_fp.close()

    return ArmData(
        qpos=qpos,
        action=actions,
        images=images,
        timestamps=record_timestamps,
    )


def detect_layout(session_path: str) -> Tuple[str, Dict[str, str]]:
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
    if left_dir and right_dir:
        return "dual", {"left": left_dir, "right": right_dir}
    if left_dir and not right_dir:
        return "single", {"single": left_dir}
    if right_dir and not left_dir:
        return "single", {"single": right_dir}
    return "single", {"single": session_path}


def extract_session_index(session_path: str) -> int:
    session_name = os.path.basename(session_path.rstrip(os.sep))
    digits = re.findall(r"\d+", session_name)
    if not digits:
        raise ValueError(f"无法从 {session_name} 提取序号")
    return int(digits[-1])


def discover_sessions(input_dir: str) -> List[str]:
    return [
        os.path.join(input_dir, d)
        for d in sorted(os.listdir(input_dir))
        if os.path.isdir(os.path.join(input_dir, d)) and d.startswith("session")
    ]

