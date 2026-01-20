import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

from raw_reader import (
    ArmData,
    ReaderConfig,
    collect_single_arm_data,
    detect_layout,
    discover_sessions,
    extract_session_index,
)


DEFAULT_READER_CONFIG = ReaderConfig()
DEFAULT_HDF5_CONFIG = {
    "camera_dataset_name": "images",
    "session_workers": 32,
}


def write_single_episode(output_path: str, data: ArmData, camera_dataset_name: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w", rdcc_nbytes=2 * 1024 ** 2) as root:
        root.attrs["sim"] = False
        obs_grp = root.create_group("observations")
        obs_grp.create_dataset("qpos", data=np.array(data.qpos, dtype=np.float32))
        imgs = np.array(data.images, dtype=np.uint8)

        pose_image_time_diff = np.array(data.timestamps)[:, 1] - np.array(data.timestamps)[:, 0]
        pose_image_time_diff = np.abs(pose_image_time_diff)
        print(f"{os.path.basename(output_path)} Pose-Image 时间戳差 平均值: {pose_image_time_diff.mean()*1000}ms, 最大值: {pose_image_time_diff.max()*1000}ms，最大值出现在第{np.argmax(pose_image_time_diff)}帧")

        front_grp = obs_grp.create_group("front")
        front_grp.create_dataset(
            camera_dataset_name,
            data=imgs,
            compression="gzip",
            compression_opts=4,
        )
        root.create_dataset("action", data=np.array(data.action, dtype=np.float32))


def write_dual_episode(output_path: str, left: ArmData, right: ArmData, camera_dataset_name: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w", rdcc_nbytes=2 * 1024 ** 2) as root:
        root.attrs["sim"] = False
        for name, data in [("robot_0", left), ("robot_1", right)]:
            group = root.create_group(name)
            obs_grp = group.create_group("observations")
            obs_grp.create_dataset("qpos", data=np.array(data.qpos, dtype=np.float32))
            imgs = np.array(data.images, dtype=np.uint8)

            
            pose_image_time_diff = np.array(data.timestamps)[:, 1] - np.array(data.timestamps)[:, 0]
            pose_image_time_diff = np.abs(pose_image_time_diff)
            print(f"{os.path.basename(output_path)} {name} Pose-Image 时间戳差 平均值: {pose_image_time_diff.mean()*1000}ms, 最大值: {pose_image_time_diff.max()*1000}ms")

            front_grp = obs_grp.create_group("front")
            front_grp.create_dataset(
                camera_dataset_name,
                data=imgs,
                compression="gzip",
                compression_opts=4,
            )
            group.create_dataset("action", data=np.array(data.action, dtype=np.float32))


def process_session(
    session_path: str,
    output_root: str,
    reader_cfg: ReaderConfig,
    camera_dataset_name: str,
    episode_index: int,
) -> Tuple[bool, Optional[str]]:
    """Process a single session and write output using a global episode index.

    The extra `episode_index` parameter is used to ensure unique, non-colliding
    filenames across multiple input directories.
    """
    try:
        mode, arms = detect_layout(session_path)
        if mode == "dual":
            left_data = collect_single_arm_data(arms["left"], reader_cfg)
            right_data = collect_single_arm_data(arms["right"], reader_cfg)
        else:
            left_data = collect_single_arm_data(next(iter(arms.values())), reader_cfg)
            right_data = None

        output_path = os.path.join(output_root, f"episode_{episode_index:03d}.hdf5")
        if mode == "dual" and right_data is not None:
            write_dual_episode(output_path, left_data, right_data, camera_dataset_name)
        else:
            write_single_episode(output_path, left_data, camera_dataset_name)
        return True, None
    except Exception as exc:  # pylint: disable=broad-except
        return False, str(exc)


def convert_all(input_dirs: list, output_dir: str, reader_cfg: ReaderConfig, hdf5_cfg: Dict) -> None:
    """Convert sessions from multiple input directories with global indexing.

    Sessions are collected from each input directory in the order they are
    provided; a global episode index is assigned sequentially to avoid name
    collisions between different input directories.
    """
    sessions = []
    for input_dir in input_dirs:
        found = discover_sessions(input_dir)
        if not found:
            print(f"[WARN] {input_dir} 未找到任何 session 目录，跳过")
            continue
        sessions.extend(found)

    if not sessions:
        raise FileNotFoundError("在给定的输入目录中未找到任何 session 目录")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] 共 {len(sessions)} 个 session（跨所有输入目录），输出目录 {output_dir}")

    with ProcessPoolExecutor(max_workers=hdf5_cfg["session_workers"]) as executor:
        futures = {
            executor.submit(
                process_session,
                session_path,
                output_dir,
                reader_cfg,
                hdf5_cfg["camera_dataset_name"],
                idx,
            ): session_path
            for idx, session_path in enumerate(sessions)
        }
        for future in as_completed(futures):
            session_path = futures[future]
            ok, message = future.result()
            if ok:
                print(f"[OK] {session_path}")
            else:
                print(f"[FAIL] {session_path}: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将单臂/双臂原始数据转换为 HDF5 episode 文件",
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="一个或多个 raw multi_sessions 目录，按给定顺序处理",
    )
    parser.add_argument("output_dir", help="HDF5 输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader_cfg = DEFAULT_READER_CONFIG
    hdf5_cfg = DEFAULT_HDF5_CONFIG.copy()
    # Collect sessions across all provided input directories and convert with
    # global indexing to avoid filename collisions.
    print(f"[INFO] 开始处理 {len(args.input_dirs)} 个输入目录")
    convert_all(args.input_dirs, args.output_dir, reader_cfg, hdf5_cfg)


if __name__ == "__main__":
    main()

