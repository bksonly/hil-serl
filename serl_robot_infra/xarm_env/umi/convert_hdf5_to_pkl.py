#!/usr/bin/env python3
"""
(hilserl_cpu) ubuntu@ubuntu-System-Product-Name:~/Desktop/hil-serl/serl_robot_infra/xarm_env/umi/hdf5$ h5ls -r episode_000.hdf5 
/                        Group
/action                  Dataset {149, 8}
/observations            Group
/observations/front      Group
/observations/front/images Dataset {149, 1280, 1280, 3} (RGB)
/observations/qpos       Dataset {149, 8} (x, y, z, qx, qy, qz, qw, gripper_width_mm)


Output pkl format (single pkl file containing a Python list of transitions):
- Each transition (time step) is a dict:
  {
    "observations": {
        "state": np.ndarray shape (8,), float32
            # Flattened concatenation: [tcp_pose(7), gripper_pose(1)]
            # tcp_pose: [x, y, z, qx, qy, qz, qw] in base coordinates, meters + unit quaternion (absolute)
            # gripper_pose: normalized gripper position in [-1, 1] (from width 0-88mm -> [-1, 1])
        "image": np.ndarray shape (H, W, 3), uint8
            # RGB image resized to image_size (default 128)
    },
    "actions": np.ndarray shape (7,), float32
        # normalized deltas in [-1, 1]:
        # action[:3]   -> xyz_delta / action_scale[0], meters
        # action[3:6]  -> rotvec_delta / action_scale[1], radians
        # action[6]    -> gripper_delta / action_scale[2], normalized gripper delta
    "next_observations": {... same structure as observations ...},
    "rewards": float,
    "masks": float,      # 1.0 for non-terminal, 0.0 at terminal
    "dones": bool,       # True at terminal transition
    "infos": dict        # empty by default
  }
"""

import os
import glob
import h5py
import pickle as pkl
import numpy as np
import cv2
import multiprocessing as mp
from tqdm import tqdm
from absl import app, flags
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation as R

IMAGE_SIZE = 128  # original pkl uses 128x128 RGB


def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local):
    """
    Transform pose from local coordinate system to base coordinate system.

    Args:
        x, y, z: position in local coordinate system
        qx, qy, qz, qw: quaternion in local coordinate system
        T_base_to_local: 4x4 transformation matrix from base to local

    Returns:
        x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, roll_base, pitch_base, yaw_base
    """
    rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T_local = np.eye(4)
    T_local[:3, :3] = rotation_local
    T_local[:3, 3] = [x, y, z]

    T_base_r = np.dot(T_local[:3, :3], T_base_to_local[:3, :3])

    x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]
    rotation_base = R.from_matrix(T_base_r)
    roll_base, pitch_base, yaw_base = rotation_base.as_euler('xyz', degrees=False)
    qx_base, qy_base, qz_base, qw_base = rotation_base.as_quat()
    return x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, roll_base, pitch_base, yaw_base


def get_base_transformation():
    """
    Get transformation matrix from base to local coordinate system.
    Hard-coded parameters from the old script.
    """
    # Base point in local coordinate system
    base_x, base_y, base_z = 0.158, 0.28, 0.145
    base_roll, base_pitch, base_yaw = np.deg2rad([179.94725, -89.999981, 0.0])

    rotation_base_to_local = R.from_euler('xyz', [base_roll, base_pitch, base_yaw]).as_matrix()
    T_base_to_local = np.eye(4)
    T_base_to_local[:3, :3] = rotation_base_to_local
    T_base_to_local[:3, 3] = [base_x, base_y, base_z]
    return T_base_to_local
FLAGS = flags.FLAGS
flags.DEFINE_string("hdf5_dir", "/home/ubuntu/Desktop/hil-serl/serl_robot_infra/xarm_env/umi/hdf5", "Directory containing HDF5 files.")
flags.DEFINE_string("output_pkl", "/home/ubuntu/Desktop/hil-serl/demo_data/unplug.pkl", "Output pkl file path.")
flags.DEFINE_list("action_scale", [0.015, 0.1, 1.0], "Action scale for [xyz_delta_m, rot_delta_rad, gripper]. Used to normalize deltas to [-1, 1].")
flags.DEFINE_integer("num_workers", 28, "Parallel workers for conversion (default: cpu_count()//2).")


def resize_image(img: np.ndarray, target_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Resize image from 1280x1280 to target_size x target_size.
    
    Args:
        img: (H, W, 3) image (RGB)
        target_size: target size (default 128)
    
    Returns:
        Resized RGB image (target_size, target_size, 3), uint8
    """
    # Ensure image is uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Resize if needed
    if img.shape[:2] != (target_size, target_size):
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return img.astype(np.uint8)


def flatten_state_dict(state_dict: Dict[str, np.ndarray], proprio_keys: List[str] = None) -> np.ndarray:
    """
    Flatten state dictionary to 1D array, matching SERLObsWrapper behavior.

    Args:
        state_dict: Dictionary with keys like "tcp_pose", "gripper_pose"
        proprio_keys: Keys in order to concatenate (default: ["tcp_pose", "gripper_pose"] - no tcp_vel for UMI data)

    Returns:
        Flattened 1D array, float32
    """
    if proprio_keys is None:
        proprio_keys = ["tcp_pose", "gripper_pose"]  # Removed tcp_vel as UMI data doesn't have it
    
    parts = []
    for key in proprio_keys:
        if key not in state_dict:
            raise KeyError(f"Missing key '{key}' in state_dict. Available keys: {list(state_dict.keys())}")
        arr = state_dict[key]
        # Ensure it's a 1D array
        if arr.ndim > 1:
            arr = arr.flatten()
        parts.append(arr.astype(np.float32))
    
    return np.concatenate(parts, axis=0).astype(np.float32)


def _convert_single(args: Tuple[str, List[float]]):
    """
    Helper for multiprocessing: convert one HDF5 file to transitions.
    Returns either a list of transitions or ("error", path, message).
    """
    hdf5_path, action_scale_list = args
    try:
        action_scale = np.array(action_scale_list, dtype=np.float32)
        return convert_hdf5_episode_to_transitions(
            hdf5_path,
            action_scale=action_scale,
        )
    except Exception as e:
        return ("error", hdf5_path, str(e))


def compute_action_delta(
    curr_tcp_pose: np.ndarray,
    next_tcp_pose: np.ndarray,
    curr_gripper: float,
    next_gripper: float,
    action_scale: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized action delta from current and next poses.
    
    HIL-SERL expects action to be normalized deltas in [-1, 1]:
    - action[:3]: xyz delta (normalized)
    - action[3:6]: rotation delta as rotvec (normalized)
    - action[6]: gripper delta (normalized)
    
    Args:
        curr_tcp_pose: (7,) current tcp pose [x, y, z, qx, qy, qz, qw]
        next_tcp_pose: (7,) next tcp pose [x, y, z, qx, qy, qz, qw]
        curr_gripper: current gripper pose (normalized [-1, 1])
        next_gripper: next gripper pose (normalized [-1, 1])
        action_scale: (3,) [xyz_scale_m, rot_scale_rad, gripper_scale]
    
    Returns:
        (7,) normalized action delta
    """
    # Compute xyz delta (in meters)
    xyz_delta = next_tcp_pose[:3] - curr_tcp_pose[:3]
    
    # Compute rotation delta as rotvec (in radians)
    curr_rot = R.from_quat(curr_tcp_pose[3:])
    next_rot = R.from_quat(next_tcp_pose[3:])
    # Relative rotation: next_rot = curr_rot * delta_rot
    # So: delta_rot = curr_rot.inv() * next_rot
    delta_rot = curr_rot.inv() * next_rot
    rotvec_delta = delta_rot.as_rotvec()
    
    # Compute gripper delta
    gripper_delta = next_gripper - curr_gripper
    
    # Normalize deltas to [-1, 1] using action_scale
    xyz_normalized = np.clip(xyz_delta / action_scale[0], -1.0, 1.0)
    rotvec_normalized = np.clip(rotvec_delta / action_scale[1], -1.0, 1.0)
    gripper_normalized = np.clip(gripper_delta / action_scale[2], -1.0, 1.0)
    
    # Combine into 7D action
    action = np.concatenate([xyz_normalized, rotvec_normalized, [gripper_normalized]])
    
    return action.astype(np.float32)


def qpos_to_state(qpos: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convert qpos to state observation, transforming coordinates from UMI local to base frame.

    Qpos format: [x, y, z, qx, qy, qz, qw, gripper_width_mm] in UMI local coordinates

    Args:
        qpos: (8,) array - [x, y, z, qx, qy, qz, qw, gripper_width_mm]

    Returns:
        Dict with state keys matching HIL-SERL format:
        {
            "tcp_pose": (7,) [x, y, z, qx, qy, qz, qw] in base coordinates,
            "gripper_pose": (1,) normalized gripper [-1, 1]
        }
        Note: tcp_vel removed as UMI data doesn't have velocity info
    """
    # Get transformation matrix
    T_base_to_local = get_base_transformation()

    # Extract raw pose from qpos (UMI local coordinates)
    x, y, z, qx, qy, qz, qw = qpos[:7]

    # Transform to base coordinates
    x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, _, _, _ = transform_to_base_quat(
        x, y, z, qx, qy, qz, qw, T_base_to_local
    )

    # Create tcp_pose in base coordinates
    tcp_pose = np.array([x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base], dtype=np.float32)

    # Extract and normalize gripper (from [0, 88]mm to [-1, 1])
    gripper_width_mm = qpos[7] if len(qpos) > 7 else 0.0
    gripper_pose = np.array([(gripper_width_mm / 88.0) * 2.0 - 1.0], dtype=np.float32)

    return {
        "tcp_pose": tcp_pose,
        "gripper_pose": gripper_pose,
    }


def convert_hdf5_episode_to_transitions(
    hdf5_path: str,
    action_scale: np.ndarray = None,
) -> List[Dict]:
    """
    Convert a single HDF5 episode file to list of transitions.
    
    Args:
        hdf5_path: Path to HDF5 file
        action_scale: (3,) action scale for normalization [xyz_scale_m, rot_scale_rad, gripper_scale]
    
    Returns:
        List of transition dictionaries
    """
    if action_scale is None:
        action_scale = np.array([0.015, 0.1, 1.0])  # Default values
    
    transitions = []
    
    with h5py.File(hdf5_path, 'r') as f:
        # Load data
        actions_hdf5 = f['action'][:]  # (T, 8) - absolute poses
        images = f['observations/front/images'][:]  # (T, 1280, 1280, 3)
        qpos_hdf5 = f['observations/qpos'][:]  # (T, 8)
        
        T = len(actions_hdf5)
        
        # Process each timestep
        for t in range(T - 1):
            # Current observation
            curr_state_dict = qpos_to_state(qpos_hdf5[t])
            curr_image = resize_image(images[t], IMAGE_SIZE)
            
            # Flatten state dict to match SERLObsWrapper format
            curr_state_flat = flatten_state_dict(curr_state_dict)
            
            obs = {
                "state": curr_state_flat,  # Flattened 1D array, not dict
                "image": curr_image,  # Single camera, key is "image"
            }
            
            # Next observation
            next_state_dict = qpos_to_state(qpos_hdf5[t + 1])
            next_image = resize_image(images[t + 1], IMAGE_SIZE)
            
            # Flatten state dict to match SERLObsWrapper format
            next_state_flat = flatten_state_dict(next_state_dict)
            
            next_obs = {
                "state": next_state_flat,  # Flattened 1D array, not dict
                "image": next_image,
            }
            
            # Compute action delta from current to next pose
            # Note: actions_hdf5 contains absolute poses, but we need deltas
            # We'll use the state's tcp_pose which comes from qpos
            curr_tcp_pose = curr_state_dict["tcp_pose"]  # (7,) [x, y, z, qx, qy, qz, qw]
            next_tcp_pose = next_state_dict["tcp_pose"]  # (7,) [x, y, z, qx, qy, qz, qw]
            curr_gripper = curr_state_dict["gripper_pose"][0]  # scalar
            next_gripper = next_state_dict["gripper_pose"][0]  # scalar
            
            action = compute_action_delta(
                curr_tcp_pose,
                next_tcp_pose,
                curr_gripper,
                next_gripper,
                action_scale,
            )
            
            # Reward: simple success reward at the end
            # For pick-and-place, you might want to add intermediate rewards
            # For now, we'll set reward=1 at the last step, 0 otherwise
            reward = 1.0 if (t == T - 2) else 0.0
            
            # Done: True only at the last step
            done = (t == T - 2)
            
            # Create transition
            transition = {
                "observations": obs,
                "actions": action,  # Already float32 from compute_action_delta
                "next_observations": next_obs,
                "rewards": float(reward),
                "masks": 1.0 - float(done),
                "dones": bool(done),
                "infos": {},  # Empty for now
            }
            
            transitions.append(transition)
    
    return transitions


def convert_all_hdf5_to_pkl(
    hdf5_dir: str,
    output_pkl: str,
    action_scale: np.ndarray = None,
    num_workers: int = None,
):
    """
    Convert all HDF5 files in a directory to a single pkl file.
    
    Args:
        hdf5_dir: Directory containing HDF5 files
        output_pkl: Output pkl file path
        action_scale: (3,) action scale for normalization [xyz_scale_m, rot_scale_rad, gripper_scale]
        num_workers: parallel workers (default cpu_count()//2, at least 1)
    """
    if action_scale is None:
        action_scale = np.array([0.015, 0.1, 1.0], dtype=np.float32)  # Default values
    else:
        action_scale = np.array(action_scale, dtype=np.float32)

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)
    else:
        num_workers = max(1, int(num_workers))
    # Find all HDF5 files
    hdf5_files = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))
    
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {hdf5_dir}")
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    # Convert all episodes
    all_transitions = []
    
    tasks = [(hdf5_file, action_scale.tolist()) for hdf5_file in hdf5_files]
    with mp.Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_convert_single, tasks),
            total=len(tasks),
            desc=f"Converting episodes (workers={num_workers})",
        ):
            if isinstance(result, tuple) and len(result) == 3 and result[0] == "error":
                _, path, msg = result
                print(f"Error processing {path}: {msg}")
                continue
            all_transitions.extend(result)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_pkl)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save to pkl
    print(f"Saving {len(all_transitions)} transitions to {output_pkl}")
    with open(output_pkl, "wb") as f:
        pkl.dump(all_transitions, f)
    
    print(f"✅ Conversion complete!")
    print(f"   Total transitions: {len(all_transitions)}")
    print(f"   Total episodes: {len(hdf5_files)}")
    print(f"   Average transitions per episode: {len(all_transitions) / len(hdf5_files):.1f}")
    
    # Print sample transition info
    if all_transitions:
        sample = all_transitions[0]
        print(f"\nSample transition structure:")
        print(f"   observations keys: {list(sample['observations'].keys())}")
        # state is now a flattened 1D array (not a dict)
        state_arr = sample['observations']['state']
        print(f"   observations['state'] shape: {state_arr.shape}, dtype: {state_arr.dtype}")
        print(f"   observations['state'] (flattened: tcp_pose[7] + gripper_pose[1] = {len(state_arr)} dims)")
        print(f"   observations['image'] shape: {sample['observations']['image'].shape}, dtype: {sample['observations']['image'].dtype}")
        print(f"   actions shape: {sample['actions'].shape}, dtype: {sample['actions'].dtype}")


def main(_):
    if FLAGS.hdf5_dir is None:
        raise ValueError("--hdf5_dir must be specified")
    
    if not os.path.exists(FLAGS.hdf5_dir):
        raise ValueError(f"Directory {FLAGS.hdf5_dir} does not exist")
    
    # Parse action_scale from flags
    action_scale = np.array([float(x) for x in FLAGS.action_scale])
    if len(action_scale) != 3:
        raise ValueError("--action_scale must have 3 values: [xyz_scale_m, rot_scale_rad, gripper_scale]")
    
    convert_all_hdf5_to_pkl(
        hdf5_dir=FLAGS.hdf5_dir,
        output_pkl=FLAGS.output_pkl,
        action_scale=action_scale,
        num_workers=FLAGS.num_workers,
    )


if __name__ == "__main__":
    app.run(main)

