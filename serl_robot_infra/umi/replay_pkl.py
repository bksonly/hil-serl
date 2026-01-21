#!/usr/bin/env python3
"""
Replay PKL trajectories on real XArm robot for validation.

This script loads a PKL file containing transitions and executes the actions
on the real XArm robot to visually verify that the data conversion is correct.

Default behavior:
- Continuous replay (not step-by-step)
- Position control mode (not servo)
- Random trajectory selection
- 0.5s delay between actions

Usage:
    python replay_pkl.py --pkl_path=/path/to/demo.pkl
    python replay_pkl.py --pkl_path=/path/to/demo.pkl --random_trajectory=5 --delay=1.0
    python replay_pkl.py --pkl_path=/path/to/demo.pkl --step_by_step --use_servo
"""

import os
import time
import pickle as pkl
import numpy as np
from absl import app, flags
from typing import List, Dict, Tuple, Any

from serl_robot_infra.xarm_env import XArmEnv, XArmEnvConfig

FLAGS = flags.FLAGS
flags.DEFINE_string("pkl_path", "/home/ubuntu/Desktop/hil-serl/demo_data/unplug.pkl", "Path to PKL file containing transitions.")
flags.DEFINE_float("delay", 0.2, "Delay between actions in seconds.")
flags.DEFINE_integer("max_steps", None, "Maximum number of steps to replay (None for all).")
flags.DEFINE_boolean("reset_at_start", True, "Whether to reset robot to initial pose at start.")
flags.DEFINE_boolean("use_servo", False, "Whether to use servo mode (True) or position mode (False). Default: position mode.")
flags.DEFINE_enum("replay_mode", "pos", ["pos", "delta"], "Replay mode: 'pos' uses absolute tcp_pose from PKL; 'delta' uses PKL action deltas.")

def _split_into_trajectories(transitions: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Split a flat list of transitions into trajectories using `dones` or `masks`.
    - `dones` True ends an episode
    - `masks` == 0 also indicates terminal transition in this repo
    """
    trajectories: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for tr in transitions:
        current.append(tr)
        done = bool(tr.get("dones", False))
        mask = tr.get("masks", 1.0)
        try:
            mask_terminal = float(mask) <= 0.0
        except Exception:
            mask_terminal = False
        if done or mask_terminal:
            trajectories.append(current)
            current = []
    if current:
        # trailing partial
        trajectories.append(current)
    # filter out empty
    trajectories = [t for t in trajectories if len(t) > 0]
    return trajectories


def load_pkl_first_trajectory(pkl_path: str) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Load transitions from PKL file.

    Supports:
    - list[dict]: flat transitions; will be split into trajectories using dones/masks
    - list[list[dict]]: already split trajectories

    Returns:
        (transitions_of_first_trajectory, meta)
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pkl.load(f)

    # Handle different PKL formats
    if isinstance(data, list) and data and isinstance(data[0], list):
        # Multiple trajectories format: [[traj1_transitions], [traj2_transitions], ...]
        trajectories = data
        print(f"Loaded {len(trajectories)} trajectories from {pkl_path}")

        fmt = "nested"

    elif isinstance(data, list) and data and isinstance(data[0], dict):
        # Flat transitions: [transition1, transition2, ...] possibly from multiple episodes
        flat = data
        trajectories = _split_into_trajectories(flat)
        print(f"Loaded flat transitions: {len(flat)} steps, split into {len(trajectories)} trajectories from {pkl_path}")
        fmt = "flat"
    else:
        raise ValueError("Unexpected PKL format")

    if not trajectories:
        raise ValueError("No trajectories found in PKL.")

    num_trajectories = len(trajectories)
    selected_idx = 0
    print(f"Using first trajectory {selected_idx}/{num_trajectories-1}")

    transitions = trajectories[selected_idx]
    print(f"Selected trajectory has {len(transitions)} transitions")

    # Validate structure
    if transitions:
        sample = transitions[0]
        required_keys = ["observations", "actions", "rewards", "masks", "dones"]
        for key in required_keys:
            if key not in sample:
                raise ValueError(f"Transition missing required key: {key}")

        action_shape = sample["actions"].shape
        if action_shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action_shape}")

    meta = {"format": fmt, "num_trajectories": num_trajectories, "selected_trajectory": selected_idx}
    return transitions, meta


def replay_transitions(
    transitions: List[Dict],
    env: XArmEnv,
    delay: float = 0.5,
    max_steps: int = None,
    use_servo: bool = True
):
    """Replay transitions on the robot."""
    num_steps = min(len(transitions), max_steps) if max_steps else len(transitions)

    mode_name = "servo" if use_servo else "position"
    print(f"Starting replay of {num_steps} steps...")
    print(f"Delay: {delay}s, Control mode: {mode_name}")

    for i in range(num_steps):
        transition = transitions[i]
        action = transition["actions"]
        reward = transition["rewards"]
        done = transition["dones"]

        print(f"\nStep {i+1}/{num_steps}")
        print(f"Action: {action}")
        print(f"Reward: {reward}, Done: {done}")

        # Execute action
        try:
            obs, rew, terminated, truncated, info = env.step(action, use_servo=use_servo)
            print(f"Executed action - Reward: {rew}, Terminated: {terminated}, Truncated: {truncated}")

            # Show current pose
            if "state" in obs and "tcp_pose" in obs["state"]:
                tcp_pose = obs["state"]["tcp_pose"]
                print(f"Current TCP pose: [{tcp_pose[0]:.3f}, {tcp_pose[1]:.3f}, {tcp_pose[2]:.3f}, "
                      f"{tcp_pose[3]:.3f}, {tcp_pose[4]:.3f}, {tcp_pose[5]:.3f}, {tcp_pose[6]:.3f}]")

        except Exception as e:
            print(f"Error executing action: {e}")
            break

        time.sleep(delay)

        # Check for termination
        if done or terminated or truncated:
            print(f"Episode ended at step {i+1}")
            break

    print(f"\nReplay completed after {min(i+1, num_steps)} steps")


def replay_positions(
    transitions: List[Dict],
    env: XArmEnv,
    delay: float = 0.5,
    max_steps: int = None,
    use_servo: bool = False,
):
    """
    Replay absolute tcp_pose (+ gripper) from PKL.

    Assumes obs['observations']['state'] is 8D: tcp_pose(7) + gripper_pose(1).
    Drives the robot in position mode by default (use_servo=False).
    """
    from scipy.spatial.transform import Rotation as R

    num_steps = min(len(transitions), max_steps) if max_steps else len(transitions)
    mode_name = "servo" if use_servo else "position"
    print(f"Starting POS replay of {num_steps} steps...")
    print(f"Delay: {delay}s, Control mode: {mode_name}")

    # Need access to underlying robot handle
    robot = getattr(env, "_robot", None)
    if robot is None:
        raise RuntimeError("XArmEnv has no robot connection (_robot is None). Are you running with fake_env=True?")

    for i in range(num_steps):
        tr = transitions[i]
        state = tr["observations"]["state"]
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] < 8:
            raise ValueError(f"Expected 8D state (tcp_pose+gripper), got shape {state.shape}")

        tcp_pose = state[:7]
        gripper_pose = float(state[7])

        x, y, z, qx, qy, qz, qw = [float(v) for v in tcp_pose]
        rot = R.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = rot.as_euler("xyz", degrees=False)

        target_pose_euler = [x, y, z, roll, pitch, yaw]

        print(f"\nStep {i+1}/{num_steps}")
        print(f"Target TCP (xyz+quat): {tcp_pose}")
        print(f"Target gripper: {gripper_pose:.3f}")

        # Send pose command
        if use_servo:
            # Servo mode: same API used in env
            robot.robot.set_mode(1)
            robot.robot.set_state(0)
            robot.set_servo_cartesian(target_pose_euler, is_radian=True)
        else:
            # Position mode
            robot.robot.set_mode(0)
            robot.robot.set_state(0)
            robot.move_end_effector_to_goal_pose(target_pose_euler, is_radian=True, wait=False)

        # Send gripper command using env helper if available
        if hasattr(env, "_send_gripper_command"):
            env._send_gripper_command(gripper_pose)

        time.sleep(delay)

        done = bool(tr.get("dones", False))
        if done:
            print(f"Episode ended at step {i+1}")
            break

    print(f"\nPOS replay completed after {min(i+1, num_steps)} steps")


def main(_):
    if FLAGS.pkl_path is None:
        raise ValueError("--pkl_path must be specified")

    # Load transitions
    transitions, meta = load_pkl_first_trajectory(FLAGS.pkl_path)

    if not transitions:
        print("No transitions to replay")
        return

    # Create XArm environment
    config = XArmEnvConfig()
    env = XArmEnv(fake_env=False, save_video=False, config=config)

    try:
        # Reset to initial pose if requested
        if FLAGS.reset_at_start:
            print("Resetting robot to initial pose...")
            obs, info = env.reset()
            print("Reset complete")

            # Wait a bit for reset to complete
            time.sleep(2.0)

        if FLAGS.replay_mode == "delta":
            replay_transitions(
                transitions=transitions,
                env=env,
                delay=FLAGS.delay,
                max_steps=FLAGS.max_steps,
                use_servo=FLAGS.use_servo,
            )
        else:
            replay_positions(
                transitions=transitions,
                env=env,
                delay=FLAGS.delay,
                max_steps=FLAGS.max_steps,
                use_servo=FLAGS.use_servo,
            )

    except KeyboardInterrupt:
        print("\nReplay interrupted by user")
    except Exception as e:
        print(f"Error during replay: {e}")
    finally:
        print("Closing environment...")
        env.close()


if __name__ == "__main__":
    app.run(main)