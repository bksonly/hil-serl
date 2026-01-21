"""Minimal Gym interface for XArm using Bestman_Real_Xarm6.

This is a *minimal* implementation to integrate XArm into the SERL pipeline.
It mirrors the key interfaces of FrankaEnv:
- observation_space: Dict with "state" and "images"
- action_space: normalized 7D action in [-1, 1]

State (for each step):
- "tcp_pose": (7,)  [x, y, z, qx, qy, qz, qw]  (meters, unit quaternion)
- "gripper_pose": (1,) normalized in [-1, 1]

Action:
- 7D Box(-1, 1):
  - a[:3]: xyz delta (normalized), scaled by ACTION_SCALE[0] to meters
  - a[3:6]: rotvec delta (normalized), scaled by ACTION_SCALE[1] to radians
  - a[6]: gripper delta (normalized), scaled by ACTION_SCALE[2]

For now, we:
- Use Bestman_Real_Xarm6 to connect to the real robot
- Use a single OpenCV camera (e.g., USB or Realsense RGB) as "image"
- Provide a fake_env mode that avoids hardware and returns dummy data
"""

import time
from typing import Dict, Tuple, Optional
import logging

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

# Configure logger for camera timeout warnings
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '\033[91m[CAMERA TIMEOUT]\033[0m %(message)s'  # Red color for timeout warnings
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

from serl_robot_infra.xarm_env.BestMan_Xarm.RoboticsToolBox.Bestman_real_xarm6 import (
    Bestman_Real_Xarm6,
)


class XArmEnvConfig:
    """Configuration for XArmEnv (kept simple for now)."""

    # Robot & gripper
    ROBOT_IP: str = "192.168.1.224"
    POS_FREQUENCY: int = 3  # Hz
    SERVO_FREQUENCY: int = 50  # Hz
    # Camera: simple OpenCV VideoCapture port
    CAMERA_PORT: int = 1

    # Action scaling: [xyz_scale_m, rot_scale_rad, gripper_scale]
    ACTION_SCALE: np.ndarray = np.array([0.015, 0.1, 1.0], dtype=np.float32)

    # Episode length
    MAX_EPISODE_LENGTH: int = 10000

    # Reset pose in task space: [x, y, z, roll, pitch, yaw] (meters, radians)
    RESET_POSE: np.ndarray = np.array([0.158, 0.28, 0.145, np.pi, -np.pi / 2.0, 0.0], dtype=np.float32)
    # RESET_POSE: np.ndarray = np.array([0.4, -0.004, 0.1398, np.pi, -np.pi / 2.0, 0.0], dtype=np.float32) # user-provided pick reset: [0.158, 0.28, 0.145, 180°, -90°, 0°]


class XArmEnv(gym.Env):
    """Minimal XArm environment compatible with SERL."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        fake_env: bool = False,
        save_video: bool = False,
        config: Optional[XArmEnvConfig] = None,
    ):
        super().__init__()
        self.config = config or XArmEnvConfig()
        self.fake_env = fake_env # learner使用fake
        self.save_video = save_video

        self.action_scale = self.config.ACTION_SCALE.astype(np.float32)
        self.max_episode_length = self.config.MAX_EPISODE_LENGTH
        self.episode_steps = 0

        # Define action/observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,), dtype=np.float32
                        ),
                        "gripper_pose": gym.spaces.Box(
                            -1.0, 1.0, shape=(1,), dtype=np.float32
                        ),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        # Single RGB camera resized to 128x128
                        "image": gym.spaces.Box(
                            low=0,
                            high=255,
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                        )
                    }
                ),
            }
        )

        # Internal handles
        self._robot: Optional[Bestman_Real_Xarm6] = None
        self._camera: Optional[cv2.VideoCapture] = None

        # Last observation cache
        self._last_state: Optional[Dict] = None

        if not self.fake_env:
            self._init_robot_connection()
            self._init_camera()
            self._reset_robot()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.episode_steps = 0

        if not self.fake_env:
            self._reset_robot()

        obs = self._get_obs()
        info: Dict = {"succeed": False}
        return obs, info

    def step(self, action: np.ndarray, use_servo) -> Tuple[Dict, float, bool, bool, Dict]:
        """Standard gym step with normalized action."""
        start_time = time.time()
        if use_servo:
            self.dt = 1.0 / self.config.SERVO_FREQUENCY
        else:
            self.dt = 1.0 / self.config.POS_FREQUENCY
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if not self.fake_env:
            self._execute_action(action, use_servo)
            # simple rate control
            elapsed = time.time() - start_time
            sleep_time = max(0.0, self.dt - elapsed)
            time.sleep(sleep_time)
            total_elapsed = (time.time() - start_time) * 1000
            print(f"耗时: {elapsed*1000:.2f}ms, 总时长: {total_elapsed:.2f}ms")

        obs = self._get_obs()
        self.episode_steps += 1

        # Minimal reward / termination: no task reward yet
        reward = 0.0
        terminated = False  # no terminal condition from reward yet
        truncated = self.episode_steps >= self.max_episode_length
        info: Dict = {"succeed": False}

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass
            self._camera = None

        # We deliberately do not power off the robot here.
        self._robot = None

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _init_robot_connection(self):
        """Initialize connection to XArm via Bestman_Real_Xarm6."""
        self._robot = Bestman_Real_Xarm6(
            self.config.ROBOT_IP, 
            None,  
            None,
            servo_mode=False  # 初始化时用 position mode，后续由 step 的 use_servo 参数控制
        )
        # Basic clear fault / go to default mode
        self._robot.clear_fault()
        
        # Initialize gripper (always needed)
        try:
            self._robot.robot.set_tgpio_modbus_baudrate(115200)
            self._robot.robot.robotiq_reset()
            self._robot.robot.robotiq_set_activate(True)
            time.sleep(1.0)
        except Exception as e:
            logger.warning(f"Failed to initialize gripper: {e}")

    def _init_camera(self):
        """Initialize a single OpenCV camera."""
        self._camera = cv2.VideoCapture(self.config.CAMERA_PORT)
        # Try to set a reasonable resolution; we will resize to 128x128 anyway.
        self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
        # Warm-up frames
        for _ in range(5):
            self._camera.grab()

    def _reset_robot(self):
        """Reset robot to initial pose."""
        if self.fake_env or self._robot is None:
            return

        # Reset always uses position mode (more reliable for moving to fixed pose)
        # Switch to position mode for reset
        self._robot.robot.set_mode(0)
        self._robot.robot.set_state(0)
        time.sleep(0.1)

        # Use Bestman API to move to RESET_POSE (x, y, z, roll, pitch, yaw)
        reset_pose = self.config.RESET_POSE.astype(float).tolist()
        self._robot.move_end_effector_to_goal_pose(reset_pose, is_radian=True, wait=True)
        
        # Open gripper as default
        self._robot.gripper_goto_robotiq(0, wait_motion=True)
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # Observation & action implementation
    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict:
        if self.fake_env:
            # Simple fake obs: zeros + black image
            state_obs = {
                "tcp_pose": np.zeros(7, dtype=np.float32),
                "gripper_pose": np.zeros(1, dtype=np.float32),
            }
            image = np.zeros((128, 128, 3), dtype=np.uint8)
            images = {"image": image}
            obs = {"state": state_obs, "images": images}
            self._last_state = obs
            return obs

        tcp_pose = self._get_tcp_pose()
        gripper_pose = self._get_gripper_pose()
        images = self._get_images()

        state_obs = {
            "tcp_pose": tcp_pose.astype(np.float32),
            "gripper_pose": gripper_pose.astype(np.float32),
        }
        obs = {"state": state_obs, "images": images}
        self._last_state = obs
        return obs


    def _execute_action(self, action: np.ndarray, use_servo: bool):
        """Convert normalized action into real robot command."""
        if self._robot is None:
            return

        # 1) De-normalize
        # 检查 action 是否包含无效值
        if not np.isfinite(action).all():
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 检查 action_scale 是否有效
        if not np.isfinite(self.action_scale).all():
            raise ValueError(f"Invalid action_scale: {self.action_scale}")
        
        xyz_delta = action[:3] * self.action_scale[0]
        rotvec_delta = action[3:6] * self.action_scale[1]
        gripper_delta = action[6] * self.action_scale[2]
        
        # 检查计算结果是否有效
        if not np.isfinite(xyz_delta).all():
            xyz_delta = np.nan_to_num(xyz_delta, nan=0.0, posinf=0.0, neginf=0.0)
        
        if not np.isfinite(rotvec_delta).all():
            rotvec_delta = np.nan_to_num(rotvec_delta, nan=0.0, posinf=0.0, neginf=0.0)
        
        if not np.isfinite(gripper_delta):
            gripper_delta = np.nan_to_num(gripper_delta, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) Get current pose & gripper
        tcp_pose = self._get_tcp_pose()  # (7,) [x,y,z,qx,qy,qz,qw]
        gripper_pose = self._get_gripper_pose()[0]  # scalar in [-1,1]

        # 3) Position update
        target_pos = tcp_pose[:3] + xyz_delta

        # 4) Orientation update via rotvec
        curr_quat = tcp_pose[3:].copy()
        quat_norm = np.linalg.norm(curr_quat)
        
        if quat_norm < 1e-6:
            raise ValueError(f"Invalid quaternion with zero norm: {curr_quat}")
        
        # 归一化四元数（确保精度）
        if abs(quat_norm - 1.0) > 1e-6:
            curr_quat = curr_quat / quat_norm
        
        # 检查是否有 NaN 或 Inf
        if not np.isfinite(curr_quat).all():
            raise ValueError(f"Invalid quaternion with NaN/Inf: {curr_quat}")
        
        curr_rot = Rotation.from_quat(curr_quat)
        delta_rot = Rotation.from_rotvec(rotvec_delta)
        next_rot = delta_rot * curr_rot
        next_quat = next_rot.as_quat()

        # 5) Gripper update (keep in [-1,1])
        next_gripper = float(np.clip(gripper_pose + gripper_delta, -1.0, 1.0))

        # 6) Convert quaternion to euler (roll, pitch, yaw) in degrees
        rpy_rad = next_rot.as_euler("xyz", degrees=False)
        rpy_deg = np.degrees(rpy_rad)

        # 7) Send commands (still use radians for robot command)
        target_pose_euler = [
            float(target_pos[0]),
            float(target_pos[1]),
            float(target_pos[2]),
            float(rpy_rad[0]),
            float(rpy_rad[1]),
            float(rpy_rad[2]),
        ]
        self._send_pos_command(target_pose_euler, use_servo)
        self._send_gripper_command(next_gripper)

    # ------------------------------------------------------------------
    # Low-level robot & camera helpers
    # ------------------------------------------------------------------
    def _send_pos_command(self, pose_euler: np.ndarray, use_servo: bool):
        """Send end-effector pose command [x,y,z,roll,pitch,yaw] to robot."""
        if self._robot is None:
            return
        
        if use_servo:
            self._robot.robot.set_mode(1)  # Servo mode
            self._robot.robot.set_state(0)
            self._robot.set_servo_cartesian(pose_euler, is_radian=True)
        else:
            self._robot.robot.set_mode(0)  # Position mode
            self._robot.robot.set_state(0)
            self._robot.move_end_effector_to_goal_pose(
                pose_euler, is_radian=True, wait=False
            )

    def _send_gripper_command(self, gripper_pos: float):
        """Send normalized gripper command in [-1,1] using robotiq."""
        if self._robot is None:
            return
        val = (gripper_pos + 1.0) / 2.0          # [0,1]，1=open, 0=closed (norm 语义)
        val = (1.0 - val) * 255.0                # [0,255]，0=open, 255=closed（Robotiq 语义）
        val = int(np.clip(val, 0, 255))
        try:
            self._robot.gripper_goto_robotiq(val, wait_motion=False)
        except Exception:
            pass

    def _get_tcp_pose(self) -> np.ndarray:
        """Get TCP pose as [x,y,z,qx,qy,qz,qw]."""
        if self._robot is None:
             return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Bestman returns [x,y,z,roll,pitch,yaw] in meters, radians
        pose = self._robot.get_current_end_effector_pose()
        x, y, z, roll, pitch, yaw = pose
        
        # 检查是否有无效值
        if not np.isfinite([x, y, z, roll, pitch, yaw]).all():
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        
        r = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False)
        qx, qy, qz, qw = r.as_quat()
        
        quat_norm = np.linalg.norm([qx, qy, qz, qw])
        if quat_norm < 1e-6:
            return np.array([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        
        return np.array([x, y, z, qx, qy, qz, qw], dtype=np.float32)

    def _get_gripper_pose(self) -> np.ndarray:
        """Get normalized gripper pose in [-1,1]."""
        if self._robot is None:
            return np.zeros(1, dtype=np.float32)

        try:
            pos = self._robot.get_gripper_position_robotiq()
            # Map raw [0,255] -> [0,1] open_width, then to [-1,1]
            open_width = 1.0 - float(pos) / 255.0
            norm = float(np.clip(open_width * 2.0 - 1.0, -1.0, 1.0))
            return np.array([norm], dtype=np.float32)
        except Exception:
            return np.zeros(1, dtype=np.float32)

    def _get_images(self) -> Dict[str, np.ndarray]:
        # image = np.zeros((128, 128, 3), dtype=np.uint8)
        # return {"image": image}
       
        """Get a single RGB image, resized to 128x128."""
        if self._camera is None:
            image = np.zeros((128, 128, 3), dtype=np.uint8)
            return {"image": image}

        # Grab & retrieve latest frame
        for _ in range(2):
            self._camera.grab()
        ret, frame = self._camera.retrieve()
        if not ret or frame is None:
            image = np.zeros((128, 128, 3), dtype=np.uint8)
            return {"image": image}

        # frame is typically BGR; convert to RGB and resize
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
        return {"image": rgb.astype(np.uint8)}


