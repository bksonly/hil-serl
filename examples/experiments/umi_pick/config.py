import os
import numpy as np
import jax
import jax.numpy as jnp

from serl_robot_infra.xarm_env import XArmEnv, XArmEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from experiments.config import DefaultTrainingConfig
from experiments.umi_pick.wrapper import UmiPickXArmEnv


class EnvConfig(XArmEnvConfig):
    """Task-specific overrides for umi_pick on XArm."""

    # Override IP / camera port as needed
    ROBOT_IP: str = "192.168.1.240"
    LOCAL_IP = None
    CAMERA_PORT: int = 0

    # Action scaling: tune if needed
    ACTION_SCALE = np.array([0.015, 0.1, 1.0], dtype=np.float32)

    # Reset pose: [x, y, z, roll, pitch, yaw] meters/radians
    # User-provided: [0.158, 0.28, 0.145, 180°, -90°, 0°]
    RESET_POSE = np.array([0.158, 0.28, 0.145, np.pi, -np.pi / 2.0, 0.0], dtype=np.float32)

    MAX_EPISODE_LENGTH: int = 150


class TrainConfig(DefaultTrainingConfig):
    # Single camera
    image_keys = ["image"]
    # Proprio keys consistent with XArmEnv observation dict
    proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose"]
    # Use learned gripper mode (continuous gripper)
    setup_mode = "single-arm-learned-gripper"
    encoder_type = "resnet-pretrained"
    checkpoint_period = 2000
    buffer_period = 1000
    random_steps = 0

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = UmiPickXArmEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        # No spacemouse / classifier for minimal setup
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        return env

