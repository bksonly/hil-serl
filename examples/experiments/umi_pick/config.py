import os
import numpy as np
import jax
import jax.numpy as jnp

from serl_robot_infra.xarm_env import XArmEnv, XArmEnvConfig, UMIIntervention
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from experiments.config import DefaultTrainingConfig
from experiments.umi_pick.wrapper import UmiPickXArmEnv


class EnvConfig(XArmEnvConfig):
    """Task-specific overrides for umi_pick on XArm."""



class TrainConfig(DefaultTrainingConfig):
    # Single camera
    image_keys = ["image"]
    # Proprio keys consistent with XArmEnv observation dict
    proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose"]
    # Use fixed-gripper mode: gripper as continuous action (not discrete {-1,0,1})
    # Note: "fixed" here means "not using discrete grasp_critic", not "gripper is fixed"
    setup_mode = "single-arm-fixed-gripper"
    encoder_type = "resnet-pretrained"
    checkpoint_period = 10
    buffer_period = 10000
    random_steps = 0
    
    # RL training parameters
    max_steps = 2000
    training_starts = 10000  # Minimum buffer size before starting training
    batch_size = 256
    cta_ratio = 2  # Critic-to-actor update ratio
    discount = 0.97
    steps_per_update = 50
    log_period = 10
    replay_buffer_capacity = 200000

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = UmiPickXArmEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        # 包上 UMI 干预 wrapper：按 i 进入遥操作，按 o 退出（统一使用 100Hz 伺服模式）
        if not fake_env:
            env = UMIIntervention(env, config_path=None)

        # SERL 观测与 chunking 包装
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        return env
    
    def process_demos(self, demo):
        """
        Process demonstration data if needed.
        For UMI data that's already converted via convert_hdf5_to_pkl.py,
        no additional processing is needed.
        """
        return demo

