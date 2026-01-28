import os
import numpy as np
import jax
import jax.numpy as jnp

from serl_robot_infra.xarm_env import XArmEnv, XArmEnvConfig, UMIIntervention
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_robot_infra.xarm_env.envs.wrappers import MultiCameraBinaryRewardClassifierWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func
from experiments.config import DefaultTrainingConfig
from experiments.umi_pick.wrapper import UmiPickXArmEnv


class EnvConfig(XArmEnvConfig):
    """Task-specific overrides for umi_pick on XArm."""



class TrainConfig(DefaultTrainingConfig):
    # Single camera
    image_keys = ["image"]
    # Reward classifier使用的图像键
    classifier_keys = ["image"]
    # Proprio keys consistent with XArmEnv observation dict (removed tcp_vel as UMI data doesn't have it)
    proprio_keys = ["tcp_pose", "gripper_pose"]
    # Use fixed-gripper mode: gripper as continuous action (not discrete {-1,0,1})
    # Note: "fixed" here means "not using discrete grasp_critic", not "gripper is fixed"
    setup_mode = "single-arm-fixed-gripper"
    encoder_type = "resnet-pretrained"
    checkpoint_period = 100
    buffer_period = 400
    random_steps = 0
    
    # RL training parameters
    max_steps = 6000
    training_starts = 10000  # Minimum buffer size before starting training
    batch_size = 1024
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
        
        # 如果启用分类器，加载并包装环境
        if classifier and self.classifier_keys is not None:
            classifier_func = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # 使用分类器输出作为奖励，阈值可以根据任务调整
                # classifier_func(obs) 通常返回 shape (B, 1) 或 (B,) 的 logit，这里取第一个标量
                logit = jnp.asarray(classifier_func(obs)).reshape(-1)[0]
                prob = float(sigmoid(logit))
                # 返回 (reward, prob) 元组，便于显示
                return int(prob > 0.75), prob

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        
        return env
    
    def process_demos(self, demo):
        """
        Process demonstration data if needed.
        For UMI data that's already converted via convert_hdf5_to_pkl.py,
        and now in base coordinates with 8D state (tcp_pose + gripper, no tcp_vel),
        no additional processing is needed.
        """
        return demo

