import gymnasium as gym
import numpy as np
import time
from typing import Optional
import threading
from pynput import keyboard

from serl_robot_infra.xarm_env.umi.umi_expert import UMIExpert
import gymnasium as gym


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    Use a classifier function on observations to produce binary reward.
    """

    def __init__(self, env: gym.Env, reward_classifier_func, target_hz: Optional[float] = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or bool(rew)
        info["succeed"] = bool(rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        return obs, info


class UMIIntervention(gym.ActionWrapper):
    """
    在 XArmEnv 上实现 UMI 遥操作接管的 wrapper。

    - 默认使用 policy 输出的 action。
    - 当按下键盘 'i' 时，进入“遥操作接管”模式：
        * 每一步从 UMIExpert 读 delta pose，转换成 7 维归一化 action（前 6 维末端增量，第 7 维夹爪暂时用 0）。
        * 如果 UMI 增量非零，则替换掉 policy action，并在 info["intervene_action"] 中记录。
    - 当按下键盘 'o' 时，退出接管模式，回到纯 policy 控制。
    """

    def __init__(self, env, config_path=None, action_indices=None):
        super().__init__(env)

        self.debug_mode = False
        
        # UMI Expert：只负责提供 delta pose / clamp
        # 确保 ROS node 已初始化（UMIExpert 需要 ROS 环境）
        try:
            import rospy
            if rospy.get_node_uri() is None:
                # ROS node 未初始化，尝试初始化
                rospy.init_node('xarm_umi_intervention', anonymous=True)
        except Exception as e:
            pass
        
        self.umi_expert = UMIExpert(
            pose_queue_size= int(500/env.config.SERVO_FREQUENCY + 1)
        )

        self.umi_expert.start()
        
        # 等待一下让 subscriber 建立连接
        import time
        time.sleep(0.5)

        # 接管状态：False = policy 控制；True = UMI 接管
        self.teleop_enabled = False
        self.teleop_lock = threading.Lock()  # 保护 teleop_enabled 的锁

        # 按键监听（pynput）
        self.keyboard_listener = None
        self._setup_keyboard_listener()
        # 初始化时打印一次状态
        print(f"\033[96m[UMIIntervention] 系统已启动，按 'i' 键开始遥操作，按 'o' 键退出遥操作\033[0m")
    # ------------------ 键盘状态 ------------------

    def _setup_keyboard_listener(self):
        """设置按键监听器（使用 pynput，不需要 root）。"""
        def on_press(key):
            try:
                # 处理字符键
                if hasattr(key, 'char') and key.char:
                    if key.char == 'i':
                        with self.teleop_lock:
                            if not self.teleop_enabled:
                                self.teleop_enabled = True
                    elif key.char == 'o':
                        with self.teleop_lock:
                            if self.teleop_enabled:
                                self.teleop_enabled = False
            except AttributeError:
                # 特殊键（如 Ctrl、Alt 等）忽略
                pass
        
        # 启动非阻塞监听器
        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.daemon = True
        self.keyboard_listener.start()


    # ------------------ 生成 UMI 动作 ------------------

    def _get_umi_action(self):
        """
        从 UMIExpert 取 delta pose 和 clamp 宽度，转换成与 XArmEnv 一致的 7D 归一化动作。

        返回:
            expert_a: np.ndarray(7,)
        """
        delta = self.umi_expert.get_pose_delta()
        if delta is None:
            return np.zeros(7, dtype=np.float32)

        # 位置增量（米）
        pos_delta = np.array([
            delta["position"]["x"],
            delta["position"]["y"],
            delta["position"]["z"],
        ], dtype=np.float32)

        # 旋转增量（旋转向量，弧度）
        rotvec = np.array(delta["rotation_vec"], dtype=np.float32)

        xyz_scale = float(self.env.action_scale[0])
        rot_scale = float(self.env.action_scale[1])
        gripper_scale = float(self.env.action_scale[2])

        a_xyz = pos_delta / max(xyz_scale, 1e-6)
        a_rot = rotvec / max(rot_scale, 1e-6)

        # clip 到 [-1, 1]
        a_xyz = np.clip(a_xyz, -1.0, 1.0)
        a_rot = np.clip(a_rot, -1.0, 1.0)

        clamp_raw = self.umi_expert.get_current_clamp()
        if clamp_raw is None:
            clamp_umi = None
        else:
            clamp_umi = clamp_raw / 88.0 * 2.0 - 1.0

        current_gripper = float(self.env._get_gripper_pose()[0]) # [-1, 1]

        a = np.zeros(7, dtype=np.float32)
        a[:3] = a_xyz
        a[3:6] = a_rot

        if clamp_umi is None:
            # 没有 clamp 数据，本步不动夹爪
            gripper_delta = 0.0
            a[6] = 0.0
        else:
            gripper_delta = (clamp_umi - current_gripper) / gripper_scale
            a[6] = np.clip(gripper_delta, -1.0, 1.0)

        return a

    # ------------------ ActionWrapper 接口 ------------------

    def action(self, action: np.ndarray):
        """
        输入:
            action: policy action (7 维)
        输出:
            new_action: 可能被 UMI 接管替换后的 action
        """
        # 检查遥操作启动状态（由按键监听器异步更新）
        with self.teleop_lock:
            teleop_active = self.teleop_enabled

        # 如果未开启 teleop
        if not teleop_active:
            self._last_used_expert = False
            if self.debug_mode:
                return np.zeros_like(action)
            else:
                # 非调试模式：直接返回 policy 的 action
                return action
        else:
            # 遥操作模式：使用 UMI 动作
            expert_a = self._get_umi_action()
            # 记录这一步使用的人类动作，用于 demo buffer
            self._last_used_expert = True
            self._last_intervene_action = expert_a.copy()
            return expert_a

    def step(self, action):
        with self.teleop_lock:
            teleop_active = self.teleop_enabled
        
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(action=new_action, use_servo=teleop_active or self.debug_mode)

        # 如果这一轮实际使用了 UMI 动作，则在 info 中写入 intervene_action
        # 这样 train_rlpd.py 中就会用人类动作覆盖 policy 动作，并按“delta action”写入 buffer
        if getattr(self, "_last_used_expert", False):
            info["intervene_action"] = self._last_intervene_action.copy()

        return obs, rew, done, truncated, info

    def close(self):
        # 停止按键监听器
        if self.keyboard_listener is not None:
            try:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
            except Exception:
                pass
        
        # 关掉底层 env 的同时，把 UMIExpert 也停掉
        try:
            self.umi_expert.stop()
        except Exception:
            pass
        return super().close()