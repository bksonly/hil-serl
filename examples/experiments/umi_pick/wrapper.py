from serl_robot_infra.xarm_env.xarm_env import XArmEnv


class UmiPickXArmEnv(XArmEnv):
    """Thin wrapper over XArmEnv for the umi_pick task."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

