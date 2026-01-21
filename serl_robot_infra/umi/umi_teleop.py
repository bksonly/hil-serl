#!/usr/bin/env python3
"""UMI 夹爪遥操作脚本：使用 delta pose 控制 xarm 机械臂。"""
import rospy
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os
import threading

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pynput import keyboard

from serl_robot_infra.umi.umi_expert import UMIExpert
from serl_robot_infra.xarm_env.BestMan_Xarm.RoboticsToolBox.Bestman_real_xarm6 import (
    Bestman_Real_Xarm6
)


class UMITeleop:
    """使用 UMI 夹爪的 delta pose 控制 xarm 机械臂。"""
    
    def __init__(self, robot_ip, pose_queue_size, 
                 max_pos_delta, max_rot_delta, frequency):
        """
        初始化遥操作系统。
        
        Args:
            robot_ip: xarm 机械臂的 IP 地址
            pose_queue_size: pose 队列大小
            max_pos_delta: 最大位置增量（米），用于安全限幅
            max_rot_delta: 最大旋转增量（弧度），用于安全限幅
            frequency: 控制频率（Hz）
        """
        # 初始化 ROS 节点
        rospy.init_node('umi_teleop', anonymous=True)
        
        # 初始化 UMI Expert
        self.umi_expert = UMIExpert(pose_queue_size=pose_queue_size)
        
        # 初始化 xarm（使用 Bestman 封装）
        self.robot_ip = robot_ip
        self.robot = None
        self._init_xarm()
        
        # 控制参数
        self.control_rate = frequency  # 控制频率
        self.shutdown_flag = False
        self.max_pos_delta = max_pos_delta
        self.max_rot_delta = max_rot_delta
        
        # 当前位姿（用于 delta 控制）
        self.current_pose = None
        
        # 遥操作启动判断（改用按键控制）
        self.teleop_started = False  # 遥操作是否已启动
        self.teleop_lock = threading.Lock()  # 保护 teleop_started 的锁
        
        # 按键监听（pynput）
        self.keyboard_listener = None
        self._setup_keyboard_listener()
        
        rospy.loginfo("UMI 遥操作系统初始化完成")
        rospy.loginfo("按 'i' 键开始遥操作，按 'o' 键退出遥操作")
    
    def _init_xarm(self):
        """初始化 xarm 机械臂并设置为伺服模式（使用 Bestman 封装）。"""
        rospy.loginfo(f"正在连接 xarm 机械臂: {self.robot_ip}")
        
        # 使用 Bestman 封装初始化，servo_mode=True
        self.robot = Bestman_Real_Xarm6(
            self.robot_ip,
            None,
            None,
            servo_mode=True
        )
        
        # 清除故障
        self.robot.clear_fault()
        
        # 获取当前位姿作为初始位置
        rospy.loginfo("获取当前机械臂位姿...")
        time.sleep(0.5)

        pose = self.robot.get_current_end_effector_pose()  # 返回 [x, y, z, roll, pitch, yaw] (米, 弧度)
        self.current_pose = np.array(pose)
        rospy.loginfo(f"当前机械臂位姿: {self.current_pose}")
        
        # 初始化夹爪（如果需要）
        self.robot.find_gripper_robotiq()
        time.sleep(0.5)
        rospy.loginfo("夹爪初始化完成")
        rospy.loginfo("xarm 机械臂初始化完成")
            
    
    def _get_current_pose(self):
        """获取当前机械臂位姿（使用 Bestman 封装）。"""
        # Bestman 的 get_current_end_effector_pose() 返回 [x, y, z, roll, pitch, yaw] (米, 弧度)
        pose = self.robot.get_current_end_effector_pose()
        return np.array(pose)
    
    def _apply_delta_to_pose(self, current_pose, delta):
        """
        将 delta 应用到当前位姿，并进行安全限幅。
        
        Args:
            current_pose: [x, y, z, roll, pitch, yaw] (米, 弧度)
            delta: 从 get_pose_delta() 返回的字典
        
        Returns:
            target_pose: [x, y, z, roll, pitch, yaw] (米, 弧度)
        """
        # 位置：直接相加，并进行限幅
        pos_delta = np.array([
            delta['position']['x'],
            delta['position']['y'],
            delta['position']['z']
        ])
        
        # 限幅位置增量
        pos_delta_norm = np.linalg.norm(pos_delta)
        if pos_delta_norm > self.max_pos_delta:
            pos_delta = pos_delta / pos_delta_norm * self.max_pos_delta
        
        target_pos = current_pose[:3] + pos_delta
        
        # 旋转：将当前欧拉角转换为旋转矩阵，应用 delta 旋转，再转回欧拉角
        R_current = R.from_euler('xyz', current_pose[3:], degrees=False)
        
        # delta 旋转向量转换为旋转矩阵，并进行限幅
        rotation_vec = np.array(delta['rotation_vec'])
        rotation_angle = np.linalg.norm(rotation_vec)
        
        # 限幅旋转角度
        if rotation_angle > self.max_rot_delta:
            rotation_vec = rotation_vec / rotation_angle * self.max_rot_delta
        
        R_delta = R.from_rotvec(rotation_vec)
        
        # 组合旋转：R_target = R_delta * R_current
        R_target = R_delta * R_current
        
        # 转回欧拉角
        target_euler = R_target.as_euler('xyz', degrees=False)
        
        return np.concatenate([target_pos, target_euler])
    
    def _setup_keyboard_listener(self):
        """设置按键监听器（使用 pynput，不需要 root）。"""
        def on_press(key):
            try:
                # 处理字符键
                if hasattr(key, 'char') and key.char:
                    if key.char == 'i':
                        with self.teleop_lock:
                            if not self.teleop_started:
                                self.teleop_started = True
                                rospy.loginfo("【遥操作已启动】按 'o' 键退出")
                    elif key.char == 'o':
                        with self.teleop_lock:
                            if self.teleop_started:
                                self.teleop_started = False
                                rospy.loginfo("【遥操作已退出】按 'i' 键开始")
            except AttributeError:
                # 特殊键（如 Ctrl、Alt 等）忽略
                pass
        
        # 启动非阻塞监听器
        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.daemon = True
        self.keyboard_listener.start()
        rospy.loginfo("按键监听器已启动（使用 pynput，不需要 root 权限）")
    
    def _control_gripper(self, clamp_value):
        """
        控制夹爪（使用 Bestman 封装）。
        
        Args:
            clamp_value: 夹爪宽度值（从 UMI 获取）
        """
        try:
            # 将 clamp 值转换为夹爪位置
            # clamp 值范围通常是 0-88mm，需要转换为 0-255
            # 参考 arm_sdk_adapter.py: closest_width = int(((88 - width) / 88 * 255))
            closest_width = int(((88 - clamp_value) / 88 * 255))
            closest_width = np.clip(closest_width, 0, 255)
            
            # 使用 Bestman 封装的接口
            self.robot.gripper_goto_robotiq(
                pos=closest_width,
                speed=0xFF,
                force=0xFF,
                wait=False
            )
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"夹爪控制失败: {e}")
    
    def _control_loop(self):
        """控制循环：100 Hz。"""
        rospy.loginfo("控制循环启动 (100 Hz)")
        rate = rospy.Rate(self.control_rate)
        
        while not self.shutdown_flag and not rospy.is_shutdown():
            try:
                # 获取当前机械臂位姿
                current_pose = self._get_current_pose()
                if current_pose is None:
                    rate.sleep()
                    continue
                
                # 更新当前位姿
                self.current_pose = current_pose
                delta = self.umi_expert.get_pose_delta()
                
                # 获取当前夹爪值
                clamp_value = self.umi_expert.get_current_clamp()
                
                # 检查遥操作启动状态（由按键监听器异步更新）
                with self.teleop_lock:
                    teleop_active = self.teleop_started
                
                # 只有在遥操作已启动时才执行控制
                if not teleop_active:
                    # 保持不动，只打印提示信息
                    rospy.loginfo_throttle(2.0, "等待遥操作启动：按 'i' 键开始")
                    rate.sleep()
                    continue
                
                # 如果 delta 和 clamp 都有效，执行控制
                if delta is not None and clamp_value is not None:
                    # 计算目标位姿
                    target_pose = self._apply_delta_to_pose(current_pose, delta)
                    
                    # 发送位姿控制命令
                    # Bestman 的 set_servo_cartesian 接受 [x, y, z, roll, pitch, yaw] (米, 弧度)
                    # 参考代码：每次发送命令前先 set_state(0) 确保状态正确
                    try:
                        self.robot.robot.set_state(0)
                        self.robot.set_servo_cartesian(
                            target_pose.tolist(),
                            is_radian=True,
                            speed=100,
                            mvacc=2000
                        )
                    except Exception as e:
                        rospy.logwarn_throttle(1.0, f"位姿控制失败: {e}")
                    
                    # 控制夹爪
                    self._control_gripper(clamp_value)
                
            except Exception as e:
                rospy.logwarn_throttle(1.0, f"控制循环错误: {e}")
            
            rate.sleep()
        
        rospy.loginfo("控制循环退出")
    
    def start(self):
        """启动遥操作系统。"""
        rospy.loginfo("启动 UMI 遥操作系统...")
        self.shutdown_flag = False
        
        # 启动 UMI Expert
        self.umi_expert.start()
        
        # 等待一下让数据开始接收
        rospy.sleep(0.5)
        
        # 启动控制循环
        self._control_loop()
    
    def stop(self):
        """停止遥操作系统。"""
        rospy.loginfo("停止 UMI 遥操作系统...")
        self.shutdown_flag = True
        
        # 停止按键监听器
        self.keyboard_listener.stop()
        self.keyboard_listener = None
        
        # 停止 UMI Expert
        self.umi_expert.stop()
        
        # 将机械臂切换回位置模式
        self.robot.set_mode(0)
        self.robot.clear_fault()
        rospy.loginfo("机械臂已切换回位置模式")


def main():
    """主函数。"""
    # ========== 配置参数（直接在这里修改） ==========
    ROBOT_IP = "192.168.1.224"  # xarm 机械臂 IP 地址
    FREQUENCY = 100  # 控制频率（Hz）
    POSE_QUEUE_SIZE = int(500/FREQUENCY + 1) # pose 队列大小
    MAX_POS_DELTA = 0.003  # 最大位置增量（米），用于安全限幅
    MAX_ROT_DELTA = 0.02  # 最大旋转增量（弧度），用于安全限幅
    
    # ============================================
    
    try:
        teleop = UMITeleop(
            robot_ip=ROBOT_IP,
            pose_queue_size=POSE_QUEUE_SIZE,
            max_pos_delta=MAX_POS_DELTA,
            max_rot_delta=MAX_ROT_DELTA,
            frequency=FREQUENCY
        )
        
        teleop.start()
        
    except KeyboardInterrupt:
        rospy.loginfo("\n收到中断信号...")
    except Exception as e:
        rospy.logerr(f"运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'teleop' in locals():
            teleop.stop()


if __name__ == "__main__":
    main()
