#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastUMI 数据采集器 - 直接MP4录制版
修改说明：
1. RGB数据流不再保存RAW，而是直接通过OpenCV写入MP4。
2. 保持了原有的多进程架构和传感器同步逻辑。
"""

import rospy
import sys
import os
import signal
import argparse
import cv2
import csv
import threading
import time
import numpy as np
import json
import subprocess
import shutil
from collections import deque
from queue import Queue, Empty
from xv_sdk.msg import PoseStampedConfidence
from datetime import datetime
import enum
import psutil
from roslib.message import get_message_class
from scipy.spatial.transform import Rotation, Slerp
from pose_merge import transform_vive_to_gripper, transform_slam_to_gripper
import multiprocessing
from multiprocessing import Process, Event, Barrier, Value


try:
    cv2.setNumThreads(1)
except Exception:
    pass


# ============================================================================
# 相对变换计算辅助函数
# ============================================================================

def quaternion_to_matrix(q):
    """四元数转旋转矩阵 q: [qx, qy, qz, qw]"""
    return Rotation.from_quat(q).as_matrix()


def matrix_to_transform(R_mat, t):
    """构造4x4变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


def compute_relative_transform_matrices(pos_left, quat_left, pos_right, quat_right):
    """
    计算左设备到右设备的相对变换
    T_left_to_right = inv(T_left) @ T_right
    
    返回: T_rel (4, 4) 相对变换矩阵
    """
    R_left = quaternion_to_matrix(quat_left)
    R_right = quaternion_to_matrix(quat_right)
    T_left = matrix_to_transform(R_left, pos_left)
    T_right = matrix_to_transform(R_right, pos_right)
    T_rel = np.linalg.inv(T_left) @ T_right
    return T_rel


class RecordingState(enum.Enum):
    """录制状态枚举"""
    IDLE = "IDLE"           # 空闲状态，等待开始
    RECORDING = "RECORDING" # 录制中
    SAVING = "SAVING"       # 保存中
    FINISHED = "FINISHED"   # 完成


class SingleDeviceRecorder:
    """
    单设备数据采集器
    
    核心功能：
    - RGB图像采集（60Hz目标，直接写入MP4）
    - SLAM位姿采集（500Hz）
    - ToF点云采集（30Hz，可选）
    - Clamp夹爪数据采集（200Hz）
    - 多线程/进程并行写入，避免I/O阻塞
    """

    def __init__(self, device_config, output_dir=None, enable_tof=True, auto_start=False,
                 compute_relative_transform=False, peer_output_dir=None, fixed_relative_pos=None, max_rgb_count=None):
        """
        参数：
            device_config: dict, 设备配置 {'xv_serial': str, 'vive_serial': str, 'label': str}
            output_dir: str, 输出目录
            enable_tof: bool, 是否启用ToF
            auto_start: bool, 是否在初始化时自动等待用户输入
            compute_relative_transform: bool, 是否计算相对变换（双设备模式）
            peer_output_dir: str, 对方设备输出目录（用于读取对方数据）
            fixed_relative_pos: list[3], 无Vive时的固定相对位置 [x, y, z]（左相对右）
            max_rgb_count: int, 最大RGB帧数，达到后自动停止（None表示不限制）
        """
        # 解析设备配置
        self.xv_serial = device_config['xv_serial']
        self.vive_serial = device_config.get('vive_serial', '')
        self.device_label = device_config.get('label', '')
        
        # 判断是否启用Vive（如果vive_serial为空、None、'UNKNOWN'或不存在，则禁用）
        self.enable_vive = bool(self.vive_serial and 
                               self.vive_serial.strip() and 
                               self.vive_serial.upper() != 'UNKNOWN')

        self.device_serial = self.xv_serial

        self.running = True
        # bridge 将在初始化 ROS 后创建
        self.bridge = None
        self.enable_tof = enable_tof
        self.auto_start = auto_start
        self.max_rgb_count = max_rgb_count
        self.global_start_time = None
        
        # 相对变换计算参数
        self.compute_relative_transform = compute_relative_transform
        self.peer_output_dir = peer_output_dir
        self.fixed_relative_pos = fixed_relative_pos  # [x, y, z] 左相对右的固定位置

        # 输出目录 - 单臂模式：session_XXX/（直接放数据文件，不需要序列号子目录）
        if output_dir is None:
            # 自动生成：DATA/session_TIMESTAMP/（单设备模式下数据文件直接放在session目录下）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join("DATA", f"session_{timestamp}")
        else:
            self.output_dir = output_dir
        # 延迟创建目录：只有在验证通过后才创建，避免验证失败时留下空目录

        # 状态
        self.recording_state = RecordingState.IDLE
        self.recording_start_time = None
        self.recording_end_time = None

        # Topics
        self.rgb_topic = f"/xv_sdk/{self.xv_serial}/color_camera/image"
        self.slam_topic = f"/xv_sdk/{self.xv_serial}/slam/pose"
        if self.enable_vive:
            vive_serial_safe = self.vive_serial.replace('-', '_')
            self.vive_topic = f"/vive/{vive_serial_safe}/pose"  # 应用坐标系（已归零）
            self.vive_world_topic = f"/vive/{vive_serial_safe}/pose_world"  # 世界坐标系
        else:
            self.vive_topic = None
            self.vive_world_topic = None
        self.tof_topic = f"/xv_sdk/{self.xv_serial}/color_camera/point_cloud"
        self.clamp_topic = f"/xv_sdk/{self.xv_serial}/clamp/Data"

        # 计数与统计
        self.rgb_count = 0
        self.slam_count = 0
        self.vive_count = 0
        self.tof_count = 0
        self.tof_dropped_count = 0
        self.clamp_count = 0
        self.merged_count = 0

        # RGB 写入
        self.frame_width = None
        self.frame_height = None
        self.first_rgb_frame = False
        self.timestamp_writer = None
        self.timestamp_file = None

        # 计算队列大小
        if self.max_rgb_count is not None:
            queue_size = int(self.max_rgb_count * 0.5)
        else:
            queue_size = 800
        
        # [修改] RGB 队列现在存储 cv2 图片对象，而不是 raw bytes
        self.rgb_queue = Queue(maxsize=queue_size)
        self.rgb_writer_thread = None
        self._stop_writers = threading.Event()
        self.should_stop_recording = threading.Event()  # 标志：是否应该停止录制
        self.rgb_frame_index = 0
        self.video_writer = None 
        
        # 其他数据类型的异步写入队列和线程
        self.slam_queue = Queue(maxsize=queue_size)
        self.slam_writer_thread = None
        self.vive_queue = Queue(maxsize=queue_size) if self.enable_vive else None
        self.vive_writer_thread = None
        self.vive_world_queue = Queue(maxsize=queue_size) if self.enable_vive else None  # 世界坐标系队列
        self.vive_world_writer_thread = None
        self.tof_queue = Queue(maxsize=queue_size) if self.enable_tof else None
        self.tof_writer_thread = None
        self.clamp_queue = Queue(maxsize=queue_size)
        self.clamp_writer_thread = None
        
        # 队列长度监控
        self.queue_log_file = None
        self.queue_log_writer = None
        self.queue_monitor_thread = None
        self.queue_log_interval = 0.1

        # RGB FPS 估计
        self.rgb_timestamps = deque(maxlen=300)
        self.rgb_current_fps = 0.0
        self.rgb_avg_fps = 0.0
        self.rgb_fps_history = deque(maxlen=300)
        self.rgb_last_fps_calc_time = time.time()
        self.fps_calc_interval = 1
        
        # 进度条显示
        self.last_progress_update_time = time.time()
        self.progress_update_interval = 0.1
        self._auto_stop_triggered = False
        self._progress_alerted = set()  # 记录已经响过提示音的百分比
        
        # 检查 sox (play 命令) 是否可用
        self._sox_available = self._check_sox_available()

        # SLAM/VIVE
        self.slam_buffer = deque(maxlen=30000)  # 500Hz ≈ 6s
        if self.enable_vive:
            self.vive_buffer = deque(maxlen=10000)  # 100Hz ≈ 10s
        else:
            self.vive_buffer = None

        self.slam_file_handle = None
        self.vive_file_handle = None
        self.vive_world_file_handle = None  # 世界坐标系文件句柄
        self.clamp_file_handle = None
        self.tof_timestamp_writer = None
        self.tof_timestamp_file = None

        # Vive 对齐状态
        self.slam_first_timestamp = None
        if self.enable_vive:
            self.vive_first_timestamp = None
            self.vive_time_offset = None
            self.offset_ready = False
            self.vive_pending_buffer = []

        # 速度估计
        self.slam_prev_pose = None
        self.slam_prev_timestamp = None
        self.slam_linear_velocity = 0.0
        self.slam_angular_velocity = 0.0
        self.slam_velocity_history = deque(maxlen=10)

        if self.enable_vive:
            self.vive_prev_pose = None
            self.vive_prev_timestamp = None
            self.vive_linear_velocity = 0.0
            self.vive_angular_velocity = 0.0
            self.vive_velocity_history = deque(maxlen=10)

        # 融合配置/统计
        self.velocity_threshold = 1.0
        self.time_match_threshold = 0.01
        self.merge_stats = {
            'total_merged': 0,
            'averaged': 0,
            'use_slam': 0,
            'use_vive': 0,
            'both_high': 0,
            'skipped': 0,
            'no_match': 0
        }
        # SLAM VIVE ToF FPS 估计
        self.slam_timestamps = deque(maxlen=300)
        self.slam_current_fps = 0.0
        self.slam_last_fps_calc_time = time.time()

        if self.enable_vive:
            self.vive_timestamps = deque(maxlen=300)
            self.vive_current_fps = 0.0
            self.vive_last_fps_calc_time = time.time()

        self.tof_timestamps = deque(maxlen=300)
        self.tof_current_fps = 0.0
        self.tof_last_fps_calc_time = time.time()

        # ToF BAG文件写入
        self.tof_bag = None
        self.tof_bag_lock = threading.Lock()
        self.tof_saved_count = 0

        # Clamp 动态类型解析
        self.clamp_msg_class = None

        # 初始化 ROS
        import rospy
        from sensor_msgs.msg import Image, PointCloud2
        from cv_bridge import CvBridge
        import sensor_msgs.point_cloud2 as pc2
        from xv_sdk.msg import PoseStampedConfidence
        from roslib.message import get_message_class
        
        self.rospy = rospy
        self.Image = Image
        self.PointCloud2 = PointCloud2
        self.pc2 = pc2
        self.PoseStampedConfidence = PoseStampedConfidence
        self.get_message_class = get_message_class
        self.bridge = CvBridge()
        
        if not rospy.core.is_initialized():
            rospy.init_node('single_session_data_recorder', anonymous=True)

        signal.signal(signal.SIGINT, self.signal_handler)

        # 信息
        if not self.auto_start:
            print("=" * 60)
            print("摄像头 单设备数据采集器")
            print("=" * 60)
        if self.device_label:
            print(f"设备标签: {self.device_label}")
        print(f"XV 序列号: {self.xv_serial}")
        if self.enable_vive:
            print(f"Vive 序列号: {self.vive_serial}")
        else:
            print("Vive: 已禁用（未配置或不可用）")
        print(f"输出目录: {self.output_dir}")
        if not self.auto_start:
            print("=" * 60)

        # Topic 检查与订阅
        self.check_topics()
        self.subscribe_topics()
        
        # 验证第一条数据，如果不满足条件则不开启采集
        if not self.print_first_messages():
            print("\n\033[91m数据验证失败，程序退出。\033[0m")
            sys.exit(1)
        
        # 验证通过后，创建输出目录
        # 在多设备模式下（auto_start=True），延迟目录创建，等所有进程都通过验证后再创建
        # 在单设备模式下（auto_start=False），立即创建目录
        if not self.auto_start:
            os.makedirs(self.output_dir, exist_ok=True)

        if not self.auto_start:
            print("\n按回车键开始录制...")
            input()
            self.start_recording()
        else:
            print(f"[{self.device_label}] 已初始化，等待启动...")

    def calculate_linear_velocity(self, current_pose, prev_pose, dt):
        if prev_pose is None or dt <= 0:
            return 0.0
        p_curr = np.array(current_pose[:3])
        p_prev = np.array(prev_pose[:3])
        return float(np.linalg.norm(p_curr - p_prev) / dt)

    def calculate_angular_velocity(self, current_pose, prev_pose, dt):
        if prev_pose is None or dt <= 0:
            return 0.0
        try:
            q_curr = Rotation.from_quat(current_pose[3:])
            q_prev = Rotation.from_quat(prev_pose[3:])
            q_diff = q_curr * q_prev.inv()
            angle = np.linalg.norm(q_diff.as_rotvec())
            return float(angle / dt)
        except Exception:
            return 0.0

    def smooth_velocity(self, velocity_history):
        if len(velocity_history) == 0:
            return 0.0
        return float(sum(velocity_history) / len(velocity_history))
    
    def _safe_flush_and_fsync(self, fh):
        try:
            if fh and (not fh.closed):
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            pass

    def flush_all_stream_files(self):
        try:
            self.process_pending_vive_data()
        except Exception:
            pass
        self._safe_flush_and_fsync(self.timestamp_file)
        self._safe_flush_and_fsync(self.slam_file_handle)
        self._safe_flush_and_fsync(self.vive_file_handle)
        self._safe_flush_and_fsync(self.vive_world_file_handle)
        self._safe_flush_and_fsync(self.clamp_file_handle)
        self._safe_flush_and_fsync(self.tof_timestamp_file)

    def average_poses(self, pose1, pose2):
        try:
            # 位置均值
            pos1 = np.asarray(pose1[:3], dtype=float)
            pos2 = np.asarray(pose2[:3], dtype=float)
            pos_avg = (pos1 + pos2) / 2.0

            # 构造两帧 quaternion 序列
            q_arr = np.array([
                pose1[3:7],  # q1: [qx, qy, qz, qw]
                pose2[3:7],  # q2
            ], dtype=float)

            key_times = [0.0, 1.0]
            key_rots = Rotation.from_quat(q_arr)
            slerp = Slerp(key_times, key_rots)
            q_avg = slerp(0.5).as_quat()

            averaged_pose = pos_avg.tolist() + q_avg.tolist()
            return averaged_pose

        except Exception as e:
            print(f"警告: SLERP失败，使用降级方案: {e}")
            pos_avg = [(pose1[i] + pose2[i]) / 2.0 for i in range(3)]
            q1 = np.asarray(pose1[3:7], dtype=float)
            q2 = np.asarray(pose2[3:7], dtype=float)
            if np.dot(q1, q2) < 0.0:
                q2 = -q2
            q_avg = q1 + q2
            norm = np.linalg.norm(q_avg)
            if norm > 1e-8:
                q_avg /= norm
            else:
                q_avg = q1
            return pos_avg + q_avg.tolist()

    def merge_poses(self, slam_pose, vive_pose, slam_vel, vive_vel):
        th = self.velocity_threshold
        if slam_vel < th and vive_vel < th:
            merged = self.average_poses(slam_pose, vive_pose)
            self.merge_stats['averaged'] += 1
            return merged, "high", "averaged"
        elif slam_vel >= th and vive_vel < th:
            self.merge_stats['use_vive'] += 1
            return vive_pose, "medium", "use_vive"
        elif slam_vel < th and vive_vel >= th:
            self.merge_stats['use_slam'] += 1
            return slam_pose, "medium", "use_slam"
        else:
            self.merge_stats['both_high'] += 1
            return None, None, None

    def check_topics(self):
        """检查话题是否存在，如果失败则抛出异常而不是直接退出"""
        try:
            topics = self.rospy.get_published_topics()
            topic_names = [t[0] for t in topics]

            missing = []
            if self.rgb_topic not in topic_names:
                missing.append(self.rgb_topic)
            if self.slam_topic not in topic_names:
                missing.append(self.slam_topic)
            if self.enable_tof and self.tof_topic not in topic_names:
                missing.append(self.tof_topic)
            if self.enable_vive and self.vive_topic and self.vive_topic not in topic_names:
                missing.append(self.vive_topic)
            if self.enable_vive and self.vive_world_topic and self.vive_world_topic not in topic_names:
                missing.append(self.vive_world_topic)
            if self.clamp_topic not in topic_names:
                print(f"警告: 未发现 Clamp 话题: {self.clamp_topic}（将继续运行）")

            if missing:
                error_msg = "以下话题不存在:\n"
                for t in missing:
                    error_msg += f"  - {t}\n"
                error_msg += "请确认设备与节点已启动且序列号正确"
                print(f"错误: {error_msg}")
                raise RuntimeError(error_msg)

            print("Topic 检查通过:")
            print("  RGB:", self.rgb_topic)
            print("  SLAM:", self.slam_topic)
            print("  ToF:", self.tof_topic if self.enable_tof else "已禁用")
            if self.enable_vive:
                print("  Vive (应用坐标系):", self.vive_topic)
                print("  Vive (世界坐标系):", self.vive_world_topic)
            else:
                print("  Vive: 已禁用")
            print("  Clamp:", self.clamp_topic)

        except Exception as e:
            print(f"错误: 无法检查话题状态: {e}")
            raise

    def subscribe_topics(self):
        """订阅所有数据话题"""
        from geometry_msgs.msg import PoseStamped

        self.rgb_subscriber = self.rospy.Subscriber(
            self.rgb_topic, self.Image, self.rgb_callback,
            queue_size=120, buff_size=2**26, tcp_nodelay=True
        )
        self.slam_subscriber = self.rospy.Subscriber(
            self.slam_topic, self.PoseStampedConfidence, self.slam_callback,
            queue_size=200, buff_size=2**20, tcp_nodelay=True
        )
        if self.enable_tof:
            self.tof_subscriber = self.rospy.Subscriber(
                self.tof_topic, self.PointCloud2, self.tof_callback,
                queue_size=60, buff_size=2**24, tcp_nodelay=True
            )
        if self.enable_vive and self.vive_topic:
            self.vive_subscriber = self.rospy.Subscriber(
                self.vive_topic, PoseStamped, self.vive_callback,
                queue_size=100, buff_size=2**20, tcp_nodelay=True
            )
            # 订阅世界坐标系 topic
            self.vive_world_subscriber = self.rospy.Subscriber(
                self.vive_world_topic, PoseStamped, self.vive_world_callback,
                queue_size=100, buff_size=2**20, tcp_nodelay=True
            )
        else:
            self.vive_subscriber = None
            self.vive_world_subscriber = None
        self.clamp_subscriber = self.rospy.Subscriber(
            self.clamp_topic, self.rospy.AnyMsg, self.clamp_callback,
            queue_size=100, buff_size=2**20, tcp_nodelay=True
        )
    
    def print_first_messages(self):
        """等待并打印第一条 clamp、slam、vive 数据，并验证数据是否符合采集条件
        
        Returns:
            bool: True表示数据验证通过，可以开始采集；False表示验证失败，不应开始采集
        """
        print("\n等待第一条数据...")
        from geometry_msgs.msg import PoseStamped
        import math
        
        # 检查可用的 topics
        try:
            topics = rospy.get_published_topics()
            topic_names = [t[0] for t in topics]
        except Exception:
            topic_names = []
        
        # 用于存储获取到的数据
        clamp_data_value = None
        slam_pos = None
        vive_pos = None
        
        # SLAM 数据
        try:
            slam_msg = rospy.wait_for_message(self.slam_topic, PoseStampedConfidence, timeout=10.0)
            timestamp = slam_msg.poseMsg.header.stamp.to_sec()
            x = slam_msg.poseMsg.pose.position.x
            y = slam_msg.poseMsg.pose.position.y
            z = slam_msg.poseMsg.pose.position.z
            qx = slam_msg.poseMsg.pose.orientation.x
            qy = slam_msg.poseMsg.pose.orientation.y
            qz = slam_msg.poseMsg.pose.orientation.z
            qw = slam_msg.poseMsg.pose.orientation.w
            confidence = slam_msg.confidence if hasattr(slam_msg, 'confidence') else 0.0
            slam_pos = (x, y, z)
            print(f"\033[92m  SLAM: timestamp={timestamp:.6f}, pos=({x:.3f}, {y:.3f}, {z:.3f}), "
                  f"quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}), confidence={confidence:.3f}\033[0m")
        except rospy.ROSException:
            print(f"  SLAM: 超时未收到数据 (topic: {self.slam_topic})")
        
        # VIVE 数据（仅当 topic 存在时）
        if self.vive_topic in topic_names:
            try:
                vive_msg = rospy.wait_for_message(self.vive_topic, PoseStamped, timeout=10.0)
                timestamp = vive_msg.header.stamp.to_sec()
                x = vive_msg.pose.position.x
                y = vive_msg.pose.position.y
                z = vive_msg.pose.position.z
                qx = vive_msg.pose.orientation.x
                qy = vive_msg.pose.orientation.y
                qz = vive_msg.pose.orientation.z
                qw = vive_msg.pose.orientation.w
                vive_pos = (x, y, z)
                print(f"\033[92m  VIVE (应用坐标系): timestamp={timestamp:.6f}, pos=({x:.3f}, {y:.3f}, {z:.3f}), "
                      f"quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})\033[0m")
            except rospy.ROSException:
                print(f"  VIVE: 超时未收到数据 (topic: {self.vive_topic})")
        else:
            print(f"  VIVE: topic 不存在，跳过 (topic: {self.vive_topic})")
        
        # VIVE 世界坐标系数据（仅当 topic 存在时）
        if self.enable_vive and self.vive_world_topic and self.vive_world_topic in topic_names:
            try:
                vive_world_msg = rospy.wait_for_message(self.vive_world_topic, PoseStamped, timeout=10.0)
                timestamp = vive_world_msg.header.stamp.to_sec()
                x = vive_world_msg.pose.position.x
                y = vive_world_msg.pose.position.y
                z = vive_world_msg.pose.position.z
                qx = vive_world_msg.pose.orientation.x
                qy = vive_world_msg.pose.orientation.y
                qz = vive_world_msg.pose.orientation.z
                qw = vive_world_msg.pose.orientation.w
                print(f"\033[92m  VIVE (世界坐标系): timestamp={timestamp:.6f}, pos=({x:.3f}, {y:.3f}, {z:.3f}), "
                      f"quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})\033[0m")
            except rospy.ROSException:
                print(f"  VIVE (世界坐标系): 超时未收到数据 (topic: {self.vive_world_topic})")
        
        # CLAMP 数据
        try:
            clamp_msg = rospy.wait_for_message(self.clamp_topic, rospy.AnyMsg, timeout=10.0)
            # 动态解析 CLAMP 消息
            if hasattr(clamp_msg, '_connection_header'):
                type_str = clamp_msg._connection_header.get('type', '')
                if type_str:
                    clamp_msg_class = get_message_class(type_str)
                    if clamp_msg_class:
                        real_msg = clamp_msg_class()
                        real_msg.deserialize(clamp_msg._buff)
                        data_value = getattr(real_msg, 'data', None)
                        if data_value is not None:
                            ts_sec = None
                            if hasattr(real_msg, 'header') and hasattr(real_msg.header, 'stamp'):
                                if (real_msg.header.stamp.secs != 0) or (real_msg.header.stamp.nsecs != 0):
                                    ts_sec = real_msg.header.stamp.to_sec()
                            if ts_sec is None:
                                ts_sec = rospy.get_time()
                            clamp_data_value = data_value
                            print(f"\033[92m  CLAMP: timestamp={ts_sec:.6f}, data={data_value}\033[0m")
                        else:
                            print(f"  CLAMP: 收到消息但 data 字段为空")
                    else:
                        print(f"  CLAMP: 无法解析消息类型: {type_str}")
                else:
                    print(f"  CLAMP: 消息类型信息不可用")
            else:
                print(f"  CLAMP: 消息格式异常")
        except rospy.ROSException:
            print(f"  CLAMP: 超时未收到数据 (topic: {self.clamp_topic})")
        except Exception as e:
            print(f"  CLAMP: 处理消息时出错: {e}")
        
        # 验证数据是否符合采集条件
        print("\n验证数据是否符合采集条件...")
        validation_errors = []
        
        # 验证夹爪数值：应在 85-91 范围内
        if clamp_data_value is not None:
            if not (85 <= clamp_data_value <= 91):
                validation_errors.append(f"夹爪数值 {clamp_data_value} 不在允许范围 [85, 91] 内")
                print(f"\033[91m  ✗ 夹爪数值验证失败: {clamp_data_value} (应在 85-91 范围内)\033[0m")
            else:
                print(f"\033[92m  ✓ 夹爪数值验证通过: {clamp_data_value}\033[0m")
        else:
            validation_errors.append("未获取到夹爪数据")
            print(f"\033[91m  ✗ 夹爪数据验证失败: 未获取到数据\033[0m")
        
        # 验证 SLAM 位置：应在 [0, 0, 0] 的 6mm (0.006m) 范围内
        if slam_pos is not None:
            distance = math.sqrt(slam_pos[0]**2 + slam_pos[1]**2 + slam_pos[2]**2)
            if distance > 0.006:  # 6mm = 0.006m
                validation_errors.append(f"SLAM 位置 ({slam_pos[0]:.3f}, {slam_pos[1]:.3f}, {slam_pos[2]:.3f}) 距离原点 {distance*100:.2f}cm，超过 6mm 限制")
                print(f"\033[91m  ✗ SLAM 位置验证失败: 距离原点 {distance*1000:.2f}mm (应在 6mm 内)\033[0m")
            else:
                print(f"\033[92m  ✓ SLAM 位置验证通过: 距离原点 {distance*1000:.2f}mm\033[0m")
        else:
            validation_errors.append("未获取到 SLAM 数据")
            print(f"\033[91m  ✗ SLAM 数据验证失败: 未获取到数据\033[0m")
        
        # 验证 VIVE 位置：应在 [0, 0, 0] 的 6mm (0.006m) 范围内（仅当 topic 存在时，使用应用坐标系）
        if self.vive_topic in topic_names:
            if vive_pos is not None:
                distance = math.sqrt(vive_pos[0]**2 + vive_pos[1]**2 + vive_pos[2]**2)
                if distance > 0.006:  # 6mm = 0.006m
                    validation_errors.append(f"VIVE 位置 ({vive_pos[0]:.3f}, {vive_pos[1]:.3f}, {vive_pos[2]:.3f}) 距离原点 {distance*100:.2f}cm，超过 6mm 限制")
                    print(f"\033[91m  ✗ VIVE 位置验证失败: 距离原点 {distance*1000:.2f}mm (应在 6mm 内)\033[0m")
                else:
                    print(f"\033[92m  ✓ VIVE 位置验证通过: 距离原点 {distance*1000:.2f}mm\033[0m")
            else:
                validation_errors.append("未获取到 VIVE 数据")
                print(f"\033[91m  ✗ VIVE 数据验证失败: 未获取到数据\033[0m")
        
        # 返回验证结果
        if validation_errors:
            print("\n\033[91m" + "="*60)
            print("数据验证失败，本次采集将不会启动")
            print("="*60 + "\033[0m")
            for error in validation_errors:
                print(f"  • {error}")
            return False
        else:
            print("\n\033[92m" + "="*60)
            print("该设备数据验证通过，可以开始采集")
            print("="*60 + "\033[0m")
            return True

    def prepare_recording(self):
        if self.recording_state != RecordingState.IDLE:
            return
        if self.device_label:
            print(f"[{self.device_label}] 准备录制...")

        # 显示队列大小信息
        queue_size = self.rgb_queue.maxsize
        if self.max_rgb_count is not None:
            print(f"队列大小: {queue_size} (max_rgb_count={self.max_rgb_count} 的 50%)")
        else:
            print(f"队列大小: {queue_size} (默认值)")

        self.init_files()

        # 计数与状态复位
        self.rgb_count = self.slam_count = 0
        if self.enable_vive:
            self.vive_count = 0
        self.tof_count = self.clamp_count = self.merged_count = 0

        self.tof_stats = {
            'min_depth': float('inf'),
            'max_depth': 0,
            'total_points': 0,
            'total_invalid_points': 0
        }
        self.merge_stats = {
            'total_merged': 0,
            'averaged': 0,
            'use_slam': 0,
            'use_vive': 0,
            'both_high': 0,
            'skipped': 0,
            'no_match': 0
        }

        # 清对齐状态
        self.slam_buffer.clear()
        if self.enable_vive:
            self.vive_buffer.clear()
            self.vive_pending_buffer.clear()
        self.slam_prev_pose = None
        self.slam_prev_timestamp = None
        self.slam_linear_velocity = 0.0
        self.slam_angular_velocity = 0.0
        self.slam_velocity_history.clear()
        if self.enable_vive:
            self.vive_prev_pose = None
            self.vive_prev_timestamp = None
            self.vive_linear_velocity = 0.0
            self.vive_angular_velocity = 0.0
            self.vive_velocity_history.clear()
        self.slam_first_timestamp = None
        if self.enable_vive:
            self.vive_first_timestamp = None
            self.vive_time_offset = None
            self.offset_ready = False
        self.first_rgb_frame = False
        self.should_stop_recording.clear()  # 重置停止标志
        self._auto_stop_triggered = False  # 重置自动停止触发标志
        self._progress_alerted.clear()  # 重置进度提示音标志

        # 启动所有写盘线程（异步写入）
        self._stop_writers.clear()
        # 调用新的MP4写入器启动函数
        self.start_rgb_writer_mp4()
        self.start_slam_writer()
        if self.enable_vive:
            self.start_vive_writer()
            self.start_vive_world_writer()  # 启动世界坐标系写入线程
        if self.enable_tof:
            self.start_tof_writer()
        self.start_clamp_writer()
        
        # 启动队列长度监控线程
        self.start_queue_monitor()

        if self.device_label:
            print(f"[{self.device_label}] 准备完成")

    def start_recording(self, global_start_time=None):
        if self.recording_state != RecordingState.IDLE:
            return
        if not hasattr(self, 'slam_file_handle') or self.slam_file_handle is None:
            if not self.auto_start:
                print("\n" + "="*60)
                print("开始录制...")
            self.prepare_recording()

        self.recording_state = RecordingState.RECORDING
        self.global_start_time = global_start_time
        self.recording_start_time = time.time()
        if self.global_start_time is not None:
            delta_ms = (self.recording_start_time - self.global_start_time) * 1000.0
            prefix = f"[{self.device_label}] " if self.device_label else ""
            print(f"{prefix}同步开始目标: {self.global_start_time:.3f}s, 实际: {self.recording_start_time:.3f}s, 偏差: {delta_ms:.1f} ms")

        if self.device_label:
            print(f"[{self.device_label}] 开始录制...")
        else:
            print("状态: 录制中...")
        
        if self.max_rgb_count is not None:
            print(f"目标RGB帧数: {self.max_rgb_count}，达到后自动停止")
            print("="*60)
        else:
            # 如果没有设置最大帧数，提示错误
            print("错误: 未设置最大RGB帧数，无法启动录制")
            print("请使用 --max-rgb 参数指定目标帧数")
            self.recording_state = RecordingState.IDLE
            return

    def freeze_recording(self):
        if self.recording_state != RecordingState.RECORDING:
            return
        self.recording_end_time = time.time()
        self.recording_state = RecordingState.SAVING

    def stop_recording(self):
        if self.recording_state != RecordingState.RECORDING:
            return
        tail_grace_s = 0.4  
        time.sleep(tail_grace_s)

        # 进入保存阶段
        self.freeze_recording()
        print("停止录制，等待写盘完成...")

        self.save_all_data()
        self.display_summary()

        self.recording_state = RecordingState.FINISHED
        self.running = False

        print("\n录制完成！程序即将退出...")
        print("="*60)

    def run(self):
        try:
            # 检查是否已经在__init__中启动了录制
            if self.recording_state == RecordingState.RECORDING:
                pass
            elif not self.auto_start:
                # 如果不是自动启动模式且还未启动，需要先按回车开始
                if self.max_rgb_count is not None:
                    print(f"\n按回车键开始录制（目标: {self.max_rgb_count} 帧）...")
                else:
                    print("\n按回车键开始录制...")
                input()
                self.start_recording()
            
            # 等待录制完成（自动停止或手动停止）
            while not self.rospy.is_shutdown() and self.running:
                # 优先检查是否应该停止（达到预设帧数）
                if self.should_stop_recording.is_set():
                    self.stop_recording()
                    break
                # 如果录制状态变为FINISHED（可能是自动停止），也退出循环
                if self.recording_state == RecordingState.FINISHED:
                    break
                self.rospy.sleep(0.1)
        except self.rospy.ROSInterruptException:
            pass
        except KeyboardInterrupt:
            print("\n收到中断信号...")
        finally:
            self.cleanup()

    def signal_handler(self, signum, frame):
        print("\n收到中断信号...")
        self.running = False

    def init_files(self):
        try:
            print(f"初始化文件，目录: {self.output_dir}")

            # 队列长度日志文件
            queue_log_path = os.path.join(self.output_dir, "queue_lengths.csv")
            self.queue_log_file = open(queue_log_path, 'w', newline='')
            self.queue_log_writer = csv.writer(self.queue_log_file)
            self.queue_log_writer.writerow(['timestamp', 'rgb_queue', 'slam_queue', 'vive_queue', 'tof_queue', 'clamp_queue'])
            print(f"  队列长度日志文件创建: {queue_log_path}")

            # RGB 目录与时间戳 CSV
            rgb_dir = os.path.join(self.output_dir, "RGB_Images")
            os.makedirs(rgb_dir, exist_ok=True)
            timestamp_path = os.path.join(rgb_dir, "timestamps.csv")
            self.timestamp_file = open(timestamp_path, 'w', newline='')
            self.timestamp_writer = csv.writer(self.timestamp_file)
            self.timestamp_writer.writerow(['frame_index','seq','header_stamp'])
            print(f"  RGB时间戳文件创建: {timestamp_path}")

            # SLAM（保存原始数据）
            slam_dir = os.path.join(self.output_dir, "SLAM_Poses")
            os.makedirs(slam_dir, exist_ok=True)
            slam_file_path = os.path.join(slam_dir, "slam_raw.txt")
            self.slam_file_handle = open(slam_file_path, 'w')
            print(f"  SLAM原始文件创建: {slam_file_path}")

            # ToF
            if self.enable_tof:
                self.init_tof_writer()
                tof_dir = os.path.join(self.output_dir, "ToF_PointClouds")
                tof_timestamp_path = os.path.join(tof_dir, "timestamps.csv")
                self.tof_timestamp_file = open(tof_timestamp_path, 'w', newline='')
                self.tof_timestamp_writer = csv.writer(self.tof_timestamp_file)
                self.tof_timestamp_writer.writerow(['pointcloud_index','timestamp'])
                print(f"  ToF时间戳文件创建: {tof_timestamp_path}")

            # Clamp
            clamp_dir = os.path.join(self.output_dir, "Clamp_Data")
            os.makedirs(clamp_dir, exist_ok=True)
            clamp_file_path = os.path.join(clamp_dir, "clamp_data_tum.txt")
            self.clamp_file_handle = open(clamp_file_path, 'w')
            print(f"  Clamp TUM文件创建: {clamp_file_path}")

            # Vive（保存原始数据）
            if self.enable_vive:
                vive_dir = os.path.join(self.output_dir, "Vive_Poses")
                os.makedirs(vive_dir, exist_ok=True)
                # 应用坐标系文件（用于转换到Gripper）
                vive_file_path = os.path.join(vive_dir, "vive_app.txt")
                self.vive_file_handle = open(vive_file_path, 'w')
                print(f"  Vive应用坐标系文件创建: {vive_file_path}")
                # 世界坐标系文件（用于相对变换计算）
                vive_world_file_path = os.path.join(vive_dir, "vive_raw.txt")
                self.vive_world_file_handle = open(vive_world_file_path, 'w')
                print(f"  Vive世界坐标系文件创建: {vive_world_file_path}")
            else:
                self.vive_file_handle = None
                self.vive_world_file_handle = None

            # Merged 目录
            merged_dir = os.path.join(self.output_dir, "Merged_Trajectory")
            os.makedirs(merged_dir, exist_ok=True)

            print("所有文件初始化完成")
        except Exception as e:
            print(f"错误: 初始化文件失败: {e}")
            self.recording_state = RecordingState.IDLE

    def cleanup_files(self):
        try:
            if self.timestamp_file:
                self.timestamp_file.flush(); self.timestamp_file.close()
                self.timestamp_file = None
                print("  RGB时间戳文件已关闭")
            if self.slam_file_handle:
                self.slam_file_handle.flush(); self.slam_file_handle.close()
                self.slam_file_handle = None
                print("  SLAM文件已关闭")
            if self.tof_timestamp_file:
                self.tof_timestamp_file.flush(); self.tof_timestamp_file.close()
                self.tof_timestamp_file = None
                print("  ToF时间戳文件已关闭")
            if self.enable_tof and self.tof_bag:
                try:
                    with self.tof_bag_lock:
                        self.tof_bag.close()
                    print("  ToF BAG文件已关闭")
                except Exception:
                    pass
                self.tof_bag = None
            if self.clamp_file_handle:
                self.clamp_file_handle.flush(); self.clamp_file_handle.close()
                self.clamp_file_handle = None
                print("  Clamp文件已关闭")
            if self.vive_file_handle:
                self.vive_file_handle.flush(); self.vive_file_handle.close()
                self.vive_file_handle = None
                print("  Vive应用坐标系文件已关闭")
            if self.vive_world_file_handle:
                self.vive_world_file_handle.flush(); self.vive_world_file_handle.close()
                self.vive_world_file_handle = None
                print("  Vive世界坐标系文件已关闭")
            
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                print("  RGB MP4文件已关闭")

        except Exception as e:
            print(f"错误: 清理文件时出错: {e}")

    def cleanup(self):
        print("\n正在清理资源...")
        if self.recording_state == RecordingState.RECORDING:
            self.stop_recording()

        # 先确保所有数据写完
        if self.recording_state != RecordingState.FINISHED:
            self.save_all_data()

        # 停止所有后台写入线程
        try:
            self._stop_writers.set()
            # 发送停止信号到所有队列
            try:
                self.rgb_queue.put(None)
            except Exception:
                pass
            try:
                self.slam_queue.put(None)
            except Exception:
                pass
            if self.enable_vive and self.vive_queue:
                try:
                    self.vive_queue.put(None)
                except Exception:
                    pass
            if self.enable_vive and self.vive_world_queue:
                try:
                    self.vive_world_queue.put(None)
                except Exception:
                    pass
            if self.enable_tof and self.tof_queue:
                try:
                    self.tof_queue.put(None)
                except Exception:
                    pass
            try:
                self.clamp_queue.put(None)
            except Exception:
                pass
            
            # 等待所有线程退出
            if self.rgb_writer_thread:
                self.rgb_writer_thread.join(timeout=5)
            if self.slam_writer_thread:
                self.slam_writer_thread.join(timeout=5)
            if self.enable_vive and self.vive_writer_thread:
                self.vive_writer_thread.join(timeout=5)
            if self.enable_tof and self.tof_writer_thread:
                self.tof_writer_thread.join(timeout=5)
            if self.clamp_writer_thread:
                self.clamp_writer_thread.join(timeout=5)
            # 等待队列监控线程退出
            if self.queue_monitor_thread:
                self.queue_monitor_thread.join(timeout=5)
        except Exception:
            pass

        self.cleanup_files()
        
        # 关闭队列长度日志文件
        if self.queue_log_file:
            try:
                self.queue_log_file.flush()
                self.queue_log_file.close()
                self.queue_log_file = None
                self.queue_log_writer = None
                print("  队列长度日志文件已关闭")
            except Exception as e:
                print(f"  警告: 关闭队列长度日志文件失败: {e}")

        print("="*60)
        print("数据采集完成!")
        print(f"数据已保存到: {self.output_dir}")
        print("="*60)

    def start_rgb_writer_mp4(self):
        """
        启动RGB写入线程
        功能：从队列消费RGB帧(OpenCV格式)，直接写入MP4文件和时间戳CSV
        """
        rgb_dir = os.path.join(self.output_dir, "RGB_Images")
        os.makedirs(rgb_dir, exist_ok=True)
        
        video_path = os.path.join(rgb_dir, "video.mp4")
        # Video writer 将在获取第一帧时初始化，以便知道分辨率
        self.video_writer = None

        def _loop():
            while True:
                try:
                    item = self.rgb_queue.get(timeout=0.2)
                except Empty:
                    if self._stop_writers.is_set():
                        continue
                    else:
                        continue

                if item is None:
                    self.rgb_queue.task_done()
                    break

                frame_index, frame_image, raw_ts, seq = item
                try:
                    # 初始化 video writer (如果是第一帧)
                    if self.video_writer is None:
                        height, width = frame_image.shape[:2]
                        # 使用 mp4v 编码器
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.video_writer = cv2.VideoWriter(video_path, fourcc, 60.0, (width, height))
                        print(f"  视频文件已创建: {video_path} (分辨率: {width}x{height}, 60FPS)")

                    # [修改] 直接写入帧
                    self.video_writer.write(frame_image)

                    # 写入时间戳 CSV
                    if self.timestamp_writer and self.timestamp_file and not self.timestamp_file.closed:
                        self.timestamp_writer.writerow([
                            int(frame_index), int(seq),
                            f"{raw_ts:.9f}"
                        ])

                    self.rgb_count += 1
                    
                    # 显示进度条和检查是否达到预设值
                    if self.max_rgb_count is not None:
                        current_time = time.time()
                        if current_time - self.last_progress_update_time >= self.progress_update_interval:
                            self._display_progress()
                            self.last_progress_update_time = current_time
                        
                        # 检查是否达到预设值
                        if self.max_rgb_count is not None and (not self._auto_stop_triggered) and \
                           self.rgb_count >= self.max_rgb_count and \
                           self.recording_state == RecordingState.RECORDING:
                            self._auto_stop_triggered = True
                            print(f"\n达到预设RGB帧数 ({self.max_rgb_count})，自动停止录制...")
                            self.should_stop_recording.set()
                except Exception as e:
                    if self.running:
                        print(f"错误: 写入RGB MP4时出错: {e}")
                finally:
                    self.rgb_queue.task_done()

        self.rgb_writer_thread = threading.Thread(target=_loop, daemon=True)
        self.rgb_writer_thread.start()

    def start_slam_writer(self):
        """启动SLAM写入线程"""
        def _loop():
            while True:
                try:
                    item = self.slam_queue.get(timeout=0.2)
                except Empty:
                    if self._stop_writers.is_set():
                        continue
                    else:
                        continue

                if item is None:
                    self.slam_queue.task_done()
                    break

                tum_line, timestamp = item
                try:
                    if self.slam_file_handle:
                        self.slam_file_handle.write(tum_line)
                    self.slam_count += 1
                    self.slam_buffer.append((tum_line, timestamp))
                except Exception as e:
                    if self.running:
                        print(f"错误: 写入SLAM数据时出错: {e}")
                finally:
                    self.slam_queue.task_done()

        self.slam_writer_thread = threading.Thread(target=_loop, daemon=True)
        self.slam_writer_thread.start()

    def start_vive_writer(self):
        """启动Vive写入线程（应用坐标系）"""
        if not self.enable_vive or self.vive_queue is None:
            return
            
        def _loop():
            while True:
                try:
                    item = self.vive_queue.get(timeout=0.2)
                except Empty:
                    if self._stop_writers.is_set():
                        continue
                    else:
                        continue

                if item is None:
                    self.vive_queue.task_done()
                    break

                tum_line = item
                try:
                    if self.vive_file_handle:
                        self.vive_file_handle.write(tum_line)
                    self.vive_count += 1
                except Exception as e:
                    if self.running:
                        print(f"错误: 写入Vive数据时出错: {e}")
                finally:
                    self.vive_queue.task_done()

        self.vive_writer_thread = threading.Thread(target=_loop, daemon=True)
        self.vive_writer_thread.start()

    def start_vive_world_writer(self):
        """启动Vive世界坐标系写入线程"""
        if not self.enable_vive or self.vive_world_queue is None:
            return
            
        def _loop():
            while True:
                try:
                    item = self.vive_world_queue.get(timeout=0.2)
                except Empty:
                    if self._stop_writers.is_set():
                        continue
                    else:
                        continue

                if item is None:
                    self.vive_world_queue.task_done()
                    break

                tum_line = item
                try:
                    if self.vive_world_file_handle:
                        self.vive_world_file_handle.write(tum_line)
                except Exception as e:
                    if self.running:
                        print(f"错误: 写入Vive世界坐标系数据时出错: {e}")
                finally:
                    self.vive_world_queue.task_done()

        self.vive_world_writer_thread = threading.Thread(target=_loop, daemon=True)
        self.vive_world_writer_thread.start()

    def start_tof_writer(self):
        """启动ToF写入线程"""
        if not self.enable_tof or self.tof_queue is None:
            return
            
        def _loop():
            while True:
                try:
                    item = self.tof_queue.get(timeout=0.2)
                except Empty:
                    if self._stop_writers.is_set():
                        continue
                    else:
                        continue

                if item is None:
                    self.tof_queue.task_done()
                    break

                msg, timestamp = item
                try:
                    # 使用线程锁保护BAG文件写入
                    with self.tof_bag_lock:
                        if self.tof_bag:
                            self.tof_bag.write('/camera/depth/color/points', msg, msg.header.stamp)
                            self.tof_saved_count += 1
                    
                    # 写入时间戳到CSV文件
                    if self.tof_timestamp_writer:
                        self.tof_timestamp_writer.writerow([self.tof_saved_count - 1, f"{timestamp:.6f}"])
                    
                    self.tof_count += 1
                    self.calculate_tof_fps()
                except Exception as e:
                    self.tof_dropped_count += 1
                    if self.running and (self.tof_dropped_count % 10 == 1):
                        print(f"警告: ToF写入失败 (已丢失{self.tof_dropped_count}帧): {e}")
                finally:
                    self.tof_queue.task_done()

        self.tof_writer_thread = threading.Thread(target=_loop, daemon=True)
        self.tof_writer_thread.start()

    def start_clamp_writer(self):
        """启动Clamp写入线程"""
        def _loop():
            while True:
                try:
                    item = self.clamp_queue.get(timeout=0.2)
                except Empty:
                    if self._stop_writers.is_set():
                        continue
                    else:
                        continue

                if item is None:
                    self.clamp_queue.task_done()
                    break

                tum_line = item
                try:
                    if self.clamp_file_handle:
                        self.clamp_file_handle.write(tum_line)
                    self.clamp_count += 1
                except Exception as e:
                    if self.running:
                        print(f"错误: 写入Clamp数据时出错: {e}")
                finally:
                    self.clamp_queue.task_done()

        self.clamp_writer_thread = threading.Thread(target=_loop, daemon=True)
        self.clamp_writer_thread.start()

    def start_queue_monitor(self):
        """启动队列长度监控线程，定期记录队列长度到日志文件"""
        def _monitor_loop():
            while not self._stop_writers.is_set() and self.running:
                try:
                    current_time = time.time()
                    
                    # 获取各队列长度
                    rgb_len = self.rgb_queue.qsize()
                    slam_len = self.slam_queue.qsize()
                    vive_len = self.vive_queue.qsize() if self.vive_queue else 0
                    tof_len = self.tof_queue.qsize() if self.tof_queue else 0
                    clamp_len = self.clamp_queue.qsize()
                    
                    # 写入日志（仅在录制状态下记录）
                    if self.recording_state == RecordingState.RECORDING and self.queue_log_writer:
                        self.queue_log_writer.writerow([
                            f"{current_time:.6f}",
                            rgb_len,
                            slam_len,
                            vive_len,
                            tof_len,
                            clamp_len
                        ])
                        self.queue_log_file.flush()
                    
                    time.sleep(self.queue_log_interval)
                except Exception as e:
                    if self.running:
                        print(f"警告: 队列监控线程出错: {e}")
                    time.sleep(self.queue_log_interval)
        
        self.queue_monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
        self.queue_monitor_thread.start()

    def init_tof_writer(self):
        """
        初始化ToF BAG文件写入器（在设备进程内）
        """
        if not self.enable_tof:
            return
        
        try:
            import rosbag
            tof_dir = os.path.join(self.output_dir, "ToF_PointClouds")
            os.makedirs(tof_dir, exist_ok=True)
            bag_path = os.path.join(tof_dir, "pointclouds.bag")
            self.tof_bag = rosbag.Bag(bag_path, 'w')
            print(f"  ToF BAG文件创建: {bag_path}")
        except Exception as e:
            print(f"  警告: 初始化ToF BAG文件失败: {e}")
            self.enable_tof = False

    def rgb_callback(self, msg):
        """RGB图像回调 - [修改] 转换为OpenCV图片格式放入队列"""
        if self.recording_state != RecordingState.RECORDING:
            return
        try:
            rgb_ts = msg.header.stamp.to_sec()
            seq = getattr(msg.header, 'seq', -1)

            # FPS 估计
            if (self.rgb_count % self.fps_calc_interval) == 0:
                self.calculate_rgb_fps(rgb_ts)

            # [修改] 使用 cv_bridge 将 ROS 消息转换为 OpenCV 图片 (numpy array)
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                print(f"CV Bridge Error: {e}")
                return

            # 获取宽高用于初始化writer (第一帧)
            if not self.first_rgb_frame:
                h, w = cv_image.shape[:2]
                self.frame_height, self.frame_width = int(h), int(w)
                self.first_rgb_frame = True

            # 将图片对象放入队列
            frame_index = self.rgb_frame_index
            self.rgb_frame_index += 1
            self.rgb_queue.put((frame_index, cv_image, rgb_ts, int(seq)))

        except Exception as e:
            if self.running:
                print(f"错误: 处理RGB图像时出错: {e}")

    def slam_callback(self, msg):
        """SLAM回调函数：只保存原始数据，不进行计算"""
        if self.recording_state != RecordingState.RECORDING:
            return
        try:
            timestamp = msg.poseMsg.header.stamp.to_sec()

            if self.slam_first_timestamp is None:
                self.slam_first_timestamp = timestamp
                self.try_calculate_vive_offset()

            self.calculate_slam_fps()

            # 只保存原始数据（XV坐标系）
            x = msg.poseMsg.pose.position.x
            y = msg.poseMsg.pose.position.y
            z = msg.poseMsg.pose.position.z
            qx = msg.poseMsg.pose.orientation.x
            qy = msg.poseMsg.pose.orientation.y
            qz = msg.poseMsg.pose.orientation.z
            qw = msg.poseMsg.pose.orientation.w
            
            # 保存原始数据：timestamp x y z qx qy qz qw
            tum_line = f"{timestamp:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n"

            # 将数据放入队列，由写入线程异步处理
            self.slam_queue.put((tum_line, timestamp))
        except Exception as e:
            if self.running:
                print(f"错误: 处理SLAM位姿时出错: {e}")

    def vive_callback(self, msg):
        """Vive回调函数：保存应用坐标系数据（用于转换到Gripper）"""
        if self.recording_state != RecordingState.RECORDING:
            return
        try:
            timestamp = msg.header.stamp.to_sec()
            if self.vive_first_timestamp is None:
                self.vive_first_timestamp = timestamp
                self.try_calculate_vive_offset()

            self.calculate_vive_fps()

            # 保存应用坐标系数据（已归零）
            x = msg.pose.position.x
            y = msg.pose.position.y
            z = msg.pose.position.z
            qx = msg.pose.orientation.x
            qy = msg.pose.orientation.y
            qz = msg.pose.orientation.z
            qw = msg.pose.orientation.w

            # 保存应用坐标系数据：timestamp x y z qx qy qz qw
            tum_line = f"{timestamp:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n"

            # 将数据放入队列，由写入线程异步处理
            if self.vive_queue:
                self.vive_queue.put(tum_line)
        except Exception as e:
            if self.running:
                print(f"错误: 处理Vive位姿时出错: {e}")

    def vive_world_callback(self, msg):
        """Vive世界坐标系回调函数：保存世界坐标系数据（用于相对变换计算）"""
        if self.recording_state != RecordingState.RECORDING:
            return
        try:
            timestamp = msg.header.stamp.to_sec()

            # 保存世界坐标系数据
            x = msg.pose.position.x
            y = msg.pose.position.y
            z = msg.pose.position.z
            qx = msg.pose.orientation.x
            qy = msg.pose.orientation.y
            qz = msg.pose.orientation.z
            qw = msg.pose.orientation.w

            # 保存世界坐标系数据：timestamp x y z qx qy qz qw
            tum_line = f"{timestamp:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n"

            # 将数据放入队列，由写入线程异步处理
            if self.vive_world_queue:
                self.vive_world_queue.put(tum_line)
        except Exception as e:
            if self.running:
                print(f"错误: 处理Vive世界坐标系位姿时出错: {e}")

    def clamp_callback(self, msg):
        if self.recording_state != RecordingState.RECORDING:
            return
        try:
            if isinstance(msg, self.rospy.AnyMsg):
                if self.clamp_msg_class is None:
                    type_str = msg._connection_header.get('type', '') if hasattr(msg, '_connection_header') else ''
                    if type_str:
                        self.clamp_msg_class = self.get_message_class(type_str)
                if self.clamp_msg_class is not None:
                    real_msg = self.clamp_msg_class()
                    real_msg.deserialize(msg._buff)
                else:
                    return
            else:
                real_msg = msg

            data_value = getattr(real_msg, 'data', None)
            if data_value is None:
                return

            ts_sec = None
            if hasattr(real_msg, 'header') and hasattr(real_msg.header, 'stamp'):
                if (real_msg.header.stamp.secs != 0) or (real_msg.header.stamp.nsecs != 0):
                    ts_sec = real_msg.header.stamp.to_sec()
            if ts_sec is None:
                ts_sec = self.rospy.get_time()

            tum_line = f"{ts_sec:.9f} {data_value}\n"
            # 将数据放入队列，由写入线程异步处理
            self.clamp_queue.put(tum_line)
        except Exception as e:
            if self.running:
                print(f"错误: 处理Clamp数据时出错: {e}")

    def tof_callback(self, msg):
        """ToF点云回调 - 将数据放入队列，由写入线程异步处理"""
        if not self.enable_tof or self.recording_state != RecordingState.RECORDING:
            return
        if self.tof_queue is None:
            return
        
        try:
            # 获取时间戳
            tof_timestamp = msg.header.stamp.to_sec()
            
            # 将数据放入队列，由写入线程异步处理
            self.tof_queue.put((msg, tof_timestamp))
        except Exception as e:
            self.tof_dropped_count += 1
            if self.running and (self.tof_dropped_count % 10 == 1):
                print(f"警告: ToF队列放入失败 (已丢失{self.tof_dropped_count}帧): {e}")

    def calculate_rgb_fps(self, timestamp):
        now = time.time()
        self.rgb_timestamps.append(now)
        if now - self.rgb_last_fps_calc_time >= 1.0:
            if len(self.rgb_timestamps) >= 2:
                dt = now - self.rgb_timestamps[0]
                if dt > 0:
                    self.rgb_current_fps = (len(self.rgb_timestamps)-1) / dt
                    self.rgb_fps_history.append(self.rgb_current_fps)
                    if len(self.rgb_fps_history) > 0:
                        self.rgb_avg_fps = sum(self.rgb_fps_history) / len(self.rgb_fps_history)
            self.rgb_last_fps_calc_time = now

    def calculate_slam_fps(self):
        current_time = time.time()
        self.slam_timestamps.append(current_time)
        if current_time - self.slam_last_fps_calc_time >= 1.0:
            if len(self.slam_timestamps) >= 2:
                time_diff = current_time - self.slam_timestamps[0]
                if time_diff > 0:
                    self.slam_current_fps = (len(self.slam_timestamps) - 1) / time_diff
            self.slam_last_fps_calc_time = current_time

    def calculate_vive_fps(self):
        current_time = time.time()
        self.vive_timestamps.append(current_time)
        if current_time - self.vive_last_fps_calc_time >= 1.0:
            if len(self.vive_timestamps) >= 2:
                time_diff = current_time - self.vive_timestamps[0]
                if time_diff > 0:
                    self.vive_current_fps = (len(self.vive_timestamps) - 1) / time_diff
            self.vive_last_fps_calc_time = current_time

    def calculate_tof_fps(self):
        current_time = time.time()
        self.tof_timestamps.append(current_time)
        if current_time - self.tof_last_fps_calc_time >= 1.0:
            if len(self.tof_timestamps) >= 2:
                time_diff = current_time - self.tof_timestamps[0]
                if time_diff > 0:
                    self.tof_current_fps = (len(self.tof_timestamps) - 1) / time_diff
            self.tof_last_fps_calc_time = current_time

    def try_calculate_vive_offset(self):
        if not self.enable_vive:
            return
        if self.slam_first_timestamp and self.vive_first_timestamp and not self.offset_ready:
            self.vive_time_offset = self.slam_first_timestamp - self.vive_first_timestamp
            self.offset_ready = True
            print(f"\n✓ Vive 时间戳对齐已启用")
            self.process_pending_vive_data()

    def save_vive_offset_info(self):
        """保存 Vive 时间戳对齐信息"""
        if not self.enable_vive or not self.offset_ready:
            return
        try:
            vive_dir = os.path.join(self.output_dir, "Vive_Poses")
            offset_info_path = os.path.join(vive_dir, "offset_info.txt")
            with open(offset_info_path, 'w') as f:
                f.write("Vive 时间戳对齐信息\n")
                f.write("=" * 60 + "\n\n")
                f.write("对齐参数:\n")
                f.write(f"  SLAM 第一帧时间戳: {self.slam_first_timestamp:.6f}s\n")
                f.write(f"  Vive 第一帧时间戳: {self.vive_first_timestamp:.6f}s\n")
                f.write("说明:\n")
                f.write("  - 所有保存的 Vive 数据已使用对齐后的时间戳\n")
                f.write("  - 时间戳已与 SLAM 数据对齐\n")
                f.write("  - 坐标已转换到 Gripper 坐标系\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if self.device_label:
                    f.write(f"设备标签: {self.device_label}\n")
                f.write(f"XV 序列号: {self.xv_serial}\n")
                f.write(f"Vive 序列号: {self.vive_serial}\n")
            print(f"    ✓ 时间戳对齐信息已保存: {offset_info_path}")
        except Exception as e:
            print(f"    警告: 保存 offset 信息失败: {e}")

    def process_pending_vive_data(self):
        """处理待处理的Vive数据（已废弃，现在使用离线处理）"""
        pass

    def process_slam_offline(self):
        """离线处理SLAM数据：坐标转换和速度计算"""
        print("  开始离线处理SLAM数据...")
        slam_dir = os.path.join(self.output_dir, "SLAM_Poses")
        slam_raw_path = os.path.join(slam_dir, "slam_raw.txt")
        slam_processed_path = os.path.join(slam_dir, "slam_processed.txt")
        
        if not os.path.exists(slam_raw_path):
            print("    警告: SLAM原始文件不存在，跳过处理")
            return False
        
        try:
            prev_pose = None
            prev_timestamp = None
            velocity_history = deque(maxlen=10)
            
            with open(slam_raw_path, 'r') as f_in, open(slam_processed_path, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) < 8:
                        continue
                    
                    timestamp = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    qx = float(parts[4])
                    qy = float(parts[5])
                    qz = float(parts[6])
                    qw = float(parts[7])
                    
                    # 原始数据（XV坐标系）
                    qpos_xv = [x, y, z, qx, qy, qz, qw]
                    
                    # 转换到Gripper坐标系
                    qpos_gripper = transform_slam_to_gripper(qpos_xv)
                    current_pose = qpos_gripper
                    
                    # 计算速度
                    linear_velocity = 0.0
                    angular_velocity = 0.0
                    if prev_pose is not None and prev_timestamp is not None:
                        dt = timestamp - prev_timestamp
                        if dt > 0:
                            linear_velocity = self.calculate_linear_velocity(current_pose, prev_pose, dt)
                            angular_velocity = self.calculate_angular_velocity(current_pose, prev_pose, dt)
                            velocity_history.append(linear_velocity)
                    
                    # 平滑速度
                    smoothed_velocity = self.smooth_velocity(velocity_history)
                    
                    # 写入处理后的数据：timestamp x y z qx qy qz qw linear_vel angular_vel
                    x, y, z, qx, qy, qz, qw = qpos_gripper
                    # tum_line = f"{timestamp:.9f} {x} {y} {z} {qx} {qy} {qz} {qw} {smoothed_velocity:.6f} {angular_velocity:.6f}\n"
                    tum_line = f"{timestamp:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n"
                    f_out.write(tum_line)
                    
                    prev_pose = current_pose
                    prev_timestamp = timestamp
            
            print(f"    ✓ SLAM数据处理完成: {slam_processed_path}")
            return True
        except Exception as e:
            print(f"    错误: 处理SLAM数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_vive_offline(self):
        """离线处理Vive数据：坐标转换、时间对齐和速度计算"""
        if not self.enable_vive:
            print("  Vive未启用，跳过Vive数据处理")
            return True
        
        print("  开始离线处理Vive数据...")
        vive_dir = os.path.join(self.output_dir, "Vive_Poses")
        vive_app_path = os.path.join(vive_dir, "vive_app.txt")  # 应用坐标系文件
        vive_processed_path = os.path.join(vive_dir, "vive_data_tum.txt")  # 转换到Gripper后的文件
        
        if not os.path.exists(vive_app_path):
            print("    警告: Vive应用坐标系文件不存在，跳过处理")
            return False
        
        # 计算时间偏移（如果还没有计算）
        if not self.offset_ready and self.slam_first_timestamp is not None and self.vive_first_timestamp is not None:
            self.vive_time_offset = self.slam_first_timestamp - self.vive_first_timestamp
            self.offset_ready = True
            print(f"    计算Vive时间偏移: {self.vive_time_offset:.6f}s")
        
        try:
            prev_pose = None
            prev_timestamp = None
            velocity_history = deque(maxlen=10)
            
            with open(vive_app_path, 'r') as f_in, open(vive_processed_path, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) < 8:
                        continue
                    
                    timestamp = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    qx = float(parts[4])
                    qy = float(parts[5])
                    qz = float(parts[6])
                    qw = float(parts[7])
                    
                    # 应用坐标系数据（已归零）
                    qpos_app = [x, y, z, qx, qy, qz, qw]
                    
                    # 时间对齐
                    aligned_timestamp = timestamp
                    if self.offset_ready:
                        aligned_timestamp = timestamp + self.vive_time_offset
                    
                    # 转换到Gripper坐标系
                    qpos_gripper = transform_vive_to_gripper(qpos_app)
                    current_pose = qpos_gripper
                    
                    # 计算速度
                    linear_velocity = 0.0
                    angular_velocity = 0.0
                    if prev_pose is not None and prev_timestamp is not None:
                        dt = aligned_timestamp - prev_timestamp
                        if dt > 0:
                            linear_velocity = self.calculate_linear_velocity(current_pose, prev_pose, dt)
                            angular_velocity = self.calculate_angular_velocity(current_pose, prev_pose, dt)
                            velocity_history.append(linear_velocity)
                    
                    # 平滑速度
                    smoothed_velocity = self.smooth_velocity(velocity_history)
                    
                    # 写入处理后的数据：timestamp x y z qx qy qz qw
                    x, y, z, qx, qy, qz, qw = qpos_gripper
                    tum_line = f"{aligned_timestamp:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n"
                    f_out.write(tum_line)
                    
                    prev_pose = current_pose
                    prev_timestamp = aligned_timestamp
            
            print(f"    ✓ Vive数据处理完成: {vive_processed_path}")
            return True
        except Exception as e:
            print(f"    错误: 处理Vive数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def compute_relative_transform_offline(self):
        """
        离线计算左右设备的相对变换
        
        两种模式：
        1. Vive模式：从vive_raw.txt读取双臂位姿，计算相对变换
        2. 固定位置模式：无Vive时，假设双臂起始位置固定，基于SLAM计算相对运动
        """
        if not self.compute_relative_transform:
            return
        
        if not self.peer_output_dir:
            print("  ⚠ 跳过相对变换计算：需要双设备模式")
            print("    相对变换计算需要左右两个设备的数据")
            print("    请使用配置文件模式启动双设备录制")
            return
        
        print("  开始计算相对变换...")
        
        # 判断是左手还是右手（根据label）
        is_left = 'left' in self.device_label.lower()
        
        if is_left:
            left_dir = self.output_dir
            right_dir = self.peer_output_dir
        else:
            left_dir = self.peer_output_dir
            right_dir = self.output_dir
        
        # 输出文件路径
        if is_left:
            output_base = os.path.join(os.path.dirname(self.output_dir), "relative_transforms")
        else:
            # 右手设备不重复计算
            return
        
        try:
            if self.enable_vive:
                # Vive模式：从vive_raw读取真实位姿
                self._compute_relative_transform_vive(left_dir, right_dir, output_base)
            else:
                # 固定位置模式：基于SLAM + 固定偏移
                self._compute_relative_transform_fixed(left_dir, right_dir, output_base)
        except Exception as e:
            print(f"    警告: 相对变换计算失败: {e}")
    
    def _load_vive_raw(self, vive_raw_path):
        """加载vive_raw.txt文件"""
        data = []
        with open(vive_raw_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    timestamp = float(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    data.append([timestamp, x, y, z, qx, qy, qz, qw])
        data = np.array(data)
        return data[:, 0], data[:, 1:4], data[:, 4:8]
    
    def _align_timestamps(self, ts_left, ts_right, threshold=0.01):
        """对齐左右设备时间戳"""
        indices_left = []
        indices_right = []
        j = 0
        for i, t_left in enumerate(ts_left):
            while j < len(ts_right) - 1 and abs(ts_right[j + 1] - t_left) < abs(ts_right[j] - t_left):
                j += 1
            if abs(ts_right[j] - t_left) <= threshold:
                indices_left.append(i)
                indices_right.append(j)
        return np.array(indices_left), np.array(indices_right)
    
    def _compute_relative_transform_vive(self, left_dir, right_dir, output_base):
        """Vive模式：从vive_raw.txt（世界坐标系）计算相对变换"""
        left_vive_path = os.path.join(left_dir, "Vive_Poses", "vive_raw.txt")
        right_vive_path = os.path.join(right_dir, "Vive_Poses", "vive_raw.txt")
        
        if not os.path.exists(left_vive_path) or not os.path.exists(right_vive_path):
            print(f"    警告: Vive世界坐标系数据不完整，跳过相对变换计算")
            return
        
        print(f"    加载Vive世界坐标系数据...")
        ts_left, pos_left, quat_left = self._load_vive_raw(left_vive_path)
        ts_right, pos_right, quat_right = self._load_vive_raw(right_vive_path)
        
        print(f"    对齐时间戳...")
        idx_left, idx_right = self._align_timestamps(ts_left, ts_right)
        
        if len(idx_left) == 0:
            print(f"    警告: 没有匹配的时间戳")
            return
        
        print(f"    计算相对变换 ({len(idx_left)} 帧)...")
        self._save_relative_transforms(
            ts_left[idx_left],
            pos_left[idx_left],
            quat_left[idx_left],
            pos_right[idx_right],
            quat_right[idx_right],
            output_base,
            mode="vive"
        )
    
    def _load_slam_processed(self, slam_path):
        """加载slam_processed.txt (TUM格式)"""
        data = []
        with open(slam_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 8:
                    timestamp = float(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    data.append([timestamp, x, y, z, qx, qy, qz, qw])
        data = np.array(data)
        return data[:, 0], data[:, 1:4], data[:, 4:8]
    
    def _compute_relative_transform_fixed(self, left_dir, right_dir, output_base):
        """固定位置模式：基于SLAM + 固定偏移计算相对变换"""
        left_slam_path = os.path.join(left_dir, "SLAM_Poses", "slam_processed.txt")
        right_slam_path = os.path.join(right_dir, "SLAM_Poses", "slam_processed.txt")
        
        if not os.path.exists(left_slam_path) or not os.path.exists(right_slam_path):
            print(f"    警告: SLAM数据不完整，跳过相对变换计算")
            return
        
        # 检查是否提供了固定偏移参数
        if self.fixed_relative_pos is None:
            print(f"    ❌ 相对变换计算失败: 无Vive模式下需要提供 --fixed-relative-pos 参数")
            print(f"    请使用格式: --fixed-relative-pos x y z")
            print(f"    例如: --fixed-relative-pos 0.3 0 0")
            print(f"    表示左手相对右手的固定位置偏移（单位：米）")
            return
        
        print(f"    加载SLAM数据...")
        ts_left, pos_left_slam, quat_left = self._load_slam_processed(left_slam_path)
        ts_right, pos_right_slam, quat_right = self._load_slam_processed(right_slam_path)
        
        # 使用固定偏移调整左手位置（以右手为基准）
        fixed_offset = np.array(self.fixed_relative_pos)
        
        print(f"    应用固定偏移: {fixed_offset}")
        # 左手SLAM位置 = 左手SLAM + 固定偏移（模拟左手在世界坐标系中的位置）
        pos_left_world = pos_left_slam + fixed_offset
        
        # 右手位置直接使用SLAM（假设右手起始点为原点）
        pos_right_world = pos_right_slam
        
        print(f"    对齐时间戳...")
        idx_left, idx_right = self._align_timestamps(ts_left, ts_right)
        
        if len(idx_left) == 0:
            print(f"    警告: 没有匹配的时间戳")
            return
        
        print(f"    计算相对变换 ({len(idx_left)} 帧)...")
        self._save_relative_transforms(
            ts_left[idx_left],
            pos_left_world[idx_left],
            quat_left[idx_left],
            pos_right_world[idx_right],
            quat_right[idx_right],
            output_base,
            mode="fixed"
        )
    
    def _save_relative_transforms(self, timestamps, pos_left, quat_left, pos_right, quat_right, output_base, mode):
        """保存相对变换数据（双向：left->right 和 right->left）"""
        n_frames = len(timestamps)
        
        # 计算左到右的变换
        translations_l2r = np.zeros((n_frames, 3))
        quaternions_l2r = np.zeros((n_frames, 4))
        
        # 计算右到左的变换（逆变换）
        translations_r2l = np.zeros((n_frames, 3))
        quaternions_r2l = np.zeros((n_frames, 4))
        
        for i in range(n_frames):
            # 左到右
            T_l2r = compute_relative_transform_matrices(
                pos_left[i], quat_left[i],
                pos_right[i], quat_right[i]
            )
            translations_l2r[i] = T_l2r[:3, 3]
            R_l2r = T_l2r[:3, :3]
            rot_l2r = Rotation.from_matrix(R_l2r)
            quaternions_l2r[i] = rot_l2r.as_quat()
            
            # 右到左（逆变换）
            T_r2l = np.linalg.inv(T_l2r)
            translations_r2l[i] = T_r2l[:3, 3]
            R_r2l = T_r2l[:3, :3]
            rot_r2l = Rotation.from_matrix(R_r2l)
            quaternions_r2l[i] = rot_r2l.as_quat()
        
        # 保存左到右的变换（仅TXT）
        txt_path_l2r = f"{output_base}_left_to_right.txt"
        with open(txt_path_l2r, 'w') as f:
            f.write(f"# 双臂相对变换数据 (模式: {mode})\n")
            f.write(f"# 左设备到右设备的相对变换 (T_left_to_right)\n")
            f.write(f"# 数据点数: {n_frames}\n")
            if mode == "fixed" and self.fixed_relative_pos:
                f.write(f"# 固定偏移: {self.fixed_relative_pos}\n")
            f.write("# 格式: timestamp tx ty tz qx qy qz qw\n")
            for i in range(n_frames):
                f.write(f"{timestamps[i]:.9f} ")
                f.write(f"{translations_l2r[i, 0]:.6f} {translations_l2r[i, 1]:.6f} {translations_l2r[i, 2]:.6f} ")
                f.write(f"{quaternions_l2r[i, 0]:.6f} {quaternions_l2r[i, 1]:.6f} {quaternions_l2r[i, 2]:.6f} {quaternions_l2r[i, 3]:.6f}\n")
        
        # 保存右到左的变换（仅TXT）
        txt_path_r2l = f"{output_base}_right_to_left.txt"
        with open(txt_path_r2l, 'w') as f:
            f.write(f"# 双臂相对变换数据 (模式: {mode})\n")
            f.write(f"# 右设备到左设备的相对变换 (T_right_to_left)\n")
            f.write(f"# 数据点数: {n_frames}\n")
            if mode == "fixed" and self.fixed_relative_pos:
                f.write(f"# 固定偏移（取反）: {[-x for x in self.fixed_relative_pos] if self.fixed_relative_pos else 'N/A'}\n")
            f.write("# 格式: timestamp tx ty tz qx qy qz qw\n")
            for i in range(n_frames):
                f.write(f"{timestamps[i]:.9f} ")
                f.write(f"{translations_r2l[i, 0]:.6f} {translations_r2l[i, 1]:.6f} {translations_r2l[i, 2]:.6f} ")
                f.write(f"{quaternions_r2l[i, 0]:.6f} {quaternions_r2l[i, 1]:.6f} {quaternions_r2l[i, 2]:.6f} {quaternions_r2l[i, 3]:.6f}\n")
        
        print(f"    ✓ 相对变换已保存（双向）:")
        print(f"      左→右: {txt_path_l2r}")
        print(f"      右→左: {txt_path_r2l}")
        print(f"    统计: {n_frames} 帧")
        print(f"      左→右平移范围 X[{translations_l2r[:, 0].min():.3f}, {translations_l2r[:, 0].max():.3f}]m")
        print(f"      右→左平移范围 X[{translations_r2l[:, 0].min():.3f}, {translations_r2l[:, 0].max():.3f}]m")

    def save_all_data(self):
        print("正在保存数据...")
        try:
            # 1) 已经在 stop_recording()->freeze_recording() 里切到 SAVING,回调不会处理新数据
            
            # 2) 等待所有队列自然排空 (关键)
            if self.rgb_writer_thread:
                self.rgb_queue.join()
            if self.slam_writer_thread:
                self.slam_queue.join()
            if self.enable_vive and self.vive_writer_thread:
                self.vive_queue.join()
            if self.enable_vive and self.vive_world_writer_thread:
                self.vive_world_queue.join()
            if self.enable_tof and self.tof_writer_thread:
                self.tof_queue.join()
            if self.clamp_writer_thread:
                self.clamp_queue.join()
            
            # 3) 队列排空后再发 sentinel,让所有写盘线程退出
            if self.rgb_writer_thread:
                try:
                    self.rgb_queue.put(None, timeout=1)
                except Exception:
                    pass
                self.rgb_writer_thread.join(timeout=10)
            if self.slam_writer_thread:
                try:
                    self.slam_queue.put(None, timeout=1)
                except Exception:
                    pass
                self.slam_writer_thread.join(timeout=10)
            if self.enable_vive and self.vive_writer_thread:
                try:
                    self.vive_queue.put(None, timeout=1)
                except Exception:
                    pass
                self.vive_writer_thread.join(timeout=10)
            if self.enable_vive and self.vive_world_writer_thread:
                try:
                    self.vive_world_queue.put(None, timeout=1)
                except Exception:
                    pass
                self.vive_world_writer_thread.join(timeout=10)
            if self.enable_tof and self.tof_writer_thread:
                try:
                    self.tof_queue.put(None, timeout=1)
                except Exception:
                    pass
                self.tof_writer_thread.join(timeout=10)
            if self.clamp_writer_thread:
                try:
                    self.clamp_queue.put(None, timeout=1)
                except Exception:
                    pass
                self.clamp_writer_thread.join(timeout=10)
            
            # 关闭ToF BAG文件
            if self.enable_tof and self.tof_bag:
                with self.tof_bag_lock:
                    self.tof_bag.close()
                    print(f"  ✓ ToF BAG文件已关闭，共保存 {self.tof_saved_count} 个点云")
                self.tof_bag = None

            # 强制把 Vive/SLAM/Clamp/Timestamps 的尾部刷到盘
            self.flush_all_stream_files()

            # 先关闭文件，保证磁盘上的文本完整可读
            self.cleanup_files()

            # 离线处理SLAM和Vive数据（坐标转换和速度计算）
            print("\n开始离线处理SLAM和Vive数据...")
            slam_success = self.process_slam_offline()
            vive_success = self.process_vive_offline() if self.enable_vive else True
            
            if not slam_success or not vive_success:
                print("  警告: 离线处理失败，可能影响后续融合")
            
            # 等待离线处理完成后，再进行融合
            print("\n开始融合轨迹...")
            self.save_merged_data()
            self.save_vive_offset_info()
            
            # 计算相对变换（双设备模式）
            self.compute_relative_transform_offline()

            # [修改] RGB 已经直接保存为 MP4，无需离线处理
            print("\nRGB MP4文件已保存，无需离线转码。")

            # ToF后处理：自动转换BAG为PCD
            if self.enable_tof:
                print("\n开始ToF点云后处理...")
                self.convert_bag_to_pcd()

            print("\n数据保存完成")
        except Exception as e:
            print(f"错误: 保存数据时出错: {e}")
            import traceback
            traceback.print_exc()


    def save_merged_data(self):
        """从已处理后的 TUM 文件进行流式双指针合并（内存O(1)）"""
        slam_dir = os.path.join(self.output_dir, "SLAM_Poses")
        merged_dir = os.path.join(self.output_dir, "Merged_Trajectory")
        os.makedirs(merged_dir, exist_ok=True)

        # 使用处理后的文件（包含速度信息）
        slam_path = os.path.join(slam_dir, "slam_processed.txt")
        merged_path = os.path.join(merged_dir, "merged_trajectory.txt")
        
        # 如果没有启用Vive，直接复制SLAM数据作为merged轨迹
        if not self.enable_vive:
            print("  Vive未启用，使用SLAM数据作为merged轨迹")
            import shutil
            if os.path.exists(slam_path):
                shutil.copy2(slam_path, merged_path)
                self.merged_count = self.slam_count
                print(f"    Merged轨迹保存完成（直接使用SLAM），共{self.merged_count}个位姿点")
                self.save_merge_stats()
            else:
                print("  警告: SLAM处理文件不存在")
            return
        
        # 有Vive时进行融合
        vive_dir = os.path.join(self.output_dir, "Vive_Poses")
        vive_path = os.path.join(vive_dir, "vive_data_tum.txt")
        
        if not (os.path.exists(vive_path) and os.path.exists(slam_path)):
            print("  警告: 无法生成融合轨迹（处理后的文件缺失）")
            print(f"    Vive文件: {vive_path} (存在: {os.path.exists(vive_path)})")
            print(f"    SLAM文件: {slam_path} (存在: {os.path.exists(slam_path)})")
            return

        # 复位统计
        self.merge_stats.update({
            'total_merged': 0, 'averaged': 0, 'use_slam': 0,
            'use_vive': 0, 'both_high': 0, 'skipped': 0, 'no_match': 0
        })

        time_thresh = self.time_match_threshold

        def parse_line(line):
            """解析TUM格式行，支持带速度信息的格式"""
            parts = line.strip().split()
            if len(parts) < 8:
                return None, None, None, None
            ts = float(parts[0])
            pose = [float(parts[i]) for i in range(1, 8)]
            # 如果包含速度信息（处理后的文件）
            linear_vel = float(parts[8]) if len(parts) > 8 else None
            angular_vel = float(parts[9]) if len(parts) > 9 else None
            return ts, pose, linear_vel, angular_vel

        with open(vive_path, 'r') as fv, open(slam_path, 'r') as fs, open(merged_path, 'w') as fm:
            from collections import deque as dq
            slam_win = dq(maxlen=3000)  # ~6s
            prev_slam_ts, prev_slam_pose = None, None
            prev_vive_ts, prev_vive_pose = None, None

            def slam_iter():
                for line in fs:
                    ts, pose, linear_vel, angular_vel = parse_line(line)
                    if ts is not None:
                        yield ts, pose, linear_vel, angular_vel

            s_iter = slam_iter()
            try:
                s_ts, s_pose, s_linear_vel, s_angular_vel = next(s_iter)
            except StopIteration:
                print("  警告: SLAM 文件为空")
                return

            for v_line in fv:
                v_ts, v_pose, v_linear_vel, v_angular_vel = parse_line(v_line)
                if v_ts is None:
                    continue

                # 推进 SLAM 到覆盖窗口
                while True:
                    slam_win.append((s_ts, s_pose, s_linear_vel, s_angular_vel))
                    try:
                        if s_ts >= v_ts + time_thresh:
                            break
                        s_ts, s_pose, s_linear_vel, s_angular_vel = next(s_iter)
                    except StopIteration:
                        break

                # 在窗口内找最近
                best = None; best_d = float('inf')
                for (ts, pose, lin_vel, ang_vel) in slam_win:
                    d = abs(ts - v_ts)
                    if d < best_d:
                        best, best_d = (ts, pose, lin_vel, ang_vel), d
                    if d < 0.002:
                        break

                if best is None or best_d > time_thresh:
                    self.merge_stats['no_match'] += 1
                    self.merge_stats['skipped'] += 1
                    prev_vive_ts, prev_vive_pose = v_ts, v_pose
                    continue

                s_ts_near, s_pose_near, s_linear_vel_near, s_angular_vel_near = best

                # 使用离线计算的速度（如果可用），否则回退到在线计算
                if v_linear_vel is not None:
                    vive_vel = v_linear_vel
                else:
                    # 回退到在线计算
                    def lin_vel(curr_ts, curr_pose, p_ts, p_pose):
                        if p_ts is None or curr_ts <= p_ts:
                            return 0.0
                        return float(np.linalg.norm(np.array(curr_pose[:3]) - np.array(p_pose[:3])) / (curr_ts - p_ts))
                    vive_vel = lin_vel(v_ts, v_pose, prev_vive_ts, prev_vive_pose)
                
                if s_linear_vel_near is not None:
                    slam_vel = s_linear_vel_near
                else:
                    # 回退到在线计算
                    def lin_vel(curr_ts, curr_pose, p_ts, p_pose):
                        if p_ts is None or curr_ts <= p_ts:
                            return 0.0
                        return float(np.linalg.norm(np.array(curr_pose[:3]) - np.array(p_pose[:3])) / (curr_ts - p_ts))
                    slam_vel = lin_vel(s_ts_near, s_pose_near, prev_slam_ts, prev_slam_pose)

                merged_pose, _, strategy = self.merge_poses(s_pose_near, v_pose, slam_vel, vive_vel)
                if merged_pose is not None:
                    x, y, z, qx, qy, qz, qw = merged_pose
                    fm.write(f"{v_ts:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
                    self.merge_stats['total_merged'] += 1
                else:
                    self.merge_stats['skipped'] += 1

                prev_vive_ts, prev_vive_pose = v_ts, v_pose
                prev_slam_ts, prev_slam_pose = s_ts_near, s_pose_near

            fm.flush()

        self.merged_count = self.merge_stats['total_merged']
        print(f"    Merged轨迹保存完成，共{self.merged_count}个位姿点")
        self.save_merge_stats()

    def convert_bag_to_pcd(self):
        """将ToF BAG文件转换为PCD格式（离线处理）"""
        tof_dir = os.path.join(self.output_dir, "ToF_PointClouds")
        bag_path = os.path.join(tof_dir, "pointclouds.bag")
        output_dir = os.path.join(tof_dir, "PointClouds")
        topic = "/camera/depth/color/points"
        
        if not os.path.exists(bag_path):
            print("  警告: BAG文件不存在，跳过PCD转换")
            return False
        
        print("\n" + "=" * 60)
        print("开始转换ToF BAG文件为PCD格式")
        print("=" * 60)
        print(f"输入: {bag_path}")
        print(f"Topic: {topic}")
        print(f"输出: {output_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 检查rosrun和pcl_ros是否可用
            rosrun_check = subprocess.run(
                ["which", "rosrun"],
                capture_output=True,
                text=True
            )
            
            if rosrun_check.returncode != 0:
                print("  ✗ 未找到rosrun命令，跳过自动转换")
                return False
            
            # 执行转换命令
            print("  执行: rosrun pcl_ros bag_to_pcd ...")
            cmd = ["rosrun", "pcl_ros", "bag_to_pcd", bag_path, topic, output_dir]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                # 统计生成的PCD文件
                pcd_files = [f for f in os.listdir(output_dir) if f.endswith('.pcd')]
                pcd_count = len(pcd_files)
                
                # 计算PCD文件总大小
                total_size_mb = sum(os.path.getsize(os.path.join(output_dir, f)) 
                                   for f in pcd_files) / (1024 * 1024)
                
                print(f"  ✓ 转换完成！生成 {pcd_count} 个PCD文件 ({total_size_mb:.1f} MB)")
                print(f"  输出目录: {output_dir}")
                print("=" * 60)
                return True
            else:
                print(f"  ✗ 转换失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("  ✗ 转换超时（5分钟），请手动执行转换")
            return False
        except FileNotFoundError:
            print("  ✗ 未找到pcl_ros包，请安装: sudo apt install ros-$ROS_DISTRO-pcl-ros")
            return False
        except Exception as e:
            print(f"  ✗ 转换过程出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_merge_stats(self):
        try:
            merged_dir = os.path.join(self.output_dir, "Merged_Trajectory")
            stats_path = os.path.join(merged_dir, "merge_stats.txt")
            with open(stats_path, 'w') as f:
                if not self.enable_vive:
                    f.write("轨迹统计信息（仅使用SLAM数据）\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("配置:\n")
                    f.write(f"  Vive: 未启用\n")
                    f.write(f"  数据源: SLAM (500Hz)\n\n")
                    f.write("数据统计:\n")
                    f.write(f"  SLAM位姿数: {self.slam_count}\n")
                    f.write(f"  Merged位姿数: {self.merged_count}\n")
                else:
                    f.write("轨迹融合统计信息（基于速度的动态融合 / 流式双指针）\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("融合配置:\n")
                    f.write(f"  速度阈值: {self.velocity_threshold} m/s\n")
                    f.write(f"  时间匹配阈值: {self.time_match_threshold} s\n")
                    f.write(f"  基准传感器: VIVE (100Hz)\n")
                    f.write(f"  辅助传感器: SLAM (500Hz)\n\n")
                    f.write("融合结果统计:\n")
                    for k in ['total_merged','averaged','use_slam','use_vive','both_high','skipped','no_match']:
                        f.write(f"  {k}: {self.merge_stats[k]}\n")
                    total = self.merge_stats['total_merged']
                    if total > 0:
                        f.write("\n策略占比:\n")
                        f.write(f"  均值融合: {self.merge_stats['averaged']/total*100:.1f}%\n")
                        f.write(f"  使用SLAM: {self.merge_stats['use_slam']/total*100:.1f}%\n")
                        f.write(f"  使用VIVE: {self.merge_stats['use_vive']/total*100:.1f}%\n")
                        if self.merge_stats['skipped'] > 0:
                            total_processed = self.merged_count + self.merge_stats['skipped']
                            f.write(f"  跳过率: {self.merge_stats['skipped']}/{total_processed} ({self.merge_stats['skipped']/total_processed*100:.1f}%)\n")
                f.write(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if self.device_label:
                    f.write(f"设备标签: {self.device_label}\n")
                f.write(f"XV 序列号: {self.xv_serial}\n")
                if self.enable_vive:
                    f.write(f"Vive 序列号: {self.vive_serial}\n")
                else:
                    f.write(f"Vive: 未启用\n")
            print(f"    ✓ 融合统计信息已保存: {stats_path}")
        except Exception as e:
            print(f"    警告: 保存融合统计信息失败: {e}")

    def _check_sox_available(self):
        """检查 sox (play 命令) 是否可用"""
        try:
            result = subprocess.run(['which', 'play'], 
                                  capture_output=True, 
                                  text=True,
                                  timeout=1)
            return result.returncode == 0
        except Exception:
            return False

    def _display_progress(self):
        """显示采集进度条"""
        if self.max_rgb_count is None or self.max_rgb_count <= 0:
            return
        
        current = self.rgb_count
        total = self.max_rgb_count
        percentage = min(100.0, (current / total) * 100.0)
        
        # 在70%、80%、90%、100%时异步播放提示音（仅在 sox 可用时）
        if self._sox_available:
            alert_percentages = [50, 70, 80, 90, 100]
            for alert_pct in alert_percentages:
                if percentage >= alert_pct and alert_pct not in self._progress_alerted:
                    # 100%时使用800Hz，其他使用400Hz
                    if alert_pct == 50:
                        subprocess.Popen(
                            ['play', '--no-show-progress', '--null', '--channels', '1', 
                             'synth', '0.5', 'sine', '800'],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    elif alert_pct == 100:
                        subprocess.Popen(
                            ['play', '--no-show-progress', '--null', '--channels', '1', 
                             'synth', '0.1', 'sine', '800'],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    else:
                        subprocess.Popen(
                            ['play', '--no-show-progress', '--null', '--channels', '1', 
                             'synth', '0.1', 'sine', '400'],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    
                    self._progress_alerted.add(alert_pct)
        
        # 计算进度条长度（50个字符）
        bar_length = 50
        filled_length = int(bar_length * current / total)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        
        # 使用 \r 实现同一行更新
        print(f'\r进度: [{bar}] {current}/{total} ({percentage:.1f}%)', end='', flush=True)
        
        # 如果完成，换行
        if current >= total:
            print()

    def display_summary(self):
        """显示录制摘要"""
        if self.recording_start_time:
            recording_duration = self.recording_end_time - self.recording_start_time
        else:
            recording_duration = 0
        print(f"\n录制摘要:")
        print(f"  时长: {recording_duration:.1f}秒")
        print(f"  RGB图像: {self.rgb_count} 帧")
        print(f"  SLAM位姿: {self.slam_count} 个")
        if self.enable_vive:
            print(f"  Vive位姿: {self.vive_count} 个")
        print(f"  Merged轨迹: {self.merged_count} 个")
        if self.enable_tof:
            saved_count = self.tof_saved_count
            if self.tof_dropped_count > 0:
                print(f"  ToF点云: {saved_count} 个已保存 (回调收到: {self.tof_count}, 丢弃: {self.tof_dropped_count})")
            else:
                print(f"  ToF点云: {saved_count} 个已保存")
        else:
            print(f"  ToF点云: 已禁用")
        print(f"  Clamp数据: {self.clamp_count} 条")
        if self.rgb_count > 0:
            print(f"  RGB平均频率: {self.rgb_avg_fps:.1f} FPS")
        if self.enable_vive and self.merged_count > 0:
            print(f"\n  融合策略统计:")
            print(f"    均值融合: {self.merge_stats['averaged']} ({self.merge_stats['averaged']/self.merged_count*100:.1f}%)")
            print(f"    使用SLAM: {self.merge_stats['use_slam']} ({self.merge_stats['use_slam']/self.merged_count*100:.1f}%)")
            print(f"    使用VIVE: {self.merge_stats['use_vive']} ({self.merge_stats['use_vive']/self.merged_count*100:.1f}%)")
            print(f"    都高速(已跳过): {self.merge_stats['both_high']}")
            if self.merge_stats['skipped'] > 0:
                total_processed = self.merged_count + self.merge_stats['skipped']
                print(f"    总跳过率: {self.merge_stats['skipped']}/{total_processed} ({self.merge_stats['skipped']/total_processed*100:.1f}%)")
        elif not self.enable_vive:
            print(f"\n  注意: 未启用Vive，Merged轨迹直接使用SLAM数据")
        print(f"\n数据已保存到: {self.output_dir}")


def run_single_device_process(device_config, device_output_dir, enable_tof, max_rgb_count,
                               ready_barrier, start_barrier, stop_event, global_start_time, process_index,
                               compute_relative_transform, peer_output_dir, fixed_relative_pos):
    """
    在独立进程中运行单个设备的数据采集器
    """
    try:
        # 在子进程中导入和初始化ROS节点（使用唯一名称）
        import rospy
        node_name = f"data_collector_device_{process_index}_{os.getpid()}"
        rospy.init_node(node_name, anonymous=True)
        
        print(f"[进程 {process_index}] ROS节点已初始化: {node_name}")
        print(f"[进程 {process_index}] 初始化设备 [{device_config.get('label', 'unknown')}]:")
        print(f"[进程 {process_index}]   XV 序列号: {device_config['xv_serial']}")
        print(f"[进程 {process_index}]   Vive 序列号: {device_config['vive_serial']}")
        print(f"[进程 {process_index}]   输出目录: {device_output_dir}")
        
        # 创建单设备采集器
        recorder = SingleDeviceRecorder(
            device_config=device_config,
            output_dir=device_output_dir,
            enable_tof=enable_tof,
            auto_start=True,  # 自动初始化订阅
            compute_relative_transform=compute_relative_transform,
            peer_output_dir=peer_output_dir,
            fixed_relative_pos=fixed_relative_pos,
            max_rgb_count=max_rgb_count
        )
        
        print(f"[进程 {process_index}] 设备采集器已创建，准备录制...")
        
        # 第一个barrier：所有进程准备就绪（添加超时和异常处理）
        # 在等待 barrier 之前，不创建目录，确保只有所有进程都验证通过后才创建
        print(f"[进程 {process_index}] 准备就绪，等待其他进程准备...")
        try:
            ready_barrier.wait(timeout=10)
        except Exception as e:
            print(f"[进程 {process_index}] ready_barrier 等待失败: {e}，主动退出")
            return
        
        # 所有进程都通过验证后，创建输出目录
        # 这样可以确保如果任何进程验证失败，其他进程也不会创建目录
        print(f"[进程 {process_index}] 所有进程验证通过，创建输出目录...")
        os.makedirs(recorder.output_dir, exist_ok=True)
        
        # 准备录制（需要目录已存在）
        recorder.prepare_recording()
        
        # 第二个barrier：等待主进程设置开始时间后，所有进程同时开始（添加超时和异常处理）
        print(f"[进程 {process_index}] 等待同步启动信号...")
        try:
            start_barrier.wait(timeout=10)
        except Exception as e:
            print(f"[进程 {process_index}] start_barrier 等待失败: {e}，主动退出")
            return
        
        # 使用全局开始时间（由主进程在start_barrier之前设置）
        current_global_start_time = global_start_time.value
        
        recorder.start_recording(current_global_start_time)
        print(f"[进程 {process_index}] 开始录制 (PID: {os.getpid()}, 开始时间: {current_global_start_time})")
        
        # 等待停止信号（可能是手动停止或自动达到预设帧数）
        while not stop_event.is_set() and not rospy.is_shutdown() and recorder.running:
            rospy.sleep(0.1)
            # 检查是否应该停止（达到预设帧数）
            if recorder.should_stop_recording.is_set():
                recorder.stop_recording()
                break
            # 如果录制状态变为FINISHED（可能是自动停止），也退出循环
            if recorder.recording_state == RecordingState.FINISHED:
                break
        
        # while 循环结束后
        if recorder.recording_state != RecordingState.FINISHED:
            print(f"[进程 {process_index}] 收到停止信号,开始停止录制...")
            recorder.stop_recording()  # 用 stop_recording 统一流程(内部会 freeze+save_all_data)
        else:
            print(f"[进程 {process_index}] 已完成自动保存,跳过二次save_all_data()")
        
        # 清理资源
        recorder.cleanup()
        
        print(f"[进程 {process_index}] 进程退出 (PID: {os.getpid()})")
        
    except KeyboardInterrupt:
        print(f"\n[进程 {process_index}] 收到中断信号...")
    except Exception as e:
        print(f"[进程 {process_index}] 发生错误: {e}")
        import traceback
        traceback.print_exc()


class MultiDeviceRecorder:
    """
    双臂同步录制管理器（多进程架构）
    
    同步机制：
    1. ready_barrier: 等待所有设备进程准备完成
    2. start_barrier: 同步启动所有设备，确保时间戳对齐
    3. stop_event: 广播停止信号
    4. global_start_time: 共享全局开始时间
    
    目标：双臂RGB时间戳对齐误差 < 10ms
    """

    def __init__(self, config, enable_tof=True, output_dir=None,
                 compute_relative_transform=False, fixed_relative_pos=None, max_rgb_count=None):
        self.config = config
        self.enable_tof = enable_tof
        self.output_dir = output_dir
        self.compute_relative_transform = compute_relative_transform
        self.fixed_relative_pos = fixed_relative_pos
        self.max_rgb_count = max_rgb_count
        self.processes = []  # 设备进程列表
        self.running = True
        self.device_output_dirs = {}  # 各设备输出目录映射 {label: dir}
        self.session_root = None  # 会话根目录
        
        # 多进程同步原语
        self.ready_barrier = None  # Barrier: 准备就绪同步
        self.start_barrier = None  # Barrier: 启动同步
        self.stop_event = None     # Event: 停止信号
        self.global_start_time = None
        
        # 检查 sox (play 命令) 是否可用
        self._sox_available = self._check_sox_available()

        signal.signal(signal.SIGINT, self.signal_handler)

        print("=" * 60)
        print("摄像头 双设备同步数据采集器（多进程模式）")
        print("=" * 60)

        self.prepare_processes()

        print("=" * 60)
        print("所有设备进程已就绪")
        print("=" * 60)

    def prepare_processes(self):
        """准备多进程"""
        devices = self.config['devices']
        device_count = len(devices)

        # 确定输出目录
        if self.output_dir is not None:
            session_root = self.output_dir
            os.makedirs(session_root, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_root = os.path.join(".", f"session_{timestamp}")
            os.makedirs(session_root, exist_ok=True)
        
        self.session_root = session_root

        # 创建多进程同步对象
        self.ready_barrier = Barrier(device_count)  # 准备就绪同步
        self.start_barrier = Barrier(device_count + 1)  # 启动同步（+1是因为主进程也参与）
        self.stop_event = Event()
        self.global_start_time = Value('d', 0.0)  # 共享的双精度浮点数
        
        # 第一遍遍历：确定所有设备输出目录路径（但不创建，等验证通过后再创建）
        for device_key in sorted(devices.keys()):
            device_config = devices[device_key]
            label = device_config.get('label', device_key)
            xv_serial = device_config['xv_serial']
            device_name = f"{label}_{xv_serial}" if label else f"{xv_serial}"
            device_output_dir = os.path.join(session_root, device_name)
            # 不在这里创建目录，等验证通过后由子进程自己创建
            self.device_output_dirs[label] = device_output_dir

        # 第二遍遍历：为每个设备创建进程
        for idx, device_key in enumerate(sorted(devices.keys())):
            device_config = devices[device_key]
            label = device_config.get('label', device_key)
            xv_serial = device_config['xv_serial']
            device_output_dir = self.device_output_dirs[label]
            
            # 确定对方设备的输出目录（用于相对变换计算）
            peer_output_dir = None
            if 'left' in label.lower():
                # 左手设备，查找右手
                for other_label in self.device_output_dirs.keys():
                    if 'right' in other_label.lower():
                        peer_output_dir = self.device_output_dirs[other_label]
                        break
            elif 'right' in label.lower():
                # 右手设备，查找左手
                for other_label in self.device_output_dirs.keys():
                    if 'left' in other_label.lower():
                        peer_output_dir = self.device_output_dirs[other_label]
                        break

            print(f"\n准备设备进程 [{label}]:")
            print(f"  XV 序列号: {xv_serial}")
            print(f"  Vive 序列号: {device_config['vive_serial']}")
            print(f"  输出目录: {device_output_dir}")
            print(f"  进程索引: {idx}")

            # 创建进程
            process = Process(
                target=run_single_device_process,
                args=(
                    device_config,
                    device_output_dir,
                    self.enable_tof,
                    self.max_rgb_count,
                    self.ready_barrier,
                    self.start_barrier,
                    self.stop_event,
                    self.global_start_time,
                    idx,
                    self.compute_relative_transform,
                    peer_output_dir,
                    self.fixed_relative_pos
                ),
                name=f"DeviceRecorder-{label}"
            )
            
            self.processes.append(process)

    def _check_sox_available(self):
        """检查 sox (play 命令) 是否可用"""
        try:
            result = subprocess.run(['which', 'play'], 
                                  capture_output=True, 
                                  text=True,
                                  timeout=1)
            return result.returncode == 0
        except Exception:
            return False

    def start_all_recording(self):
        """启动所有设备的录制进程"""
        print("\n" + "=" * 60)
        print("启动所有设备录制进程...")
        print("=" * 60)

        # 启动所有进程
        for idx, process in enumerate(self.processes):
            print(f"启动设备进程 {idx + 1}/{len(self.processes)}...")
            process.start()

        # 等待所有进程启动并初始化
        print("等待所有进程初始化ROS节点...")
        time.sleep(3)  # 给进程一些时间初始化ROS节点和订阅者

        # 等待所有进程准备就绪（ready_barrier）
        # 此时所有进程都已完成prepare_recording并在ready_barrier处等待
        print("等待所有进程准备就绪...")
        # ready_barrier只在子进程之间，主进程不参与，所以需要等待子进程完成
        
        # 等待所有进程通过ready_barrier（它们会自己同步）
        time.sleep(1)
        
        # 在开始倒计时之前，先检查所有进程是否还活着
        # 如果任何进程在初始化验证时失败退出，应该立即停止
        alive_processes = [p for p in self.processes if p.is_alive()]
        alive_count = len(alive_processes)
        
        if alive_count < len(self.processes):
            print(f"\n\033[91m错误: 只有 {alive_count}/{len(self.processes)} 个设备进程成功通过验证！\033[0m")
            print("失败的进程:")
            for idx, process in enumerate(self.processes):
                if not process.is_alive():
                    print(f"  \033[91m设备 {idx + 1} 进程已退出 (PID: {process.pid}) - 可能是初始状态验证失败\033[0m")
            print("\n\033[91m采集已取消，不会开始倒计时。\033[0m")
            print("请检查设备初始状态（夹爪数值应在85-89范围内，SLAM/VIVE位置应在原点1cm内）")
            print("\n触发停止所有进程...")
            self.stop_all_recording()
            return
        
        # 在所有进程准备好后，进行倒计时提示，然后设置全局开始时间
        # 倒计时不影响录制内容，因为所有进程都在barrier处等待
        print("\n准备开始录制，倒计时：")
        for i in range(3, 0, -1):
            if self._sox_available:
                os.system('play --no-show-progress --null --channels 1 synth %s sine %f > /dev/null 2>&1' % (0.1, 400))
            print(f"  {i}...")
            time.sleep(1)
        if self._sox_available:
            os.system('play --no-show-progress --null --channels 1 synth %s sine %f > /dev/null 2>&1' % (0.1, 800))
        print("  开始！\n")
        
        # 设置全局开始时间（在倒计时结束后立即设置）
        # 这样确保所有进程使用完全相同的时间戳
        self.global_start_time.value = time.time()
        print(f"全局开始时间已设置: {self.global_start_time.value}")

        # 通过start_barrier触发所有进程同时开始录制
        # start_barrier包含主进程（+1），所以主进程也需要wait
        print("触发同步启动...")
        try:
            self.start_barrier.wait(timeout=10)  # 等待所有进程（包括主进程）到达
        except Exception as e:
            print(f"警告: barrier等待超时或出错: {e}")

        # 检查实际存活的进程数量
        alive_processes = [p for p in self.processes if p.is_alive()]
        alive_count = len(alive_processes)
        
        if alive_count < len(self.processes):
            print(f"\n错误: 只有 {alive_count}/{len(self.processes)} 个设备进程成功启动！")
            print("失败的进程:")
            for idx, process in enumerate(self.processes):
                if not process.is_alive():
                    print(f"  设备 {idx + 1} 进程已退出 (PID: {process.pid})")
            print("\n触发停止所有进程...")
            self.stop_all_recording()
            return
        
        print(f"\n✓ {alive_count} 个设备进程已启动并开始录制")
        for idx, process in enumerate(self.processes):
            if process.is_alive():
                print(f"  设备 {idx + 1} 进程运行中 (PID: {process.pid})")
        if self.max_rgb_count is not None:
            print(f"目标RGB帧数: {self.max_rgb_count}，达到后自动停止")
        else:
            print("按回车键停止录制...")
        print("=" * 60)

    def stop_all_recording(self):
        """停止所有设备的录制"""
        print("\n" + "=" * 60)
        print("停止所有设备录制...")
        print("=" * 60)

        tail_grace_s = 0.4
        time.sleep(tail_grace_s)

        # 发送停止信号
        print("发送停止信号到所有进程...")
        self.stop_event.set()

        # 等待所有进程完成
        print("等待所有进程完成...")
        for idx, process in enumerate(self.processes):
            if process.is_alive():
                print(f"等待设备进程 {idx + 1} 完成...")
                process.join(timeout=80)  # 最多等待80秒
                if process.is_alive():
                    print(f"警告: 设备进程 {idx + 1} 未在80秒内完成，强制终止...")
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()

        print("\n✓ 所有设备录制完成")
        print("=" * 60)

    def display_summary(self):
        """显示录制摘要（从输出目录读取）"""
        print("\n" + "=" * 60)
        print("录制摘要")
        print("=" * 60)
        print("注意: 详细摘要请查看各设备输出目录中的日志")
        for idx, device_dir in enumerate(self.device_output_dirs):
            print(f"\n设备 {idx + 1} 输出目录: {device_dir}")

    def signal_handler(self, signum, frame):
        """信号处理"""
        print("\n收到中断信号...")
        self.running = False
        if self.stop_event:
            self.stop_event.set()

    def run(self):
        """运行多进程录制"""
        try:
            if self.max_rgb_count is None:
                print("错误: 未设置最大RGB帧数，无法启动录制")
                print("请使用 --max-rgb 参数指定目标帧数")
                return
            
            if self.max_rgb_count is not None:
                print(f"\n按回车键开始录制所有设备（目标: {self.max_rgb_count} 帧）...")
            else:
                print("\n按回车键开始录制所有设备...")
            input()
            self.start_all_recording()
            
            # 等待所有进程完成（通过检查进程是否存活）
            if self.max_rgb_count is not None:
                # 如果设置了最大帧数，每个子进程中的SingleDeviceRecorder会自动检测并停止
                # 主进程只需要等待所有子进程完成即可，不需要再调用stop_all_recording()
                print("等待所有设备达到目标帧数...")
                while any(p.is_alive() for p in self.processes):
                    time.sleep(0.5)
                # 所有进程已自动停止并完成，直接显示摘要
                print("\n所有设备已达到目标帧数并自动停止")
                self.display_summary()
            else:
                # 未设置最大帧数时，需要手动停止
                input()
                self.stop_all_recording()
                self.display_summary()
        except KeyboardInterrupt:
            print("\n收到中断信号...")
            self.stop_event.set()
        finally:
            self.cleanup()

    def cleanup(self):
        """清理所有进程资源"""
        print("\n正在清理所有设备进程...")
        
        # 确保停止事件已设置
        if self.stop_event:
            self.stop_event.set()
        
        # 等待所有进程退出
        for idx, process in enumerate(self.processes):
            if process.is_alive():
                print(f"等待设备进程 {idx + 1} 退出...")
                process.join(timeout=5)
                if process.is_alive():
                    print(f"强制终止设备进程 {idx + 1}...")
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
                        process.join()
        
        print("✓ 所有进程清理完成")


def load_config():
    config_paths = [
        "../start_process/config.json",
        "./start_process/config.json",
        "../config.json"
    ]
    config_file = None
    for path in config_paths:
        if os.path.exists(path):
            config_file = path
            break
    if config_file is None:
        print("错误: 找不到配置文件")
        print("\n请先运行以下命令之一：")
        print("  1. 启动完整系统（推荐）:")
        print("      cd start_process && ./unified_launcher.sh")
        print("  2. 仅运行设备配对:")
        print("      cd start_process && python3 device_pairing.py")
        sys.exit(1)
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        if 'single_device' not in config:
            print(f"错误: 配置文件缺少 'single_device' 字段: {config_file}")
            sys.exit(1)
        if 'devices' not in config:
            print(f"错误: 配置文件缺少 'devices' 字段: {config_file}")
            sys.exit(1)
        if 'device_0' not in config['devices']:
            print(f"错误: 配置文件缺少 'device_0': {config_file}")
            sys.exit(1)
        if not config['single_device'] and 'device_1' not in config['devices']:
            print(f"错误: 双设备模式但缺少 'device_1': {config_file}")
            sys.exit(1)
        print(f"✓ 成功加载配置: {config_file}")
        return config
    except json.JSONDecodeError as e:
        print(f"错误: 配置文件 JSON 格式错误: {e}")
        print(f"配置文件: {config_file}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取配置文件失败: {e}")
        print(f"配置文件: {config_file}")
        sys.exit(1)

def find_device_by_xv_serial(config, xv_serial):
    devices = config.get('devices', {})
    for _, device_config in devices.items():
        if device_config.get('xv_serial') == xv_serial:
            return device_config
    return None

def main():
    """
    主入口函数
    
    运行模式：
    1. 单设备模式：--device <serial>
    2. 配置文件模式：从config.json读取设备配置（支持单/双设备）F
    
    性能选项：
    --tof on: 开启ToF采集（会增加磁盘负载，可能影响RGB频率）
    """
    
    parser = argparse.ArgumentParser(description='FastUMI 数据采集器 - 双臂同步版本 (直接MP4)')
    parser.add_argument('--device', '-d', help='指定设备序列号（强制单设备模式）')
    parser.add_argument('--output', '-o', help='输出目录（单/双设备均可用，双设备时为会话根目录）')
    parser.add_argument('--tof', choices=['on', 'off'], default='on', help='是否采集ToF点云，默认off')
    parser.add_argument('--compute-relative-transform', choices=['on', 'off'], default='on', 
                        help='是否计算双臂相对变换（默认off，仅双设备模式有效）')
    parser.add_argument('--fixed-relative-pos', type=float, nargs=3, metavar=('X', 'Y', 'Z'), 
                        help='无Vive时左手相对右手的固定位置 [x y z]（米），例如: 0.3 0 0')
    parser.add_argument('--max-rgb', type=int, default=420,
                        help='最大RGB帧数，达到后自动停止（默认不限制）')
    args = parser.parse_args()

    enable_tof = (args.tof == 'on')
    compute_relative_transform = (args.compute_relative_transform == 'on')
    fixed_relative_pos = args.fixed_relative_pos
    max_rgb_count = args.max_rgb

    config = load_config()

    if args.device:
        print(f"单设备模式：{args.device}")
        device_config = find_device_by_xv_serial(config, args.device)
        if not device_config:
            print(f"警告: 配置中未找到设备 {args.device}，使用降级模式（无 Vive 配对信息）")
            device_config = {'xv_serial': args.device, 'vive_serial': 'UNKNOWN', 'label': ''}
        
        # 单设备模式：数据文件直接放在session目录下，不需要序列号子目录
        output_dir = args.output
        
        recorder = SingleDeviceRecorder(
            device_config=device_config,
            output_dir=output_dir,
            enable_tof=enable_tof,
            auto_start=False,
            max_rgb_count=max_rgb_count
        )
        recorder.run()
    else:
        if config['single_device']:
            print("单设备模式（根据配置文件）")
            device_config = config['devices']['device_0']
            
            # 单设备模式：数据文件直接放在session目录下，不需要序列号子目录
            output_dir = args.output
            
            recorder = SingleDeviceRecorder(
                device_config=device_config,
                output_dir=output_dir,
                enable_tof=enable_tof,
                auto_start=False,
                max_rgb_count=max_rgb_count
            )
            recorder.run()
        else:
            print("双设备同步模式（根据配置文件）")
            recorder = MultiDeviceRecorder(
                config=config,
                enable_tof=enable_tof,
                output_dir=args.output,
                compute_relative_transform=compute_relative_transform,
                fixed_relative_pos=fixed_relative_pos,
                max_rgb_count=max_rgb_count
            )
            recorder.run()


if __name__ == '__main__':
    # 设置多进程启动方法
    # Linux上使用fork更高效，Windows必须使用spawn
    try:
        if sys.platform == 'win32':
            multiprocessing.set_start_method('spawn', force=True)
        else:
            # Linux/Unix: 优先使用fork
            try:
                multiprocessing.set_start_method('fork', force=True)
            except (RuntimeError, ValueError):
                # 如果fork不可用，使用spawn
                multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过，则使用当前方法
        pass
    main()
