"""UMI Expert for teleoperation with UMI gripper and XArm."""
# 训练分类器时不需要 ROS，这里做延迟/可选导入，避免在无 ROS 环境下崩溃。
try:
    import rospy
    from xv_sdk.msg import Clamp, PoseStampedConfidence
except ImportError:
    rospy = None
    Clamp = PoseStampedConfidence = None

import threading
from collections import deque
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as R


class UMIExpert:
    """接收 UMI rostopic，维护 pose 队列和当前 clamp 值。"""
    
    def __init__(self, pose_queue_size):
        if rospy is None:
            raise ImportError("rospy 未安装：UMIExpert 需要 ROS 环境。训练分类器请确保未实例化 UMIExpert。")
        # 默认路径：相对于当前文件的 start_process/config.json
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "start_process", "config.json")
    
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 获取设备序列号
        devices = config.get("devices", {})
        device_0 = devices.get("device_0", {})
        xv_serial = device_0.get("xv_serial", "")
        
        if not xv_serial:
            raise ValueError("配置文件中未找到 xv_serial")
        
        # 构建 topic 名称
        self.pose_topic = f"/xv_sdk/{xv_serial}/slam/pose"
        self.clamp_topic = f"/xv_sdk/{xv_serial}/clamp/Data"
        
        # 数据存储
        self.pose_queue_size = pose_queue_size
        self.pose_queue = deque(maxlen=pose_queue_size)  # 自动限制大小的队列
        self.current_clamp = None
        
        # 线程同步
        self.pose_lock = threading.Lock()
        self.clamp_lock = threading.Lock()
        self.pose_subscriber = None
        self.clamp_subscriber = None
        self.shutdown_flag = False
        
        rospy.loginfo(f"UMIExpert 初始化完成:")
        rospy.loginfo(f"  Pose topic: {self.pose_topic}")
        rospy.loginfo(f"  Clamp topic: {self.clamp_topic}")
        rospy.loginfo(f"  Pose queue size: {self.pose_queue_size}")

    def transform_to_base_quat(self, x, y, z, qx, qy, qz, qw, T_base_to_umi):
        '''transform the pose of fastumi to robot base, returns quaternion'''
        T_base_to_umi = np.array(T_base_to_umi)
        
        rotation_umi = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T_umi = np.eye(4)
        T_umi[:3, :3] = rotation_umi
        T_umi[:3, 3] = [x, y, z]

        # 计算变换后的位姿：T_base = T_base_to_umi * T_umi  
        T_base = np.matmul(T_base_to_umi, T_umi)
        
        x_base, y_base, z_base = T_base[:3, 3]
        rotation_base = R.from_matrix(T_base[:3, :3])
        qx_base, qy_base, qz_base, qw_base = rotation_base.as_quat()
        return x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base
        
    def _pose_callback(self, msg):
        """回调函数：处理接收到的 pose 消息并存入队列。"""
        try:
            pose_msg = msg.poseMsg
            T_base2umi = np.array([[ 0.    ,  0.        ,  1.      ,  0.  ],
                                     [ -1.    ,  0.       ,  0.      ,  0.   ],
                                     [0.     ,  -1.        ,  0.      ,  0.  ],
                                     [ 0.    ,  0.        ,  0.      ,  1.        ]])
            pose_base = self.transform_to_base_quat(
                pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z, 
                pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, 
                pose_msg.pose.orientation.w, T_base2umi)
            pose_data = {
                'position': {
                    'x': pose_base[0],
                    'y': pose_base[1],
                    'z': pose_base[2]
                },
                'orientation': {
                    'x': pose_base[3],
                    'y': pose_base[4],
                    'z': pose_base[5],
                    'w': pose_base[6]
                },
                'confidence': msg.confidence,
                'timestamp': pose_msg.header.stamp.to_sec()
            }
            # 存入队列
            with self.pose_lock:
                self.pose_queue.append(pose_data)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Pose callback error: {e}")
    
    def _clamp_callback(self, msg):
        """回调函数：处理接收到的 clamp 消息并更新当前值。"""
        try:
            with self.clamp_lock:
                self.current_clamp = msg.data
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Clamp callback error: {e}")
    
    def get_pose_queue(self):
        """
        Returns:
            list: pose 数据列表（从旧到新）
        """
        with self.pose_lock:
            return list(self.pose_queue)
    
    def get_latest_pose(self):
        with self.pose_lock:
            if len(self.pose_queue) > 0:
                return self.pose_queue[-1]
            return None
    
    def get_current_clamp(self):
        """
        获取当前 clamp 值。
        
        Returns:
            当前 clamp 数据，如果未收到则返回 None
        """
        with self.clamp_lock:
            return self.current_clamp
    
    def get_pose_delta(self):
        """
        计算 pose 队列中最新的元素减去最老的元素的差值。
        
        Returns:
            dict: 包含以下字段的字典：
                - 'position': dict，包含 'x', 'y', 'z' 位置差值
                - 'rotation': dict，包含 'axis' (旋转轴，单位向量) 和 'angle' (旋转角度，弧度)
                - 'rotation_vec': numpy array，旋转向量 (axis * angle)，便于限幅
                如果队列中元素少于2个，返回 None
        """
        with self.pose_lock:
            if len(self.pose_queue) < 2:
                return None
            
            oldest_pose = self.pose_queue[0]
            latest_pose = self.pose_queue[-1]
        
        # 计算位置差：最新 - 最老
        pos_delta = {
            'x': latest_pose['position']['x'] - oldest_pose['position']['x'],
            'y': latest_pose['position']['y'] - oldest_pose['position']['y'],
            'z': latest_pose['position']['z'] - oldest_pose['position']['z']
        }
        
        # 计算旋转差：使用四元数计算相对旋转
        # 提取四元数 (x, y, z, w)
        q_oldest = np.array([
            oldest_pose['orientation']['x'],
            oldest_pose['orientation']['y'],
            oldest_pose['orientation']['z'],
            oldest_pose['orientation']['w']
        ])
        
        q_latest = np.array([
            latest_pose['orientation']['x'],
            latest_pose['orientation']['y'],
            latest_pose['orientation']['z'],
            latest_pose['orientation']['w']
        ])
        
        # 创建 Rotation 对象
        R_oldest = R.from_quat(q_oldest)
        R_latest = R.from_quat(q_latest)
        
        # 计算相对旋转：R_delta = R_latest * R_oldest^-1
        R_delta = R_latest * R_oldest.inv()
        
        # 转换为轴角表示
        # as_rotvec() 返回旋转向量 (axis * angle)，长度为角度（弧度），方向为旋转轴
        rotation_vec = R_delta.as_rotvec()
        angle = np.linalg.norm(rotation_vec)
        
        # 计算旋转轴（单位向量）
        if angle > 1e-6:  # 避免除以零
            axis = rotation_vec / angle
        else:
            axis = np.array([0.0, 0.0, 1.0])  # 默认轴（无旋转时）
        
        return {
            'position': pos_delta,
            'rotation': {
                'axis': axis.tolist(),  # 转换为列表便于序列化
                'angle': float(angle)   # 弧度
            },
            'rotation_vec': rotation_vec.tolist()  # 旋转向量 (axis * angle)，便于限幅
        }
    
    def start(self):
        """启动 ROS Subscribers。"""
        rospy.loginfo("Starting UMIExpert subscribers...")
        self.shutdown_flag = False
        
        # 创建 Subscribers（使用回调函数，不阻塞主线程）
        # queue_size=1 确保不缓冲旧消息，使用最新数据
        self.pose_subscriber = rospy.Subscriber(
            self.pose_topic,
            PoseStampedConfidence,
            self._pose_callback,
            queue_size=1
        )
        
        self.clamp_subscriber = rospy.Subscriber(
            self.clamp_topic,
            Clamp,
            self._clamp_callback,
            queue_size=1
        )
        
        rospy.loginfo("UMIExpert subscribers started.")
    
    def stop(self):
        """停止 ROS Subscribers。"""
        rospy.loginfo("Shutting down UMIExpert...")
        self.shutdown_flag = True
        
        # 取消订阅
        if self.pose_subscriber is not None:
            self.pose_subscriber.unregister()
            self.pose_subscriber = None
        if self.clamp_subscriber is not None:
            self.clamp_subscriber.unregister()
            self.clamp_subscriber = None
        
        rospy.loginfo("UMIExpert stopped.")
