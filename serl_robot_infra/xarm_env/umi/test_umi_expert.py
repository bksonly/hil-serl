#!/usr/bin/env python3
"""测试脚本：打印接收到的 UMI pose 和 clamp 信息。"""
import rospy
import sys
import os
import numpy as np

# 添加项目路径到 PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from serl_robot_infra.xarm_env.umi.umi_expert import UMIExpert


def main():
    """主函数：初始化 ROS 节点并打印接收到的数据。"""
    # 初始化 ROS 节点
    rospy.init_node('umi_expert_test', anonymous=True)
    
    # 创建 UMIExpert 实例
    # 可以通过参数指定 config 路径和队列大小
    pose_queue_size = 10  # 可以修改这个参数
    expert = UMIExpert(pose_queue_size=pose_queue_size)
    
    # 启动监听线程
    expert.start()
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("UMI Expert 测试脚本启动")
    rospy.loginfo("按 Ctrl+C 退出")
    rospy.loginfo("=" * 60)
    
    # 打印频率控制
    print_rate = rospy.Rate(2)  # 每 0.5 秒打印一次
    
    try:
        while not rospy.is_shutdown():
            # 获取最新 pose
            latest_pose = expert.get_latest_pose()
            
            # 获取 pose 队列
            pose_queue = expert.get_pose_queue()
            
            # 获取当前 clamp
            current_clamp = expert.get_current_clamp()
            
            # 获取 pose 差值
            pose_delta = expert.get_pose_delta()
            
            # 打印信息
            print("\n" + "=" * 60)
            print(f"队列中的 pose 数量: {len(pose_queue)}/{pose_queue_size}")
            
            if latest_pose is not None:
                print(f"\n最新 Pose:")
                print(f"  位置: x={latest_pose['position']['x']:.4f}, "
                      f"y={latest_pose['position']['y']:.4f}, "
                      f"z={latest_pose['position']['z']:.4f}")
                print(f"  四元数: x={latest_pose['orientation']['x']:.4f}, "
                      f"y={latest_pose['orientation']['y']:.4f}, "
                      f"z={latest_pose['orientation']['z']:.4f}, "
                      f"w={latest_pose['orientation']['w']:.4f}")
                if 'confidence' in latest_pose:
                    print(f"  置信度: {latest_pose['confidence']:.4f}")
                print(f"  时间戳: {latest_pose['timestamp']:.6f}")
            else:
                print("\n最新 Pose: 暂无数据")
            
            if current_clamp is not None:
                print(f"\n当前 Clamp: {current_clamp}")
            else:
                print("\n当前 Clamp: 暂无数据")
            
            # 打印 pose 差值
            if pose_delta is not None:
                print(f"\nPose 差值 (最新 - 最老):")
                print(f"  位置差: x={pose_delta['position']['x']:.4f}, "
                      f"y={pose_delta['position']['y']:.4f}, "
                      f"z={pose_delta['position']['z']:.4f}")
                print(f"  旋转角度: {pose_delta['rotation']['angle']:.4f} 弧度 "
                      f"({np.degrees(pose_delta['rotation']['angle']):.2f} 度)")
                print(f"  旋转轴: [{pose_delta['rotation']['axis'][0]:.4f}, "
                      f"{pose_delta['rotation']['axis'][1]:.4f}, "
                      f"{pose_delta['rotation']['axis'][2]:.4f}]")
                print(f"  旋转向量 (axis * angle): [{pose_delta['rotation_vec'][0]:.4f}, "
                      f"{pose_delta['rotation_vec'][1]:.4f}, "
                      f"{pose_delta['rotation_vec'][2]:.4f}]")
            else:
                print("\nPose 差值: 队列中元素少于2个，无法计算")
            
            # 打印队列中所有 pose 的简要信息
            if len(pose_queue) > 0:
                print(f"\n队列中的 Pose 列表 (共 {len(pose_queue)} 个):")
                for i, pose in enumerate(pose_queue):
                    print(f"  [{i}] pos=({pose['position']['x']:.3f}, "
                          f"{pose['position']['y']:.3f}, "
                          f"{pose['position']['z']:.3f}) "
                          f"t={pose['timestamp']:.3f}")
            
            print("=" * 60)
            
            print_rate.sleep()
            
    except KeyboardInterrupt:
        rospy.loginfo("\n收到中断信号，正在退出...")
    finally:
        expert.stop()
        rospy.loginfo("测试脚本退出")


if __name__ == "__main__":
    main()
