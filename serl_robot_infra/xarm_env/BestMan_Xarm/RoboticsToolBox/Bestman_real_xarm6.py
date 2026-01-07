# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import time
import datetime
import ikpy
import serial
import serial.tools.list_ports
from ikpy.chain import Chain
from ikpy.inverse_kinematics import inverse_kinematic_optimization
from scipy.spatial.transform import Rotation as R
import sys
import os
import minimalmodbus as mm
# 假设您的环境中使用的是 pyrobotiqgripper 包 (根据原文件 import 判断)
import pyRobotiqGripper 

from xarm.wrapper import XArmAPI
current_dir = os.path.dirname(os.path.abspath(__file__))


class Bestman_Real_Xarm6:
    # ----------------------------------------------------------------
    # Functions for initalization
    # ----------------------------------------------------------------

    def __init__(self, robot_ip, local_ip, frequency):
        # Initialize the robot and gripper with the provided IPs and frequency
        self.robot = XArmAPI(robot_ip)
        local_ip = None
        self.mode = self.robot.set_mode(0) # 0: default
        self.robot_states = self.robot.set_state(0)
        self.first_init_flag = True
        self.gripper = True # have gripper by default
        self.frequency = frequency
        
        # 新增: 用于存储 USB 直连的 gripper 对象
        self.usb_gripper = None

        # # URDF
        # urdf_file = os.path.join(current_dir, "../Asset/xarm6_robot.urdf")
        # self.robot_chain = Chain.from_urdf_file(urdf_file)
        # self.active_joints = [
        #      joint for joint in self.robot_chain.links 
        #      if isinstance(joint, ikpy.link.URDFLink) and (joint.joint_type == 'revolute' or joint.joint_type == 'prismatic')
        # ]

    # ----------------------------------------------------------------
    # Functions for basic control
    # ----------------------------------------------------------------
    def clear_fault(self):
        '''Clear fault, and updates the current robot states.'''
        self.robot.set_state(0)

    def update_robot_states(self):
        '''Updates the current robot states.'''
        self.robot.getRobotStates(self.robot_states)

    def set_mode(self, _mode):
        '''
        Parameters:
        _mode=
            0: position controlmode
            1: servo motionmode
            2: joint teachingmodemode (invalid)3: cartesian teaching
            4: joint velocity control mode
            5: cartesian velocity control mode
            6: joint online trajectory planningmode
            7: cartesian online trajectory planning

        Notes:
            https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/doc/api/xarm_api.md#mode
        '''
        print("set mode")
        self.robot.set_mode(_mode)

    def go_home(self,dist=0):
        '''Move arm to initial pose.'''

        self.robot.set_position(x=396.4+dist, y=-5.5,z=360,roll=-90,pitch=-90,yaw=-90,wait=False)
        time.sleep(3)

    def pose_to_euler(self, pose):
        '''
        Convert robot pose from a list [x, y, z, qw, qx, qy, qz] to [x, y, z] and Euler angles.
         
        Parameters:
        pose: list of 7 floats - [x, y, z, qw, qx, qy, qz]
         
        Returns:
        tuple: (x, y, z, roll, pitch, yaw) where (x, y, z) is the position and (roll, pitch, yaw) are the Euler angles in radians.
        '''
        x, y, z, qw, qx, qy, qz = pose
        r = R.from_quat([qx, qy, qz, qw])  # Reordering to match scipy's [qx, qy, qz, qw]
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        return [x, y, z, roll, pitch, yaw]

    def euler_to_pose(self, position_euler):
        '''
        Convert robot pose from [x, y, z, roll, pitch, yaw] to [x, y, z, qw, qx, qy, qz].
         
        Parameters:
        position_euler: list of 6 floats - [x, y, z, roll, pitch, yaw]
         
        Returns:
        list: [x, y, z, qw, qx, qy, qz]
        '''
        x, y, z, roll, pitch, yaw = position_euler
        r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        qx, qy, qz, qw = r.as_quat()  # Getting [qx, qy, qz, qw] from scipy
        return [x, y, z, qw, qx, qy, qz]  # Reordering to match [qw, qx, qy, qz]
     
    def set_payload(self,payload_weight, payload_cog):

        self.robot.set_tcp_load(payload_weight, payload_cog)
     

    # ----------------------------------------------------------------
    # Functions for robot parameters acquisition
    # ----------------------------------------------------------------

    def get_joint_bounds(self):
        '''
        Retrieves the joint bounds of the robot arm.

        Returns:
            list: A list of tuples representing the joint bounds, where each tuple contains the minimum and maximum values for a joint.
        '''
        maxbounds = self.robot.info().qMax
        minbounds = self.robot.info().qMin
        jointbounds = list(zip(maxbounds,minbounds))
        return jointbounds
     

    def get_dof(self):
        '''
        Retrieves the degree of freedom (DOF) of the robot arm.

        Returns:
            int: The degree of freedom of the robot arm.
        '''
        return 6

    def get_joint_idx(self):
        '''
        Retrieves the indices of the joints in the robot arm.

        Returns:
            list: A list of indices for the joints in the robot arm.
        '''
        return list(range(len(self.active_joints)))
     
    def get_links_info(self):
        '''
        Retrieves the TCP (Tool Center Point) link of the robot arm.

        Returns:
            str: The TCP link of the robot arm.
        '''
        return self.robot_chain.links[6].name

    def get_joint_state(self):
        '''
        Retrieves the current joint angles of the robot arm.

        Returns:
            list: A list of the current joint angles of the robot arm.
        '''
        _joint_states = self.robot.get_joint_states(is_radian=True)
        _joint_angles = _joint_states[1][0][0:6]

        return _joint_angles
     
    def get_current_joint_angles(self):
        '''
        Retrieves the current joint angles of the robot arm.

        Returns:
            list: A list of the current joint angles of the robot arm.
        '''
        _joint_states = self.robot.get_joint_states(is_radian=True)
        _joint_angles = _joint_states[1][0][0:6]

        return _joint_angles
    
    def get_current_joint_velocities(self):
        '''
        Retrieves the current joint velocities of the robot arm.

        Returns:
            list: A list of the current joint velocities of the robot arm.
        '''

        _joint_states = self.robot.get_joint_states(is_radian=True)
        _joint_velocities = _joint_states[1][1][0:6]

        return _joint_velocities

    def get_current_end_effector_pose(self):
        '''
        Retrieves the current pose of the robot arm's end effector.

        This function obtains the position and orientation of the end effector.

        Returns:
            pose: the [x, y, z, roll, pitch, yaw] value of tcp in meter and radian
        '''

        _pose = self.robot.get_position(is_radian=True)
        pose = _pose[1]
        pose[0] = pose[0] / 1000
        pose[1] = pose[1] / 1000
        pose[2] = pose[2] / 1000
        return pose    

    def get_current_tcp_speed(self):
        '''
        Retrieves the current tcp velocities of the robot arm.

        Returns:
            list: A list of the current tcp velocities of the robot arm.
        '''
        speed = self.robot.realtime_tcp_speed
        return speed

     
    # ----------------------------------------------------------------
    # Functions for joint control
    # ----------------------------------------------------------------

    def print_links_info(self, name):
        '''
        Prints the joint and link information of a robot.

        Args:
            name (str): 'base' or 'arm'
        '''
        if name == 'base':
            print("Base joint and link information:")
            for i, link in enumerate(self.robot_chain.links[:1]):  # Assuming the base is the first link
                print(f"Link {i}: {link.name}")
        elif name == 'arm':
            print("Arm joint and link information:")
            for i, link in enumerate(self.robot_chain.links[1:]):  # Assuming the arm starts from the second link
                print(f"Link {i + 1}: {link.name}")

    def move_arm_to_joint_values(self, joint_angles, target_vel=None, target_acc=None, MAX_VEL=None, MAX_ACC=None, wait_for_finish=None):
        '''
        Move arm to a specific set of joint angles, considering physics.

        Args:
            Set the servo angle, the API will modify self.last_used_angles value
        Note:
            https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/doc/api/xarm_api.md#def-set_servo_angleself-servo_idnone-anglenone-speednone-mvaccnone-mvtimenone-relativefalse-is_radiannone-waitfalse-timeoutnone-radiusnone-kwargs
        '''

        #! Force mode switch
        self.robot.set_mode(6) # 6: online joint
        self.robot.set_state(0)

        self.robot.set_servo_angle(angle=joint_angles, is_radian=True, speed=0.7, wait=wait_for_finish) # speed in rad/s

    def move_arm_to_joint_values_traj(self, target_trajectory, target_vel=None, target_acc=None, MAX_VEL=None, MAX_ACC=None):
        '''
        Move arm to a few set of joint angles, considering physics.

        Args:
            target_trajectory: A list of desired joint angles (in radians) for each joint of the arm.
            target_vel: Optional. A list of target velocities for each joint.
            target_acc: Optional. A list of target accelerations for each joint.
            MAX_VEL: Optional. A list of maximum velocities for each joint.
            MAX_ACC: Optional. A list of maximum accelerations for each joint.
        '''
        period = 1.0 / self.frequency
        self.update_robot_states()
        DOF = len(self.robot_states.q)

        for target_pos in target_trajectory:
            # Monitor fault on robot server
            if self.robot.isFault():
                raise Exception("Fault occurred on robot server, exiting ...")

            # Send command
            self.robot.set_servo_angle(angle=target_pos, is_radian=True, speed=target_vel, wait=True) # speed in rad/s
            print(f"Sent joint positions: {target_pos}")

            # Use sleep to control loop period
            time.sleep(period)

    # ----------------------------------------------------------------
    # Functions for eef control
    # ----------------------------------------------------------------

    # TODO

    def set_eef_values(self, _velocity_setpoint, _duration):
        '''
        Move arm's end effector to a target tcp velocity.

        Args:
            _velocity_setpoint: mm/s in transition, rad/s in orientation
            _duration: function calling loop time
        Notes:
            https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/doc/api/xarm_api.md#def-vc_set_cartesian_velocityself-speeds-is_radiannone-is_tool_coordfalse-duration-1-kwargs
        '''

        self.robot.set_mode(5) # 5 for cartesian vel
        self.robot.set_state(0)
        self.robot.vc_set_cartesian_velocity(speeds=[_velocity_setpoint[0],
                                                                        _velocity_setpoint[1],
                                                                        _velocity_setpoint[2],
                                                                        _velocity_setpoint[3],
                                                                        _velocity_setpoint[4],
                                                                        _velocity_setpoint[5]],
                                                                        duration=_duration)

    def move_end_effector_to_goal_pose(self, end_effector_goal_pose, is_radian=True, speed=10000, mvacc=500000, wait=False):
        '''
        Move arm's end effector to a target position.

        Args:
            end_effector_goal_pose (Pose): The desired pose of the end effector (includes both position in mm and euler_orientation), please use radian for orientaiton.
        '''
        # self.robot.set_state(4)
        self.robot.set_state(0)
        self.robot.set_position(x=end_effector_goal_pose[0]*1000 , y=end_effector_goal_pose[1]*1000,z=end_effector_goal_pose[2]*1000,
                                        roll=end_effector_goal_pose[3],pitch=end_effector_goal_pose[4],yaw=end_effector_goal_pose[5], speed=speed, mvacc=mvacc, is_radian=is_radian, wait=wait)



    def rotate_end_effector_joint(self, angle):
        '''
        Rotate the end effector of the robot arm by a specified angle by joint.

        Args:
            angle (float): The desired rotation angle in radians.
        '''
        current_joint_angles = self.get_current_joint_angles()
         
        target_joint_angles = current_joint_angles.copy()
        target_joint_angles[6] += angle 
        target_vel = 1
        self.robot.set_mode(6) # 0: joint control mode; 6: online joint
        self.robot.set_state(0)

        self.robot.set_servo_angle(angle=target_joint_angles, is_radian=True, speed=target_vel, wait=False) # speed in rad/s

    # ----------------------------------------------------------------
    # Functions for IK/FK
    # ----------------------------------------------------------------

    def joints_to_cartesian(self, joint_angles):
        '''
        Transforms the robot arm's joint angles to its Cartesian coordinates.

        Args:
            joint_angles (list): A list of joint angles for the robot arm.

        Returns:
            tuple: A tuple containing the Cartesian coordinates (position and orientation) of the robot arm.
        '''
        # Validate the number of joint values matches the number of active joints
        if len(joint_angles) != len(self.active_joints):
            raise ValueError("The number of joint values does not match the number of active joints")
         
        # Map joint values to the full joint chain
        full_joint_angles = np.zeros(len(self.robot_chain.links))
        active_joint_indices = [self.robot_chain.links.index(joint) for joint in self.active_joints]

        for i, joint_value in enumerate(joint_angles):
            full_joint_angles[active_joint_indices[i]] = joint_value

        # Calculate the end effector position and orientation
        cartesian_matrix = self.robot_chain.forward_kinematics(full_joint_angles)

        # Extract position and orientation
        position = cartesian_matrix[:3, 3]
        orientation_matrix = cartesian_matrix[:3, :3]

        # Convert rotation matrix to quaternion
        r = R.from_matrix(orientation_matrix)
        quaternion = r.as_quat()
        orientation = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]

        return position, orientation
     
    def cartesian_to_joints(self, position, orientation):
        '''
        Transforms the robot arm's Cartesian coordinates to its joint angles.

        Args:
            position (list): The Cartesian position of the robot arm.
            orientation (list): The Cartesian orientation of the robot arm.

        Returns:
            list: A list of joint angles corresponding to the given Cartesian coordinates.
        '''
        rotation_matrix = R.from_euler('xyz', orientation).as_matrix()

        # Combine rotation matrix and position into a list
        target_pose = np.eye(4)
        target_pose[:3, :3] = rotation_matrix
        target_pose[:3, 3] = position

        initial_joint_angles = [0] * len(self.robot_chain)

        # inverse kinematics calculations and return joint angles
        joint_angles = ikpy.inverse_kinematics.inverse_kinematic_optimization(
        chain=self.robot_chain,
        target_frame=target_pose,
        starting_nodes_angles=initial_joint_angles,
        orientation_mode='all',         
        )

        return joint_angles[1:8]


    def calculate_IK_error(self, goal_position, goal_orientation):
        '''
        Calculate the inverse kinematics (IK) error for performing pick-and-place manipulation of an object using a robot arm.

        Args:
            goal_position: The desired goal position for the target object.
            goal_orientation: The desired goal orientation for the target object.
        '''
        pass

    # ----------------------------------------------------------------
    # Functions for gripper
    # ----------------------------------------------------------------
     
    ### xArm 
     
    def find_gripper_xarm(self):
        '''
        Searches for the gripper on available serial ports and returns the port if found.

        Returns:
            str: The serial port where the gripper is connected, or None if not found.
        '''
        _pos = self.robot.get_gripper_position()
        _ver = self.robot.get_gripper_version()

        if _ver is not None and _pos is not None:
            print("Have Xarm gripper", _ver)
            return True
        else:
            print("Not found Xarm gripper")
            return None

    def get_gripper_position_xarm(self):
        '''
        Get the position of the XArm gripper.
        '''
        gripper_pos = self.robot.get_gripper_position()

        return gripper_pos[1]

    def gripper_goto_xarm(self, value, speed=5000, force=None):
        '''
        Moves the gripper to a specified position with given speed.

        Args:
            value (int): Position of the gripper. Integer between 0 and 800.
                        0 represents the open position, and 255 represents the closed position.
            speed (int): Speed of the gripper movement, between 0 and 8000.
            force (int): Not applicable for xarm gripper
         
        Note:
            - 0 means fully closed.
            - 800 means fully open.
        '''
        self.robot.set_gripper_position(pos=value, speed=speed, wait=False, timeout=1, auto_enable=True)

    def open_gripper_xarm(self):
        ''' Opens the gripper to its maximum position with maximum speed and force. '''
        self.gripper_goto_xarm(value=850, speed=5000, force=None)

    def close_gripper_xarm(self):
        '''Closes the gripper to its minimum position with maximum speed and force.'''
        self.gripper_goto_xarm(value=0, speed=5000, force=None)
     
    ### Robotiq
    
    # ------------------------------------------------------------------
    # New: USB Connection Logic for Robotiq (Merged)
    # ------------------------------------------------------------------

    def _scan_robotiq_usb_port(self):
        """
        [NEW] 扫描串口查找 Robotiq 夹爪（USB 直连电脑）。
        依次尝试 /dev/ttyS* 和 /dev/ttyUSB*。
        """
        ports = [f"/dev/ttyS{i}" for i in range(2)] + [f"/dev/ttyUSB{i}" for i in range(5)]

        for port in ports:
            try:
                # print(f"Trying port: {port}")
                ser = serial.Serial(port, baudrate=115200, timeout=1.0)
                device = mm.Instrument(ser, 9, mode=mm.MODE_RTU)
                device.write_registers(1000, [0, 100, 0])
                registers = device.read_registers(2000, 3, 4)
                pos_request_echo_reg3 = registers[1] & 0b0000000011111111
                if pos_request_echo_reg3 == 100:
                    print(f"Gripper found on port: {port}")
                    return port
            except Exception as e:
                pass
                # print(f"Port {port} is not the gripper: {e}")
            finally:
                if "ser" in locals() and ser.is_open:
                    ser.close()
        return None

    def connect_robotiq_via_usb(self):
        """
        [NEW] 连接 USB 直连的 Robotiq 夹爪并在需要时激活。
        调用此函数将启用 USB 控制模式，覆盖默认的 xArm 透传控制。
        """
        try:
            gripper_port = self._scan_robotiq_usb_port()
            if gripper_port:
                self.usb_gripper = pyRobotiqGripper.RobotiqGripper(portname=gripper_port)
                print(f"Connected to gripper on port {self.usb_gripper.portname}.")
                if not self.usb_gripper.isActivated():
                    print("Activating gripper...")
                    self.usb_gripper.activate()
            else:
                print("Failed to connect to the gripper. No port detected.")
        except Exception as e:
            print(f"Failed to connect to gripper: {str(e)}")

    # ------------------------------------------------------------------
    # Robotiq Control (Merged: supports both xArm-Pass-Through and USB)
    # ------------------------------------------------------------------
     
    def find_gripper_robotiq(self):
        """
        Config the parameter via Python SDK (Standard xArm connection)
        """
        # Baud rate
        # Modify the baud rate to 115200, the default is 2000000.
        self.robot.set_tgpio_modbus_baudrate(115200)  

        # TCP Payload and offset
        # Robotiq 2F/85 Gripper
        self.robot.set_tcp_load(0.925, [0, 0, 58])
        self.robot.set_tcp_offset([0, 0, 174, 0, 0, 0])
        self.robot.save_conf()

        # Self-Collision Prevention Model
        # Robotiq 2F/85 Gripper
        self.robot.set_collision_tool_model(4)

        self.robot.robotiq_reset()
        self.robot.robotiq_set_activate()    #enable the robotiq gripper
        
    def get_gripper_position_robotiq(self, number_of_registers=3):
        """
        Reading the status of robotiq gripper.
        Supports USB connection if connect_robotiq_via_usb() was called successfully.
        """
        if self.usb_gripper:
            # USB Mode
            try:
                pos = self.usb_gripper.getPosition()
                return int(pos)
            except Exception as e:
                print(f"Failed to read USB gripper position: {e}")
                return 0
        else:
            # Standard xArm Mode
            status = self.robot.robotiq_get_status(number_of_registers=number_of_registers)
            gripper_width = status[1][-2]
            return gripper_width

    def gripper_goto_robotiq(self, pos, speed=0xFF, force=0xFF, wait=False, timeout=5, **kwargs):
        """
        Go to the position with determined speed and force.
        
        :param pos: position of the gripper. Integer between 0 and 255. 0 being the open position and 255 being the close position.
        """
        if self.usb_gripper:
            # USB Mode
            try:
                self.usb_gripper.goTo(position=pos, speed=speed, force=force)
                print(f"USB Gripper moved to {pos}")
                return (0, "Success (USB)")
            except Exception as e:
                print(f"USB Gripper Error: {e}")
                return (1, str(e))
        else:
            # Standard xArm Mode
            return self.robot.robotiq_set_position(pos, speed=speed, force=force, wait=wait, timeout=timeout, **kwargs)
     
    def open_gripper_robotiq(self, speed=0xFF, force=0xFF, wait=False, timeout=5, **kwargs):
        """
        Open the robotiq gripper.
        USB Mode: Position 0.
        """
        if self.usb_gripper:
            # USB Mode (Open = 0)
            return self.gripper_goto_robotiq(pos=0, speed=speed, force=force)
        else:
            # Standard xArm Mode
            return self.robot.robotiq_open(speed=speed, force=force, wait=wait, timeout=timeout, **kwargs)

    def close_gripper_robotiq(self, speed=0xFF, force=0xFF, wait=False, timeout=5, **kwargs):
        """
        Close the robotiq gripper.
        USB Mode: Position 255.
        """
        if self.usb_gripper:
             # USB Mode (Close = 255)
            return self.gripper_goto_robotiq(pos=255, speed=speed, force=force)
        else:
            # Standard xArm Mode
            return self.robot.robotiq_close(speed=speed, force=force, wait=wait, timeout=timeout, **kwargs)
