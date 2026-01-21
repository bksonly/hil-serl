#!/bin/bash

# 修复 ROS1 编译依赖问题
# 解决 ddynamic_reconfigure 和 rosdep 缺失问题

set -e

echo "=========================================="
echo "修复 ROS1 编译依赖"
echo "=========================================="

# 1. 安装 rosdep（如果未安装）
echo ""
echo "1. 检查并安装 rosdep..."
if ! command -v rosdep &> /dev/null; then
    echo "   安装 rosdep..."
    sudo apt-get update
    sudo apt-get install -y python3-rosdep
    sudo rosdep init || echo "   注意: rosdep 可能已初始化"
    rosdep update
else
    echo "   ✓ rosdep 已安装"
fi

# 2. 安装 ddynamic_reconfigure
echo ""
echo "2. 检查并安装 ddynamic_reconfigure..."
if ! dpkg -l | grep -q "ros-noetic-ddynamic-reconfigure"; then
    echo "   安装 ros-noetic-ddynamic-reconfigure..."
    sudo apt-get update
    sudo apt-get install -y ros-noetic-ddynamic-reconfigure
else
    echo "   ✓ ddynamic_reconfigure 已安装"
fi

# 3. 确保 ROS 环境已 source
echo ""
echo "3. 检查 ROS 环境..."
if [ -z "$ROS_DISTRO" ]; then
    echo "   加载 ROS 环境..."
    source /opt/ros/noetic/setup.bash
    export ROS_DISTRO=noetic
else
    echo "   ✓ ROS 环境已加载 (ROS_DISTRO=$ROS_DISTRO)"
fi

# 4. 验证安装
echo ""
echo "4. 验证安装..."
if command -v rosdep &> /dev/null; then
    echo "   ✓ rosdep 可用"
else
    echo "   ✗ rosdep 仍然不可用"
    exit 1
fi

if dpkg -l | grep -q "ros-noetic-ddynamic-reconfigure"; then
    echo "   ✓ ddynamic_reconfigure 已安装"
else
    echo "   ✗ ddynamic_reconfigure 仍然未安装"
    exit 1
fi

# 5. 检查 ROS 包路径
echo ""
echo "5. 检查 ROS 包路径..."
if [ -d "/opt/ros/noetic/share/ddynamic_reconfigure" ]; then
    echo "   ✓ ddynamic_reconfigure 在 ROS 路径中"
else
    echo "   ⚠ 警告: ddynamic_reconfigure 目录未找到，可能需要重新安装"
fi

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
echo ""
echo "现在可以尝试重新编译："
echo "  cd ~/catkin_ws"
echo "  source /opt/ros/noetic/setup.bash"
echo "  catkin_make -DXVSDK_INCLUDE_DIRS=\"/usr/include/xvsdk\" -DXVSDK_LIBRARIES=\"/usr/lib/libxvsdk.so\""
echo ""
