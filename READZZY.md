## 数采
cd serl_robot_infra/umi
cd data_collector_opt
conda create -n shucai python=3.8.0 #一定要用这个版本的python，没法兼容
pip install -r requirements.txt

cd start_process
bash unified.launch
python device_pairing.py #换了设备要更新config.json

数据存在serl_robot_infra/umi/data_collector_opt/DATA，拔插头

## 数据格式转换pkl（直接从raw转换，跳过hdf5中间步骤，脚本里改输入输出路径）
python serl_robot_infra/umi/raw2pkl.py

## replay pkl检验
PYTHONPATH=/home/ubuntu/Desktop/hil-serl:$PYTHONPATH python serl_robot_infra/umi/replay_pkl.py 

## 标注
PYTHONPATH=/home/ubuntu/Desktop/hil-serl   python examples/annotate_rewards_from_pkl.py

## 分类器训练
source /opt/ros/noetic/setup.bash
PYTHONPATH=/home/ubuntu/Desktop/hil-serl python examples/train_reward_classifier.py --exp_name umi_pick 

## 分类器测试
PYTHONPATH=/home/ubuntu/Desktop/hil-serl python examples/test_reward_classifier.py 
参数在脚本里改

## BC训练
PYTHONPATH=/home/ubuntu/Desktop/hil-serl python examples/train_bc.py   --exp_name=umi_pick   --bc_checkpoint_path=/home/ubuntu/Documents/data/pick1/bc_ckpts

## 离线RL
PYTHONPATH=/home/ubuntu/Desktop/hil-serl \
python examples/train_rlpd.py \
  --exp_name=umi_pick \
  --learner=True \
  --offline=True \
  --demo_path=/home/ubuntu/Desktop/hil-serl/demo_data/unplug.pkl \
  --checkpoint_path=/home/ubuntu/Desktop/hil-serl/demo_ckpt \
  --debug=True

Actor推理
PYTHONPATH=/home/ubuntu/Desktop/hil-serl \
python examples/train_rlpd.py \
    --exp_name=umi_pick \
    --actor \
    --eval_checkpoint_step=10 \
    --eval_n_trajs=5 \
    --checkpoint_path=/home/ubuntu/Desktop/hil-serl/demo_ckpt

## 在线RL
PYTHONPATH=/home/ubuntu/Desktop/hil-serl:$PYTHONPATH \
python examples/train_rlpd.py \
  --exp_name=umi_pick \
  --learner \
  --demo_path=/home/ubuntu/Desktop/hil-serl/demo_data/unplug.pkl \
  --checkpoint_path=/home/ubuntu/Desktop/hil-serl/demo_ckpt \
  --debug=True

source ~/catkin_ws/devel/setup.bash
PYTHONPATH=/home/ubuntu/Desktop/hil-serl:$PYTHONPATH
python examples/train_rlpd.py \
  --exp_name=umi_pick \
  --actor \
  --checkpoint_path=/home/ubuntu/Desktop/hil-serl/demo_ckpt

# 配环境
## 遥操作以外环境
conda create -n hilserl_cpu python=3.10

cd serl_launcher
pip install -r requirements.txt (xarm所需的一些包我暂时也放在这了)

cd serl_robot_infra/xarm_env/BestMan_Xarm
git clone https://github.com/xArm-Developer/xArm-Python-SDK.git
cd xArm-Python-SDK
python3 setup.py install

pip install -e serl_launcher
pip install -e serl_robot_infra

## 遥操作所需环境

cd serl_robot_infra/umi/start_process

### ROS1 noetic安装

本地系统级安装
wget http://fishros.com/install -O fishros && . fishros

### rosdep修复 （暂时先这样）

./fix_ros1_dependencies.sh

### XVSDK安装

本地系统级安装，这里可能不全，后续再维护
sudo -E ./install-ros1.sh XVSDK_focal_amd64_0107.deb
sudo apt -y install ros-noetic-ddynamic-reconfigure

### 电脑里之前做过数采的可以直接跳到这一步

### rostopic发布

bash  unified_launcher.sh 

/rosout
/rosout_agg
/tf
/tf_static
/xv_sdk/250801DR48FP25002063/clamp/Data
/xv_sdk/250801DR48FP25002063/color_camera/camera_info
/xv_sdk/250801DR48FP25002063/color_camera/image
/xv_sdk/250801DR48FP25002063/color_camera/point_cloud
/xv_sdk/250801DR48FP25002063/fisheye_cameras/left/camera_info
/xv_sdk/250801DR48FP25002063/fisheye_cameras/left/image
/xv_sdk/250801DR48FP25002063/fisheye_cameras/left2/camera_info
/xv_sdk/250801DR48FP25002063/fisheye_cameras/left2/image
/xv_sdk/250801DR48FP25002063/fisheye_cameras/right/camera_info
/xv_sdk/250801DR48FP25002063/fisheye_cameras/right/image
/xv_sdk/250801DR48FP25002063/fisheye_cameras/right2/camera_info
/xv_sdk/250801DR48FP25002063/fisheye_cameras/right2/image
/xv_sdk/250801DR48FP25002063/imu_sensor/data_raw
/xv_sdk/250801DR48FP25002063/parameter_descriptions
/xv_sdk/250801DR48FP25002063/parameter_updates
/xv_sdk/250801DR48FP25002063/rgbd_camera/camera_info
/xv_sdk/250801DR48FP25002063/rgbd_camera/image
/xv_sdk/250801DR48FP25002063/slam/pose
/xv_sdk/250801DR48FP25002063/slam/trajectory
/xv_sdk/250801DR48FP25002063/tof_camera/camera_info
/xv_sdk/250801DR48FP25002063/tof_camera/image
/xv_sdk/new_device
/xv_sdk/parameter_descriptions
/xv_sdk/parameter_updates


### usb连上夹爪 配对 序列号写入config

pip install psutil==5.9.8 PyYAML openvr rosdep

bash pairing_process.sh

### 测试遥操作
source ~/catkin_ws/devel/setup.bash 写入.bashrc
pip install pynput

PYTHONPATH=/home/ubuntu/Desktop/hil-serl:$PYTHONPATH python serl_robot_infra/umi/umi_teleop.py

pip install keyboard

### 在线rl人类干预所需环境
多usb
cd serl_robot_infra/umi/start_process
sudo -E bash multi-support.sh

filter_unified_launcher.sh
默认filter-by-config，首次使用需要--no-filter，然后运行bash pairing_process.sh，默认no vive，要启用需要--vive