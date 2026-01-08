PYTHONPATH=/home/ubuntu/Desktop/hil-serl python examples/train_bc.py   --exp_name=umi_pick   --bc_checkpoint_path=/home/ubuntu/Documents/data/pick1/bc_ckpts

PYTHONPATH=/home/ubuntu/Desktop/hil-serl \
python examples/train_rlpd.py \
  --exp_name=umi_pick \
  --learner=True \
  --offline=True \
  --demo_path=/home/ubuntu/Desktop/hil-serl/demo_data/pick_parallel.pkl \
  --checkpoint_path=/home/ubuntu/Documents/data/pick1/rlpd_ckpts \
  --debug=True

conda create -n hilserl_cpu python=3.10

cd serl_launcher
pip install -r requirements.txt (xarm所需的一些包我暂时也放在这了)

cd serl_robot_infra/xarm_env/BestMan_Xarm
git clone https://github.com/xArm-Developer/xArm-Python-SDK.git
cd xArm-Python-SDK
python3 setup.py install

pip install -e serl_launcher
pip install -e serl_robot_infra