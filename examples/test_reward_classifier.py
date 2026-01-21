#!/usr/bin/env python3
"""
从 PKL 文件中随机抽取一个 transition，调用已训练好的 reward classifier 推理。
按一次空格：重新随机抽取并推理；按 q/ESC 或关闭窗口退出。

用法示例：
PYTHONPATH=/home/ubuntu/Desktop/hil-serl python examples/test_reward_classifier.py 
"""

import os
import pickle as pkl
import random
import numpy as np
import cv2
import jax
import jax.numpy as jnp
from absl import app, flags

from serl_launcher.networks.reward_classifier import load_classifier_func
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("pkl_path", "/home/ubuntu/Desktop/hil-serl/demo_data/unplug.pkl", "PKL 文件路径，包含 transitions 列表。")
flags.DEFINE_string("checkpoint_dir", "/home/ubuntu/Desktop/hil-serl/classifier_ckpt", "分类器 checkpoint 目录。")
flags.DEFINE_string("exp_name", "umi_pick", "对应 CONFIG_MAPPING 的实验名。")
flags.DEFINE_integer("resize", 128, "送入分类器的图像尺寸（与训练一致）。")
flags.DEFINE_integer("seed", 0, "随机种子，用于抽取 transition。")


def resize_image(img: np.ndarray, target_size: int = 128) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[:2] != (target_size, target_size):
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.uint8)


def load_pkl_transitions(pkl_path: str):
    """加载 PKL 文件中的所有 transitions。"""
    with open(pkl_path, 'rb') as f:
        transitions = pkl.load(f)
    
    if not isinstance(transitions, list):
        raise ValueError(f"PKL文件应该包含一个列表，但得到了 {type(transitions)}")
    
    if len(transitions) == 0:
        raise ValueError("PKL文件为空")
    
    # 验证格式
    sample = transitions[0]
    if "observations" not in sample or "image" not in sample["observations"]:
        raise ValueError("PKL文件格式错误：每个transition应包含 observations['image']")
    
    return transitions


def sample_pkl_transition(transitions: list, rng: random.Random, image_size: int):
    """从 transitions 列表中随机抽取一个 transition。"""
    transition_idx = rng.randrange(len(transitions))
    transition = transitions[transition_idx]
    img_raw = transition["observations"]["image"]  # (H, W, 3), 通常是 128x128
    
    if img_raw.dtype != np.uint8:
        img_raw = np.clip(img_raw, 0, 255).astype(np.uint8)
    
    img_resized = resize_image(img_raw, image_size)
    return transition_idx, transition, img_raw, img_resized


def run_inference(classifier_fn, image_key, transitions, rng, resize):
    transition_idx, transition, img_raw, img_resized = sample_pkl_transition(transitions, rng, resize)
    obs = {image_key: jnp.array(img_resized)[None]}  # shape (1, H, W, 3)
    logits = classifier_fn(obs)
    prob = jax.nn.sigmoid(logits).reshape(-1)[0]
    prob_f = float(prob)
    logit_f = float(jnp.asarray(logits).reshape(-1)[0])
    label = "SUCCESS" if prob_f >= 0.5 else "FAIL"
    color = (0, 255, 0) if prob_f >= 0.5 else (0, 0, 255)

    print(f"Transition索引: {transition_idx}/{len(transitions)-1}")
    print(f"图像尺寸: {img_raw.shape}")
    print(f"logit: {logit_f:.4f}, prob: {prob_f:.4f}, label: {label}")

    # 放大图像用于显示（PKL中的图像通常是128x128）
    img_display = cv2.resize(img_raw, (512, 512), interpolation=cv2.INTER_LINEAR) if img_raw.shape[:2] == (128, 128) else img_raw
    img_show = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR) if img_display.shape[-1] == 3 else img_display.copy()
    cv2.putText(img_show, f"{label} ({prob_f:.2f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return img_show


def main(_):
    rng = random.Random(FLAGS.seed)

    # 检查输入文件
    if not os.path.exists(FLAGS.pkl_path):
        raise ValueError(f"PKL文件不存在: {FLAGS.pkl_path}")
    
    # 加载所有 transitions
    print(f"加载 PKL 文件: {FLAGS.pkl_path}")
    transitions = load_pkl_transitions(FLAGS.pkl_path)
    print(f"共 {len(transitions)} 个 transitions")

    assert FLAGS.exp_name in CONFIG_MAPPING, "未知 exp_name"
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    classifier_keys = config.classifier_keys
    if classifier_keys is None or len(classifier_keys) != 1:
        raise ValueError(f"classifier_keys 需为单一图像键，当前: {classifier_keys}")
    image_key = classifier_keys[0]

    # 先构造一个样本形状用于初始化网络
    dummy_img = np.zeros((FLAGS.resize, FLAGS.resize, 3), dtype=np.uint8)
    dummy_obs = {image_key: jnp.array(dummy_img)[None]}

    key = jax.random.PRNGKey(0)
    classifier_fn = load_classifier_func(
        key=key,
        sample=dummy_obs,
        image_keys=classifier_keys,
        checkpoint_path=FLAGS.checkpoint_dir,
    )

    cv2.namedWindow("Classifier Result", cv2.WINDOW_NORMAL)
    while True:
        img_show = run_inference(classifier_fn, image_key, transitions, rng, FLAGS.resize)
        cv2.imshow("Classifier Result", img_show)
        k = cv2.waitKey(0) & 0xFF
        # 空格：再次随机采样和推理；q/ESC：退出
        if k in [ord('q'), 27]:
            break
        elif k == ord(' '):
            continue
        else:
            # 其他键也继续下一次采样，保持简单
            continue
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
