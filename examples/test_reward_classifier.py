#!/usr/bin/env python3
"""
随机抽取 HDF5 文件与其中一帧，调用已训练好的 reward classifier 推理。
按一次空格：重新随机抽取并推理；按 q/ESC 或关闭窗口退出。

用法示例：
PYTHONPATH=/home/ubuntu/Desktop/hil-serl \
python examples/test_reward_classifier.py \
    --hdf5_dir=/home/ubuntu/Documents/data/pick1/hdf5 \
    --checkpoint_dir=/home/ubuntu/Desktop/hil-serl/classifier_ckpt \
    --exp_name=umi_pick
"""

import os
import glob
import random
import h5py
import numpy as np
import cv2
import jax
import jax.numpy as jnp
from absl import app, flags

from serl_launcher.networks.reward_classifier import load_classifier_func
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("hdf5_dir", "/home/ubuntu/Documents/data/pick1/hdf5", "HDF5 目录，包含 *.hdf5。")
flags.DEFINE_string("checkpoint_dir", "/home/ubuntu/Desktop/hil-serl/classifier_ckpt", "分类器 checkpoint 目录。")
flags.DEFINE_string("exp_name", "umi_pick", "对应 CONFIG_MAPPING 的实验名。")
flags.DEFINE_integer("resize", 128, "送入分类器的图像尺寸（与训练一致）。")
flags.DEFINE_integer("seed", 0, "随机种子，用于抽取文件和帧。")


def resize_image(img: np.ndarray, target_size: int = 128) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[:2] != (target_size, target_size):
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.uint8)


def sample_hdf5_and_frame(hdf5_dir: str, rng: random.Random, image_size: int):
    files = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))
    if not files:
        raise ValueError(f"在 {hdf5_dir} 未找到 .hdf5 文件")
    hdf5_path = rng.choice(files)

    with h5py.File(hdf5_path, "r") as f:
        if "observations/front/images" in f:
            images = f["observations/front/images"][:]  # (T, H, W, 3)
        else:
            images = f["observations/images"][:]
    T = len(images)
    frame_idx = rng.randrange(T)
    img_raw = images[frame_idx]
    img_resized = resize_image(img_raw, image_size)
    return hdf5_path, frame_idx, img_raw, img_resized


def run_inference(classifier_fn, image_key, hdf5_dir, rng, resize):
    hdf5_path, frame_idx, img_raw, img_resized = sample_hdf5_and_frame(hdf5_dir, rng, resize)
    obs = {image_key: jnp.array(img_resized)[None]}  # shape (1, H, W, 3)
    logits = classifier_fn(obs)
    prob = jax.nn.sigmoid(logits).reshape(-1)[0]
    prob_f = float(prob)
    logit_f = float(jnp.asarray(logits).reshape(-1)[0])
    label = "SUCCESS" if prob_f >= 0.5 else "FAIL"
    color = (0, 255, 0) if prob_f >= 0.5 else (0, 0, 255)

    print(f"文件: {hdf5_path}")
    print(f"帧: {frame_idx} / {img_raw.shape}")
    print(f"logit: {logit_f:.4f}, prob: {prob_f:.4f}, label: {label}")

    img_show = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR) if img_raw.shape[-1] == 3 else img_raw.copy()
    cv2.putText(img_show, f"{label} ({prob_f:.2f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return img_show


def main(_):
    rng = random.Random(FLAGS.seed)

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
        img_show = run_inference(classifier_fn, image_key, FLAGS.hdf5_dir, rng, FLAGS.resize)
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
