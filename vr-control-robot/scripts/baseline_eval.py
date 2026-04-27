import argparse
from pathlib import Path
import h5py
import numpy as np


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="./artifacts/datasets/smoke_train.h5")
    args = parser.parse_args()

    h5_path = Path(args.h5)
    with h5py.File(h5_path, "r") as h5:
        act = np.asarray(h5["act"], dtype=np.float32)
        ep = np.asarray(h5["episode_id"], dtype=np.int32)

    # Baseline 1: 全零动作
    pred_zero = np.zeros_like(act)
    zero_mse = mse(pred_zero, act)

    # Baseline 2: 全局均值动作
    mean_action = np.mean(act, axis=0, keepdims=True)
    pred_mean = np.repeat(mean_action, repeats=act.shape[0], axis=0)
    mean_mse = mse(pred_mean, act)

    # Baseline 3: 同 episode 前一帧动作（第一帧回退到 0）
    pred_prev = np.zeros_like(act)
    pred_prev[1:] = act[:-1]
    boundary = np.where(ep[1:] != ep[:-1])[0] + 1
    pred_prev[boundary] = 0.0
    prev_mse = mse(pred_prev, act)

    print(f"[baseline] h5={h5_path}")
    print(f"[baseline] zero_action_mse={zero_mse:.6f}")
    print(f"[baseline] mean_action_mse={mean_mse:.6f}")
    print(f"[baseline] prev_action_mse={prev_mse:.6f}")


if __name__ == "__main__":
    main()
