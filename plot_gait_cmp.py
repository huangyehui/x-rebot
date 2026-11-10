#!/usr/bin/env python3
"""多模型步态对比图：左右脚离地高度 + 左右髋角随时间变化。

用法:
    python -m humanoid_rl.full_body.plot_gait_cmp --out cmp.png \
        --models 原模型=checkpoints/full_body_walk_robust3/walk 手术=checkpoints/full_body_walk_surgery/walk_l05
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from humanoid_rl.full_body.gait_diag import load

# 中文字体(有就用，没有就退回默认)
for f in ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "AR PL UMing CN"):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False


def rollout(path, steps, seed=0):
    model, venv = load(path)
    env = venv.venv.envs[0].unwrapped
    venv.seed(seed)
    obs = venv.reset()
    rec = {k: [] for k in ("rz", "lz", "rh", "lh")}
    for _ in range(steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = venv.step(a)
        rec["rz"].append(env.data.xpos[env._rfoot_id][2])
        rec["lz"].append(env.data.xpos[env._lfoot_id][2])
        rec["rh"].append(env.data.qpos[env._rhip_adr])
        rec["lh"].append(env.data.qpos[env._lhip_adr])
        if done[0]:
            break
    return {k: np.array(v) for k, v in rec.items()}, env.dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="名字=路径")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out", default="gait_cmp.png")
    a = p.parse_args()

    n = len(a.models)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.2 * n), squeeze=False)
    for i, spec in enumerate(a.models):
        name, path = spec.split("=", 1)
        r, dt = rollout(path, a.steps)
        t = np.arange(len(r["rz"])) * dt
        ax = axes[i, 0]
        ax.plot(t, r["rz"], "r-", label="右脚")
        ax.plot(t, r["lz"], "b-", label="左脚")
        ax.set_title(f"{name}: 脚离地高度 (存活 {len(t)} 步)")
        ax.set_ylabel("z (m)"); ax.set_xlim(0, a.steps * dt); ax.legend(loc="upper right")
        ax = axes[i, 1]
        ax.plot(t, r["rh"], "r-", label="右髋")
        ax.plot(t, r["lh"], "b-", label="左髋")
        ax.set_title(f"{name}: 髋关节角")
        ax.set_ylabel("rad"); ax.set_xlim(0, a.steps * dt); ax.legend(loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("时间 (s)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"已保存 {a.out}")


if __name__ == "__main__":
    main()
