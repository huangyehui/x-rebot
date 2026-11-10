#!/usr/bin/env python3
"""从 tensorboard 事件文件导出训练曲线为一张 png，方便快速判断收敛。

用法:
    python -m humanoid_rl.full_body.plot_metrics \
        --logdir checkpoints/full_body_walk_robust3/tb_logs \
        --out checkpoints/full_body_walk_robust3/metrics.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 关注的指标：收敛看 loss/value_loss/approx_kl，学习效果看 ep_rew/ep_len
PANELS = [
    ("rollout/ep_rew_mean", "ep_rew_mean (episode reward)"),
    ("rollout/ep_len_mean", "ep_len_mean (episode length)"),
    ("train/loss", "total loss"),
    ("train/value_loss", "value loss"),
    ("train/approx_kl", "approx_kl (update size)"),
    ("train/explained_variance", "explained_variance"),
]


def _load(logdir, merge):
    """返回 [(run_name, EventAccumulator), ...]，按修改时间排序。"""
    runs = sorted(glob.glob(os.path.join(logdir, "*", "events*")),
                  key=os.path.getmtime)
    if not runs:
        # logdir 本身可能就是单个 run 目录
        runs = sorted(glob.glob(os.path.join(logdir, "events*")),
                      key=os.path.getmtime)
    if not merge and runs:
        runs = runs[-1:]  # 只画最新 run
    out = []
    for f in runs:
        ea = EventAccumulator(f)
        ea.Reload()
        out.append((os.path.basename(os.path.dirname(f)) or "run", ea))
    return out


def main():
    parser = argparse.ArgumentParser(description="导出训练曲线为 png")
    parser.add_argument("--logdir", type=str,
                        default="checkpoints/full_body_walk_robust3/tb_logs")
    parser.add_argument("--out", type=str,
                        default="checkpoints/full_body_walk_robust3/metrics.png")
    parser.add_argument("--merge", action="store_true",
                        help="把该 logdir 下所有 run 画在同一张图(默认只画最新 run)")
    args = parser.parse_args()

    eas = _load(args.logdir, args.merge)
    if not eas:
        print(f"[plot] 未找到事件文件: {args.logdir}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for ax, (tag, title) in zip(axes, PANELS):
        plotted = False
        for name, ea in eas:
            if tag in ea.Tags()["scalars"]:
                pts = ea.Scalars(tag)
                xs = [p.step for p in pts]
                ys = [p.value for p in pts]
                ax.plot(xs, ys, label=name if len(eas) > 1 else None, lw=1.2)
                plotted = True
        ax.set_title(title)
        ax.set_xlabel("timesteps")
        ax.grid(alpha=0.3)
        if not plotted:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center",
                    transform=ax.transAxes)
        if len(eas) > 1 and plotted:
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"[plot] 已保存: {args.out}  (基于 {len(eas)} 个 run)")


if __name__ == "__main__":
    main()
