#!/usr/bin/env python3
"""加载 full_body 行走模型评估行走效果

用法:
    python -m humanoid_rl.full_body.eval_walk --model checkpoints/full_body_walk_robust3/walk
    python -m humanoid_rl.full_body.eval_walk --model checkpoints/full_body_walk_robust3/walk --gif walk_eval.gif
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from humanoid_rl.full_body.walk_env import FullBodyWalkEnv


def main():
    parser = argparse.ArgumentParser(description="评估 full_body 行走模型")
    parser.add_argument("--model", type=str,
                        default="checkpoints/full_body_walk_robust3/walk")
    parser.add_argument("--target-speed", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gif", type=str, default="", help="保存演示 gif 的路径(可选)")
    args = parser.parse_args()

    vn_path = f"{args.model}_vecnorm.pkl"
    render_mode = "rgb_array" if args.gif else None

    # 评估用 VecNormalize 的观测统计（training=False，不更新、不归一化奖励）
    venv = DummyVecEnv([lambda: FullBodyWalkEnv(render_mode=render_mode,
                                                target_speed=args.target_speed)])
    if os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
        venv.training = False
        venv.norm_reward = False
        print(f"[eval] 已加载观测归一化统计: {vn_path}")
    else:
        print(f"[eval] 未找到 vecnorm，使用原始观测: {vn_path}")

    model = PPO.load(args.model, device="cpu")
    print(f"[eval] 已加载模型: {args.model}  目标速度 {args.target_speed} m/s")

    raw_env = venv.venv.envs[0].unwrapped
    frames = []

    lengths, survived = [], []
    mean_vel, dist_list, mean_tu, mean_hu, mean_lat = [], [], [], [], []
    for ep in range(args.episodes):
        obs = venv.reset()
        ep_len = 0
        vels, tus, hus, lats = [], [], [], []
        last_dist = 0.0
        for _ in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = venv.step(action)
            info = infos[0]
            vels.append(info["forward_vel"])
            tus.append(info["torso_upright"])
            hus.append(info["head_upright"])
            lats.append(info["lateral_dev"])
            last_dist = info["distance"]
            ep_len += 1
            if args.gif and ep == 0:
                f = raw_env.render()
                if f is not None:
                    frames.append(f)
            if done:
                break
        fell = ep_len < args.max_steps
        lengths.append(ep_len)
        survived.append(0.0 if fell else 1.0)
        mean_vel.append(np.mean(vels)); dist_list.append(last_dist)
        mean_tu.append(np.mean(tus)); mean_hu.append(np.mean(hus))
        mean_lat.append(np.mean(lats))
        status = "摔倒/停滞" if fell else "走完全程"
        print(f"  ep{ep+1}: 存活 {ep_len:4d}/{args.max_steps} 步  "
              f"前进 {last_dist:5.2f}m  均速 {np.mean(vels):.3f}m/s  "
              f"躯干直立 {np.mean(tus):.3f}  头直立 {np.mean(hus):.3f}  "
              f"侧偏 {np.mean(lats):.3f}m  → {status}")

    print("=" * 64)
    print(f"平均存活: {np.mean(lengths):.0f}/{args.max_steps} 步   "
          f"走完全程率: {np.mean(survived):.0%}")
    print(f"平均前进: {np.mean(dist_list):.2f}m   均速: {np.mean(mean_vel):.3f}m/s "
          f"(目标 {args.target_speed})   躯干直立: {np.mean(mean_tu):.3f}   "
          f"头部直立: {np.mean(mean_hu):.3f}   侧偏: {np.mean(mean_lat):.3f}m")
    print("=" * 64)

    if args.gif and frames:
        import imageio
        imageio.mimsave(args.gif, frames, fps=30)
        print(f"演示动画已保存: {args.gif} ({len(frames)} 帧)")

    venv.close()


if __name__ == "__main__":
    main()
