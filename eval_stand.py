#!/usr/bin/env python3
"""加载 full_body 站立模型评估站立效果

用法:
    python -m humanoid_rl.full_body.eval_stand                         # 默认加载 checkpoints/full_body/stand
    python -m humanoid_rl.full_body.eval_stand --model checkpoints/full_body_robust2/stand
    python -m humanoid_rl.full_body.eval_stand --gif stand_eval.gif    # 另存演示动画
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from humanoid_rl.full_body.full_body_env import FullBodyStandEnv


def main():
    parser = argparse.ArgumentParser(description="评估 full_body 站立模型")
    parser.add_argument("--model", type=str, default="checkpoints/full_body/stand")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gif", type=str, default="", help="保存演示 gif 的路径(可选)")
    args = parser.parse_args()

    vn_path = f"{args.model}_vecnorm.pkl"
    render_mode = "rgb_array" if args.gif else None

    # 评估时用 VecNormalize 的观测统计（training=False，不更新、不归一化奖励）
    venv = DummyVecEnv([lambda: FullBodyStandEnv(render_mode=render_mode)])
    if os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
        venv.training = False
        venv.norm_reward = False
        print(f"[eval] 已加载观测归一化统计: {vn_path}")
    else:
        print(f"[eval] 未找到 vecnorm，使用原始观测: {vn_path}")

    model = PPO.load(args.model, device="cpu")
    print(f"[eval] 已加载模型: {args.model}")

    raw_env = venv.venv.envs[0].unwrapped  # 取底层 env 读取 info/render
    frames = []

    lengths, survived, mean_ph, mean_hu, mean_tu, mean_com = [], [], [], [], [], []
    for ep in range(args.episodes):
        obs = venv.reset()
        ep_len = 0
        phs, hus, tus, coms = [], [], [], []
        for _ in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = venv.step(action)
            info = infos[0]
            phs.append(info["pelvis_height"])
            hus.append(info["head_upright"])
            tus.append(info["torso_upright"])
            coms.append(info["com_err"])
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
        mean_ph.append(np.mean(phs)); mean_hu.append(np.mean(hus))
        mean_tu.append(np.mean(tus)); mean_com.append(np.mean(coms))
        status = "摔倒" if fell else "站稳全程"
        print(f"  ep{ep+1}: 存活 {ep_len:4d}/{args.max_steps} 步  "
              f"骨盆高 {np.mean(phs):.3f}m  头直立 {np.mean(hus):.3f}  "
              f"躯干直立 {np.mean(tus):.3f}  质心偏移 {np.mean(coms):.3f}m  → {status}")

    print("=" * 60)
    print(f"平均存活: {np.mean(lengths):.0f}/{args.max_steps} 步   "
          f"全程站稳率: {np.mean(survived):.0%}")
    print(f"骨盆高度: {np.mean(mean_ph):.3f}m (目标 {raw_env.init_pelvis_height})   "
          f"头部直立: {np.mean(mean_hu):.3f}   躯干直立: {np.mean(mean_tu):.3f}   "
          f"质心偏移: {np.mean(mean_com):.3f}m")
    print("=" * 60)

    if args.gif and frames:
        import imageio
        imageio.mimsave(args.gif, frames, fps=30)
        print(f"演示动画已保存: {args.gif} ({len(frames)} 帧)")

    venv.close()


if __name__ == "__main__":
    main()
