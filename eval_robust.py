#!/usr/bin/env python3
"""抗扰站立评估：在随机推力下对比模型的平衡鲁棒性

用法:
    python -m humanoid_rl.full_body.eval_robust --model checkpoints/full_body_robust3/stand
    python -m humanoid_rl.full_body.eval_robust --model checkpoints/full_body/stand --push-force 150
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from humanoid_rl.full_body.full_body_env import FullBodyStandEnv


def evaluate(model_path, push_force, episodes, max_steps, push_interval, seed):
    vn_path = f"{model_path}_vecnorm.pkl"

    def make():
        return FullBodyStandEnv(disturb=True, push_force=push_force,
                                push_interval=push_interval, reset_noise=0.05)

    venv = DummyVecEnv([make])
    if os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
        venv.training = False
        venv.norm_reward = False
    model = PPO.load(model_path, device="cpu")

    lengths, survived, com_errs = [], [], []
    for ep in range(episodes):
        venv.seed(seed + ep)
        obs = venv.reset()
        ep_len = 0
        coms = []
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = venv.step(action)
            coms.append(infos[0]["com_err"])
            ep_len += 1
            if done:
                break
        fell = ep_len < max_steps
        lengths.append(ep_len)
        survived.append(0.0 if fell else 1.0)
        com_errs.append(float(np.mean(coms)))
    venv.close()
    return np.mean(lengths), np.mean(survived), np.mean(com_errs)


def main():
    parser = argparse.ArgumentParser(description="抗扰站立评估")
    parser.add_argument("--model", type=str, default="checkpoints/full_body_robust3/stand")
    parser.add_argument("--baseline", type=str, default="checkpoints/full_body/stand",
                        help="对比基准模型(未做抗扰训练的原站立模型)")
    parser.add_argument("--push-force", type=float, default=150.0)
    parser.add_argument("--push-interval", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    print("=" * 64)
    print(f"抗扰评估: 推力 {args.push_force}N, 每 {args.push_interval} 步推一次, "
          f"{args.episodes} 回合")
    print("=" * 64)

    targets = [("抗扰模型", args.model)]
    if args.baseline and os.path.exists(f"{args.baseline}.zip"):
        targets.append(("原始模型", args.baseline))

    for name, path in targets:
        ml, sr, ce = evaluate(path, args.push_force, args.episodes,
                              args.max_steps, args.push_interval, args.seed)
        print(f"  {name:8s} ({os.path.basename(os.path.dirname(path))}): "
              f"平均存活 {ml:6.0f}/{args.max_steps} 步  "
              f"站稳率 {sr:5.0%}  质心偏移 {ce:.3f}m")
    print("=" * 64)


if __name__ == "__main__":
    main()
