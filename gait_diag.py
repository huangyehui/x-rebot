#!/usr/bin/env python3
"""左右腿步态对比诊断：量化"卡顿/拖行"。

指标(左右腿分别统计):
  clear   摆动期平均离地高度(m)，越小越像拖步
  drag    脚贴地(z<阈值)时的水平滑移速度(m/s)，越大越像拖着走
  steps   每秒抬脚次数(Hz)
  chat    动作高频能量(二阶差分均方)，越大越"卡顿/哆嗦"
  hipROM  髋关节摆幅(rad)

用法:
    python -m humanoid_rl.full_body.gait_diag --model checkpoints/full_body_walk_robust3/walk
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from humanoid_rl.full_body.walk_env import FullBodyWalkEnv

# 执行器索引: 0 r_hip 1 r_knee 2 r_ankle 3 l_hip 4 l_knee 5 l_ankle
LEG_ACT = {"R": [0, 1, 2], "L": [3, 4, 5]}
GROUND_Z = 0.088   # 脚 body 原点贴地时约 0.073~0.083(实测)


def load(model_path, target_speed=0.5):
    venv = DummyVecEnv([lambda: FullBodyWalkEnv(target_speed=target_speed)])
    vn = f"{model_path}_vecnorm.pkl"
    if os.path.exists(vn):
        venv = VecNormalize.load(vn, venv)
        venv.training = False
        venv.norm_reward = False
    model = PPO.load(model_path, device="cpu")
    return model, venv


def run(model, venv, episodes=3, max_steps=1000, seed=0, policy_fn=None):
    """policy_fn(obs)->action 可替换策略(用于评估改过的网络)。"""
    env = venv.venv.envs[0].unwrapped if hasattr(venv, "venv") else venv.envs[0].unwrapped
    venv.seed(seed)
    stats = {k: [] for k in ["len", "dist", "vel",
                             "R_clear", "L_clear", "R_drag", "L_drag",
                             "R_steps", "L_steps", "R_chat", "L_chat",
                             "R_hipROM", "L_hipROM"]}
    for ep in range(episodes):
        obs = venv.reset()
        acts, z, vx, hip = [], {"R": [], "L": []}, {"R": [], "L": []}, {"R": [], "L": []}
        prev = {"R": env.data.xpos[env._rfoot_id].copy(), "L": env.data.xpos[env._lfoot_id].copy()}
        n = 0
        for _ in range(max_steps):
            if policy_fn is None:
                a, _ = model.predict(obs, deterministic=True)
            else:
                a = policy_fn(obs)
            obs, _, done, infos = venv.step(a)
            acts.append(np.clip(a[0], -1, 1))
            for s, fid, hadr in (("R", env._rfoot_id, env._rhip_adr),
                                 ("L", env._lfoot_id, env._lhip_adr)):
                p = env.data.xpos[fid].copy()
                z[s].append(p[2])
                vx[s].append(np.linalg.norm(p[:2] - prev[s][:2]) / env.dt)
                prev[s] = p
                hip[s].append(env.data.qpos[hadr])
            n += 1
            if done[0]:
                break
        dist = infos[0]["distance"]
        stats["len"].append(n)
        stats["dist"].append(dist)
        stats["vel"].append(dist / (n * env.dt))
        A = np.array(acts)
        jerk = A[2:] - 2 * A[1:-1] + A[:-2]
        T = n * env.dt
        for s in ("R", "L"):
            zz = np.array(z[s]); vv = np.array(vx[s])
            on = zz < GROUND_Z
            stats[f"{s}_clear"].append(float(zz[~on].mean() - GROUND_Z) if (~on).any() else 0.0)
            stats[f"{s}_drag"].append(float(vv[on].mean()) if on.any() else 0.0)
            # 抬脚次数：贴地->离地 的上升沿
            lifts = np.sum((~on[1:]) & on[:-1])
            stats[f"{s}_steps"].append(lifts / T)
            stats[f"{s}_chat"].append(float((jerk[:, LEG_ACT[s]] ** 2).mean()))
            h = np.array(hip[s])
            stats[f"{s}_hipROM"].append(float(np.percentile(h, 95) - np.percentile(h, 5)))
    return {k: float(np.mean(v)) for k, v in stats.items()}


def show(tag, s):
    print(f"[{tag}] len {s['len']:.0f}  dist {s['dist']:.2f}m  vel {s['vel']:.3f}")
    for k in ["clear", "drag", "steps", "chat", "hipROM"]:
        print(f"    {k:7s}  R {s['R_' + k]:.4f}   L {s['L_' + k]:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/full_body_walk_robust3/walk")
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()
    model, venv = load(args.model)
    show(os.path.basename(args.model), run(model, venv, args.episodes))


if __name__ == "__main__":
    main()
