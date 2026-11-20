#!/usr/bin/env python3
"""评估 PD 行走模型，并输出步态流畅度/对称性指标

用法:
    python -m humanoid_rl.full_body.eval_walk_pd --model checkpoints/full_body_walk_pd/walk_pd
    python -m humanoid_rl.full_body.eval_walk_pd --model ... --gif walk_pd.gif
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from humanoid_rl.full_body.walk_env_pd import FullBodyWalkPDEnv


def gait_metrics(rows, dt):
    """rows: 每步的 info。返回步频、步长、对称性、平滑度等指标。"""
    on_r = np.array([r["contact_r"] for r in rows])
    on_l = np.array([r["contact_l"] for r in rows])
    xr = np.array([r["foot_x_r"] for r in rows])
    xl = np.array([r["foot_x_l"] for r in rows])
    out = {}
    for side, on, x_self, x_other in (("r", on_r, xr, xl), ("l", on_l, xl, xr)):
        strikes = np.where((on[1:] > 0.5) & (on[:-1] < 0.5))[0] + 1   # 离地 → 着地
        dur = len(rows) * dt
        out[f"step_freq_{side}"] = len(strikes) / dur if dur > 0 else 0.0
        out[f"step_len_{side}"] = float(np.mean(x_self[strikes] - x_other[strikes])) if len(strikes) else 0.0
    f_r, f_l = out["step_freq_r"], out["step_freq_l"]
    s_r, s_l = out["step_len_r"], out["step_len_l"]
    out["freq_ratio_R/L"] = f_r / f_l if f_l > 0 else float("nan")
    out["len_ratio_R/L"] = s_r / s_l if abs(s_l) > 1e-6 else float("nan")
    out["action_rate"] = float(np.mean([r["action_rate"] for r in rows]))
    out["track_err"] = float(np.mean([r["track_err"] for r in rows]))
    out["r_contact"] = float(np.mean([r["r_contact"] for r in rows]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="checkpoints/full_body_walk_pd/walk_pd")
    p.add_argument("--target-speed", type=float, default=0.5)
    p.add_argument("--gait-freq", type=float, default=1.0)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--gif", type=str, default="")
    args = p.parse_args()

    render_mode = "rgb_array" if args.gif else None
    venv = DummyVecEnv([lambda: FullBodyWalkPDEnv(render_mode=render_mode,
                                                  target_speed=args.target_speed,
                                                  gait_freq=args.gait_freq)])
    vn = f"{args.model}_vecnorm.pkl"
    if os.path.exists(vn):
        venv = VecNormalize.load(vn, venv)
        venv.training, venv.norm_reward = False, False
    model = PPO.load(args.model, device="cpu")
    raw = venv.venv.envs[0].unwrapped if hasattr(venv, "venv") else venv.envs[0].unwrapped

    frames, summary = [], []
    for ep in range(args.episodes):
        obs = venv.reset()
        rows = []
        for t in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = venv.step(action)
            rows.append(infos[0])
            if args.gif and ep == 0 and t % 2 == 0:   # 隔帧取图，gif 按真实速度播放
                f = raw.render()
                if f is not None:
                    frames.append(f)
            if done[0]:
                break
        m = gait_metrics(rows, raw.dt)
        m["steps"] = len(rows)
        m["distance"] = rows[-1]["distance"]
        m["mean_vel"] = float(np.mean([r["forward_vel"] for r in rows]))
        summary.append(m)
        print(f"ep{ep + 1}: 存活 {len(rows)} 步  前进 {m['distance']:.2f}m  均速 {m['mean_vel']:.2f}  "
              f"步频 R/L {m['step_freq_r']:.2f}/{m['step_freq_l']:.2f}Hz  "
              f"步长 R/L {m['step_len_r']:.2f}/{m['step_len_l']:.2f}m  "
              f"动作变化率 {m['action_rate']:.3f}  跟踪误差 {m['track_err']:.3f}  "
              f"接触时序 {m['r_contact']:.2f}")
    keys = ["steps", "distance", "mean_vel", "freq_ratio_R/L", "len_ratio_R/L",
            "action_rate", "track_err", "r_contact"]
    print("=" * 60)
    for k in keys:
        print(f"{k:16s} {np.nanmean([s[k] for s in summary]):.3f}")
    if args.gif and frames:
        import imageio
        imageio.mimsave(args.gif, frames, fps=int(round(1 / raw.dt / 2)))
        print(f"已保存 {args.gif}")


if __name__ == "__main__":
    main()
