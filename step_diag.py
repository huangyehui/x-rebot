#!/usr/bin/env python3
"""左右步长诊断：补充 gait_diag 没覆盖的"往前迈多远"。

指标(左右分别):
  step    着地瞬间领先另一只脚的 x 距离(m)，负数 = 落地仍在另一只脚后面
  reach   脚相对骨盆的最大前伸(95 分位, m)
  hip_mu  髋角均值(rad，负 = 前屈)
另外给出存活步数与前进距离。

用法:
    python -m humanoid_rl.full_body.step_diag --models 名字=路径 ... [--episodes 5]
"""
import argparse
from multiprocessing import Pool

import numpy as np

GROUND_Z = 0.09


def evaluate(path, episodes=5, seed=777):
    import torch
    torch.set_num_threads(1)
    from humanoid_rl.full_body.gait_diag import load
    model, venv = load(path)
    env = venv.venv.envs[0].unwrapped
    venv.seed(seed)
    acc = {k: [] for k in ("len", "dist", "R_step", "L_step", "R_reach", "L_reach", "R_hip", "L_hip")}
    for _ in range(episodes):
        obs = venv.reset()
        rec = {"R": [], "L": []}
        pel, n, dist = [], 0, 0.0
        for _ in range(1000):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = venv.step(a)
            rec["R"].append((*env.data.xpos[env._rfoot_id], env.data.qpos[env._rhip_adr]))
            rec["L"].append((*env.data.xpos[env._lfoot_id], env.data.qpos[env._lhip_adr]))
            pel.append(env.data.xpos[env._pelvis_id][0])
            dist = infos[0]["distance"]
            n += 1
            if done[0]:
                break
        R, L, P = np.array(rec["R"])[60:], np.array(rec["L"])[60:], np.array(pel)[60:]
        acc["len"].append(n); acc["dist"].append(dist)
        for s, me, other in (("R", R, L), ("L", L, R)):
            on = me[:, 2] < GROUND_Z
            td = [i for i in range(1, len(on)) if on[i] and not on[i - 1]]
            steps = [me[i, 0] - other[i, 0] for i in td]
            acc[f"{s}_step"].append(float(np.mean(steps)) if steps else np.nan)
            acc[f"{s}_reach"].append(float(np.percentile(me[:, 0] - P, 95)))
            acc[f"{s}_hip"].append(float(me[:, 3].mean()))
    return path, {k: float(np.nanmean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    a = ap.parse_args()
    specs = [m.split("=", 1) for m in a.models]
    with Pool(min(len(specs), 14)) as pool:
        res = dict(pool.starmap(evaluate, [(p, a.episodes) for _, p in specs]))
    print(f"{'模型':20s} {'存活':>5s} {'距离':>6s} | {'步长R':>7s} {'步长L':>7s} | "
          f"{'前伸R':>6s} {'前伸L':>6s} | {'髋均值R':>7s} {'髋均值L':>7s}")
    for name, p in specs:
        s = res[p]
        print(f"{name:20s} {s['len']:5.0f} {s['dist']:6.2f} | {s['R_step']:+7.3f} {s['L_step']:+7.3f} | "
              f"{s['R_reach']:6.3f} {s['L_reach']:6.3f} | {s['R_hip']:+7.2f} {s['L_hip']:+7.2f}")


if __name__ == "__main__":
    main()
