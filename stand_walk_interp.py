#!/usr/bin/env python3
"""站立模型 → 走路模型 的权重空间插值与逐层替换。

两个模型都先把 VecNormalize 折进第一层(原始观测空间)，再做
    θ(t) = (1-t)·θ_stand + t·θ_walk
以及"站立模型换入走路模型的某几层"。闭环仿真看何时开始走。

用法:
    python -m humanoid_rl.full_body.stand_walk_interp
"""
from itertools import combinations
from multiprocessing import Pool

import numpy as np

from humanoid_rl.full_body.edit_locality import raw_weights, forward, LAYERS

STAND = "checkpoints/full_body_robust3/stand"
WALK = "checkpoints/full_body_walk_robust3/walk"

_G = {}


def _init():
    import torch
    torch.set_num_threads(1)
    from stable_baselines3.common.vec_env import DummyVecEnv
    from humanoid_rl.full_body.walk_env import FullBodyWalkEnv
    _G["venv"] = DummyVecEnv([lambda: FullBodyWalkEnv(target_speed=0.5)])


def _eval(job):
    from humanoid_rl.full_body.gait_diag import run
    name, w = job
    # 关掉停滞终止：站立模型不前进也不该被判"失败"，只看是否摔倒
    _G["venv"].envs[0].unwrapped._stall_limit = 10 ** 9
    fn = lambda obs: np.clip(forward(w, obs[0]), -1, 1)[None, :].astype(np.float32)
    return name, run(None, _G["venv"], 5, policy_fn=fn)


def main():
    S, W = raw_weights(STAND), raw_weights(WALK)
    for L in LAYERS:
        d = sum(np.sum((W[L][i] - S[L][i]) ** 2) for i in (0, 1))
        a = sum(np.sum(S[L][i] ** 2) for i in (0, 1))
        print(f"{L} 相对变化 {np.sqrt(d / a):.3f}")

    jobs = []
    for t in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        jobs.append((f"插值 t={t}",
                     {L: tuple((1 - t) * S[L][i] + t * W[L][i] for i in (0, 1)) for L in LAYERS}))
    for k in (1, 2):
        for Ls in combinations(LAYERS, k):
            jobs.append((f"站立+走路的{'+'.join(Ls)}",
                         {L: (W[L] if L in Ls else S[L]) for L in LAYERS}))

    with Pool(min(len(jobs), 14), initializer=_init) as pool:
        res = dict(pool.map(_eval, jobs))

    print(f"\n{'配置':20s} {'存活':>6s} {'距离':>6s} {'速度':>6s} {'抬脚R':>6s} {'抬脚L':>6s} {'髋R':>6s} {'髋L':>6s}")
    for name, _ in jobs:
        s = res[name]
        print(f"{name:20s} {s['len']:6.0f} {s['dist']:6.2f} {s['vel']:6.3f} {s['R_steps']:6.2f} "
              f"{s['L_steps']:6.2f} {s['R_hipROM']:6.2f} {s['L_hipROM']:6.2f}")


if __name__ == "__main__":
    main()
