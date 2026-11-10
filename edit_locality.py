#!/usr/bin/env python3
"""修复的"局部性"检查：从手术模型到修好的模型，权重改了多少、集中在哪一层。

所有模型先把 VecNormalize 折进第一层，换算到原始观测空间，避免续训时
μ/σ 漂移导致权重不可比：
    W1' = W1 / σ,   b1' = b1 − W1 · (μ / σ)
(忽略 clip_obs=10 的截断，正常行走时几乎不触发。)

1. 逐层相对变化 ‖θ_B − θ_A‖ / ‖θ_A‖
2. 层替换：在手术模型 A 上换入修好模型 B 的某些层，闭环仿真看能否走、步态是否对称。
   如果只换一层就能修好，说明存在"小范围的直接编辑"。

用法:
    python -m humanoid_rl.full_body.edit_locality
"""
import pickle
from itertools import combinations
from multiprocessing import Pool

import numpy as np

A_PATH = "checkpoints/full_body_walk_surgery/walk_l05"            # 手术后、未训练
B_PATH = "checkpoints/full_body_walk_surgery_ft4_hipsym/walk"     # 修好的
LAYERS = ["L1", "L2", "L3"]


def raw_weights(path):
    from stable_baselines3 import PPO
    m = PPO.load(path, device="cpu")
    sd = {k: v.detach().double().numpy() for k, v in m.policy.state_dict().items()}
    vn = pickle.load(open(f"{path}_vecnorm.pkl", "rb"))
    mu, sd_ = vn.obs_rms.mean, np.sqrt(vn.obs_rms.var + vn.epsilon)
    W1 = sd["mlp_extractor.policy_net.0.weight"]
    return {
        "L1": (W1 / sd_, sd["mlp_extractor.policy_net.0.bias"] - W1 @ (mu / sd_)),
        "L2": (sd["mlp_extractor.policy_net.2.weight"], sd["mlp_extractor.policy_net.2.bias"]),
        "L3": (sd["action_net.weight"], sd["action_net.bias"]),
    }


def forward(w, s):
    h = np.tanh(w["L1"][0] @ s + w["L1"][1])
    h = np.tanh(w["L2"][0] @ h + w["L2"][1])
    return w["L3"][0] @ h + w["L3"][1]


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
    fn = lambda obs: np.clip(forward(w, obs[0]), -1, 1)[None, :].astype(np.float32)
    return name, run(None, _G["venv"], 5, policy_fn=fn)


def main():
    A, B = raw_weights(A_PATH), raw_weights(B_PATH)

    print("逐层相对变化 ‖B−A‖/‖A‖ (手术模型 → ft4):")
    tot_d = tot_a = 0.0
    for L in LAYERS:
        d = sum(np.sum((B[L][i] - A[L][i]) ** 2) for i in (0, 1))
        a = sum(np.sum(A[L][i] ** 2) for i in (0, 1))
        tot_d += d; tot_a += a
        n = sum(A[L][i].size for i in (0, 1))
        print(f"  {L}: {np.sqrt(d / a):.3f}   ({n} 参数)")
    print(f"  全部: {np.sqrt(tot_d / tot_a):.3f}")

    jobs = [("A 手术(未训练)", A), ("B ft4", B)]
    for k in (1, 2):
        for Ls in combinations(LAYERS, k):
            w = {L: (B[L] if L in Ls else A[L]) for L in LAYERS}
            jobs.append((f"A + B的{'+'.join(Ls)}", w))
    # 线性插值：A→B 路径上是否连续可走
    for t in (0.25, 0.5, 0.75):
        w = {L: tuple((1 - t) * A[L][i] + t * B[L][i] for i in (0, 1)) for L in LAYERS}
        jobs.append((f"插值 t={t}", w))

    with Pool(min(len(jobs), 14), initializer=_init) as pool:
        res = dict(pool.map(_eval, jobs))

    print(f"\n{'配置':18s} {'存活':>6s} {'距离':>6s} {'抬脚R':>6s} {'抬脚L':>6s} {'髋R':>6s} {'髋L':>6s}")
    for name, _ in jobs:
        s = res[name]
        print(f"{name:18s} {s['len']:6.0f} {s['dist']:6.2f} {s['R_steps']:6.2f} "
              f"{s['L_steps']:6.2f} {s['R_hipROM']:6.2f} {s['L_hipROM']:6.2f}")


if __name__ == "__main__":
    main()
