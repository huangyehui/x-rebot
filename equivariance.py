#!/usr/bin/env python3
"""策略镜像等变误差：在策略自己走出来的状态上，比较 π(s) 与 P·π(M·s)。

如果策略严格镜像对称，二者相等，镜像手术(任何 λ)都不改变行为。
误差越大，手术引入的扰动越大。

    err      = 平均 |π(s) − P·π(M·s)|，按执行器分组(腿/腰/臂/颈)
    rel      = err / 平均 |π(s)|
    ood      = 镜像状态 M·s 在 VecNormalize 统计下的平均 |z|，对比原状态，
               衡量镜像状态是否落在训练分布之外

用法:
    python -m humanoid_rl.full_body.equivariance --models 名字=路径 ...
"""
import argparse
import pickle

import numpy as np

GROUPS = {"右腿": [0, 1, 2], "左腿": [3, 4, 5], "腰": [6, 7], "臂": list(range(8, 14)), "颈": [14, 15]}


def measure(path, steps=1500, seed=777):
    import torch as th
    from humanoid_rl.full_body.gait_diag import load
    from humanoid_rl.full_body.weight_surgery import obs_mirror_matrix, act_mirror_matrix
    model, venv = load(path)
    env = venv.venv.envs[0].unwrapped
    vn = pickle.load(open(f"{path}_vecnorm.pkl", "rb"))
    mu, sd = vn.obs_rms.mean, np.sqrt(vn.obs_rms.var + vn.epsilon)
    M, P = obs_mirror_matrix(), act_mirror_matrix()

    def pi(raw):
        z = np.clip((raw - mu) / sd, -vn.clip_obs, vn.clip_obs)
        with th.no_grad():
            a = model.policy._predict(th.tensor(z[None], dtype=th.float32), deterministic=True)
        return np.clip(a.numpy()[0], -1, 1)

    venv.seed(seed)
    obs = venv.reset()
    A, Am, Z, Zm = [], [], [], []
    for i in range(steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = venv.step(a)
        if i < 60:                      # 跳过起步瞬态
            continue
        s = env._get_obs()
        A.append(pi(s)); Am.append(P @ pi(M @ s))
        Z.append(np.abs((s - mu) / sd).mean()); Zm.append(np.abs((M @ s - mu) / sd).mean())
        if done[0]:
            obs = venv.reset()
    A, Am = np.array(A), np.array(Am)
    diff = np.abs(A - Am)
    out = {g: float(diff[:, idx].mean()) for g, idx in GROUPS.items()}
    out["全部"] = float(diff.mean())
    out["rel"] = float(diff.mean() / np.abs(A).mean())
    out["z"] = float(np.mean(Z)); out["z_mir"] = float(np.mean(Zm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    print(f"{'模型':12s} " + " ".join(f"{g:>6s}" for g in [*GROUPS, "全部"]) +
          f" {'相对':>6s} | {'|z|原':>6s} {'|z|镜像':>7s}")
    for spec in a.models:
        name, path = spec.split("=", 1)
        r = measure(path)
        print(f"{name:12s} " + " ".join(f"{r[g]:6.3f}" for g in [*GROUPS, "全部"]) +
              f" {r['rel']:6.2f} | {r['z']:6.2f} {r['z_mir']:7.2f}")


if __name__ == "__main__":
    main()
