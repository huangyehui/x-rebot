#!/usr/bin/env python3
"""搜索式手术：在镜像手术的编辑空间里用 CMA-ES 搜索，只做前向评估、不训练。

问题：允许大量闭环评估时，能否零训练修好 robust3 的右腿(不摔 + 左右对称)？

编辑空间(只改输出层，隐藏层保持"原网络 + 镜像副本"不变):
    h = [h_orig ; h_mir]  (512 维)
    a_j = (1-λ_j)(W3[j]·h_o + b3[j]) + λ_j (PW3[j]·h_m + Pb3[j])
          + Δb_j + U[j] · B^T (h - h̄)
  λ_j   : 16 个执行器各一个镜像混合系数 ∈[0,1]
  Δb_j  : 16 个输出偏置修正
  U     : 16×k 低秩修正，B 为正常行走时 h 的前 k 个主成分(固定)
  共 32 + 16k 维。k=8 时 160 维。

适应度(越大越好，同一代所有候选用相同随机种子，降低噪声):
    surv = 存活步数/1000
    score = 2·surv + 0.5·min(距离/7, 1)
            + surv · (1 − |f_R − f_L|/f_L − (1 − min(髋R,髋L)/max(髋R,髋L)))
  只有站得住时对称项才有分量，避免奖励"摔倒前的对称"。

用法:
    python -m humanoid_rl.full_body.surgery_search --gens 150 --popsize 28
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

SRC = "checkpoints/full_body_walk_robust3/walk"
NA, H = 16, 256
RIGHT_LEG = [0, 1, 2]

_G = {}


# ---------- 组网 ----------
def base_parts():
    """返回隐藏层(512 宽)权重和输出层两路的原始/镜像参数，均在归一化观测空间。"""
    from humanoid_rl.full_body.weight_surgery import build, act_mirror_matrix
    new, _, _ = build(lam=(0, 0, 0))
    sd = {k: v.detach().double().numpy() for k, v in new.policy.state_dict().items()}
    W3o = sd["action_net.weight"][:, :H]          # λ=0 时就是原网络 W3
    b3o = sd["action_net.bias"]
    P = act_mirror_matrix()
    return {
        "W1": sd["mlp_extractor.policy_net.0.weight"], "b1": sd["mlp_extractor.policy_net.0.bias"],
        "W2": sd["mlp_extractor.policy_net.2.weight"], "b2": sd["mlp_extractor.policy_net.2.bias"],
        "W3o": W3o, "b3o": b3o, "W3m": P @ W3o, "b3m": P @ b3o,
    }, new


def hidden(p, z):
    h = np.tanh(p["W1"] @ z + p["b1"])
    return np.tanh(p["W2"] @ h + p["b2"])


def output_layer(p, B, hbar, k, x):
    """把编辑向量 x 解码成完整输出层 (W3: 16×512, b3: 16)。"""
    lam, db, U = decode(x, k)
    W3 = np.zeros((NA, 2 * H)); b3 = np.zeros(NA)
    W3[:, :H] = (1 - lam)[:, None] * p["W3o"]
    W3[:, H:] = lam[:, None] * p["W3m"]
    b3[:] = (1 - lam) * p["b3o"] + lam * p["b3m"] + db
    # 低秩修正: U · B^T (h - h̄)  =>  W3 += U B^T,  b3 -= U B^T h̄
    UB = U @ B.T
    W3 += UB
    b3 -= UB @ hbar
    return W3, b3


def decode(x, k):
    lam = np.clip(_G["lam0"] + x[:NA], 0.0, 1.0)
    db = 0.3 * x[NA:2 * NA]
    U = 0.3 * x[2 * NA:].reshape(NA, k) / _G["pc_std"][None, :k] if k else np.zeros((NA, 0))
    return lam, db, U


# ---------- 评估 ----------
def fitness(s):
    """所有项都乘以前进进度 prog：站着不动 = 0 分。

    第一版只用存活 + 对称性打分，CMA-ES 很快找到"原地站着、两腿都不动"
    (频率 0=0、摆幅相等，对称项满分)，得分 ≈3，高于正常行走。这里堵掉这个漏洞:
      - 对称项只在前进时计分(乘 prog)
      - 任一条腿抬脚频率 <0.5Hz 视为没在迈步，扣 0.5
    参考值: 原 robust3 ≈1.6，ft4 ≈2.8，站立 ≈0。
    """
    surv = s["len"] / 1000.0
    prog = float(np.clip(s["dist"] / 7.0, 0.0, 1.0))
    fl = max(s["L_steps"], 0.5)
    freq_err = min(abs(s["R_steps"] - s["L_steps"]) / fl, 1.0)
    hr, hl = s["R_hipROM"], s["L_hipROM"]
    hip_err = 1.0 - min(hr, hl) / max(hr, hl, 1e-6)
    no_step = 0.5 if min(s["R_steps"], s["L_steps"]) < 0.5 else 0.0
    return 2.0 * surv * prog + prog * (1.0 - freq_err - hip_err - no_step)


def _init(state):
    import torch
    torch.set_num_threads(1)
    _G.update(state)
    from humanoid_rl.full_body.gait_diag import load
    _, venv = load(SRC)
    _G["venv"] = venv


def _eval(job):
    from humanoid_rl.full_body.gait_diag import run
    x, seed, episodes = job
    p, k = _G["p"], _G["k"]
    W3, b3 = output_layer(p, _G["B"], _G["hbar"], k, np.asarray(x))

    def fn(obs):
        a = W3 @ hidden(p, obs[0]) + b3
        return np.clip(a, -1, 1)[None, :].astype(np.float32)

    s = run(None, _G["venv"], episodes, seed=seed, policy_fn=fn)
    return fitness(s), s


def collect_pca(p, steps=3000):
    """在原网络(λ=0)正常行走下收集 512 维隐藏激活，做 PCA。"""
    from humanoid_rl.full_body.gait_diag import load
    _, venv = load(SRC)
    venv.seed(7)
    obs = venv.reset()
    Hs = []
    for _ in range(steps):
        h = hidden(p, obs[0])
        Hs.append(h)
        a = p["W3o"] @ h[:H] + p["b3o"]
        obs, _, done, _ = venv.step(np.clip(a, -1, 1)[None, :].astype(np.float32))
    Hs = np.array(Hs)
    hbar = Hs.mean(0)
    _, S, Vt = np.linalg.svd(Hs - hbar, full_matrices=False)
    std = S / np.sqrt(len(Hs))
    return Vt.T, hbar, std, float((S[:8] ** 2).sum() / (S ** 2).sum())


def export(new_model, W3, b3, out):
    """把编辑结果写成标准 SB3 模型(512 宽策略网络)。"""
    import shutil
    import torch as th
    sd = new_model.policy.state_dict()
    sd["action_net.weight"] = th.tensor(W3, dtype=th.float32)
    sd["action_net.bias"] = th.tensor(b3, dtype=th.float32)
    new_model.policy.load_state_dict(sd)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new_model.save(out)
    shutil.copy(f"{SRC}_vecnorm.pkl", f"{out}_vecnorm.pkl")


def main():
    import cma
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="低秩修正维数(0=只搜 λ 和偏置)")
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--popsize", type=int, default=28)
    ap.add_argument("--episodes", type=int, default=2, help="每个候选评估回合数")
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--lam0-right", type=float, default=0.5,
                    help="右腿 λ 起点：0.5=原手术(会摔)，0=原 robust3(能走但不对称)")
    ap.add_argument("--out", default="checkpoints/surgery_search")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    p, new_model = base_parts()
    B, hbar, pc_std, var8 = collect_pca(p)
    lam0 = np.zeros(NA); lam0[RIGHT_LEG] = a.lam0_right
    state = {"p": p, "B": B[:, :a.k], "hbar": hbar, "pc_std": pc_std, "k": a.k, "lam0": lam0}
    _G.update(state)
    dim = 2 * NA + NA * a.k
    print(f"编辑空间 {dim} 维 (k={a.k}, 前 8 个主成分解释方差 {var8:.1%})")

    es = cma.CMAEvolutionStrategy(np.zeros(dim), a.sigma,
                                  {"popsize": a.popsize, "seed": 0, "verbose": -9})
    log, best = [], (-np.inf, None, None)
    env_steps, t0 = 0, time.time()
    with Pool(a.workers, initializer=_init, initargs=(state,)) as pool:
        # 起点基线
        f0, s0 = pool.map(_eval, [(np.zeros(dim), 10_000, a.episodes)])[0]
        print(f"起点(λ_R={a.lam0_right}): fit {f0:.3f}  len {s0['len']:.0f}  "
              f"抬脚 R/L {s0['R_steps']:.2f}/{s0['L_steps']:.2f}")
        for g in range(a.gens):
            X = es.ask()
            seed = 1000 + g                                  # 同代共用种子(配对比较)
            R = pool.map(_eval, [(x, seed, a.episodes) for x in X])
            fits = [r[0] for r in R]
            es.tell(X, [-f for f in fits])
            env_steps += sum(r[1]["len"] * a.episodes for r in R)
            i = int(np.argmax(fits))
            if fits[i] > best[0]:
                best = (fits[i], np.array(X[i]), R[i][1])
            s = R[i][1]
            log.append({"gen": g, "best": fits[i], "mean": float(np.mean(fits)),
                        "len": s["len"], "R_steps": s["R_steps"], "L_steps": s["L_steps"],
                        "R_hip": s["R_hipROM"], "L_hip": s["L_hipROM"], "env_steps": env_steps})
            if g % 5 == 0 or g == a.gens - 1:
                print(f"gen {g:3d}  best {fits[i]:.3f} mean {np.mean(fits):.3f} | len {s['len']:4.0f} "
                      f"dist {s['dist']:4.1f} | 抬脚 R/L {s['R_steps']:.2f}/{s['L_steps']:.2f} "
                      f"| 髋 R/L {s['R_hipROM']:.2f}/{s['L_hipROM']:.2f} | 交互 {env_steps / 1e6:.2f}M "
                      f"| {time.time() - t0:.0f}s", flush=True)

        # 最优解在 10 个全新种子上复评，防止过拟合到搜索时的种子
        xb = best[1]
        final = pool.map(_eval, [(xb, 50_000 + i, 1) for i in range(10)])

    keys = ["len", "dist", "R_steps", "L_steps", "R_hipROM", "L_hipROM", "R_clear", "L_clear", "R_chat"]
    fs = {k: float(np.mean([r[1][k] for r in final])) for k in keys}
    print("\n最优编辑在 10 个新种子上:")
    print("  " + "  ".join(f"{k} {fs[k]:.3f}" for k in keys))
    lam, db, U = decode(xb, a.k)
    print(f"  λ = {np.round(lam, 2)}")
    print(f"  总交互步数 {env_steps / 1e6:.2f}M")

    W3, b3 = output_layer(p, B[:, :a.k], hbar, a.k, xb)
    export(new_model, W3, b3, os.path.join(a.out, "walk"))
    json.dump({"log": log, "best_x": xb.tolist(), "final": fs, "lam": lam.tolist(),
               "env_steps": env_steps, "args": vars(a)},
              open(os.path.join(a.out, "search.json"), "w"), ensure_ascii=False, indent=1)
    print(f"已导出 {a.out}/walk.zip")


if __name__ == "__main__":
    main()
