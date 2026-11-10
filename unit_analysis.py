#!/usr/bin/env python3
"""隐藏单元功能定位：权重归属 + 成组消融(闭环仿真)。

问题：策略网络隐藏层里是否存在"主要服务右腿、与平衡无关"的单元？

1. 权重归属(相关性证据)
   第二层单元 k:  r_k = Σ_{j∈右腿}|W3[j,k]| / Σ_j |W3[j,k]|
   第一层单元 k:  用路径权重 P = |W3|·|W2| 代替 |W3| 计算同样的比例
   左腿占比 l_k 同理。均匀分布时基线约为 3/16 ≈ 0.19。

2. 成组消融(因果证据)
   均值消融：把单元输出固定为其在正常行走中的平均激活，等价于
       b_next += W_next[:,K] @ mean[K];  W_next[:,K] = 0
   按 r_k 取前 N 个单元消融，与随机 N 个单元、左腿前 N 个单元对比，
   闭环仿真记录左右腿步态指标与存活步数。所有配置使用相同随机种子(配对比较)。

用法:
    python -m humanoid_rl.full_body.unit_analysis --model checkpoints/full_body_walk_robust3/walk
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

RIGHT = [0, 1, 2]
LEFT = [3, 4, 5]


# ---------- 权重与前向 ----------
def load_weights(model_path):
    from stable_baselines3 import PPO
    m = PPO.load(model_path, device="cpu")
    sd = {k: v.detach().double().numpy() for k, v in m.policy.state_dict().items()}
    return {
        "W1": sd["mlp_extractor.policy_net.0.weight"], "b1": sd["mlp_extractor.policy_net.0.bias"],
        "W2": sd["mlp_extractor.policy_net.2.weight"], "b2": sd["mlp_extractor.policy_net.2.bias"],
        "W3": sd["action_net.weight"], "b3": sd["action_net.bias"],
    }


def forward(w, z):
    h1 = np.tanh(w["W1"] @ z + w["b1"])
    h2 = np.tanh(w["W2"] @ h1 + w["b2"])
    return w["W3"] @ h2 + w["b3"], h1, h2


def ablate(w, layer, units, mean1, mean2):
    """均值消融：返回新权重字典(不改原字典)。layer=1 或 2。"""
    w = {k: v.copy() for k, v in w.items()}
    K = np.asarray(units, dtype=int)
    if len(K) == 0:
        return w
    if layer == 2:
        w["b3"] = w["b3"] + w["W3"][:, K] @ mean2[K]
        w["W3"][:, K] = 0.0
    else:
        w["b2"] = w["b2"] + w["W2"][:, K] @ mean1[K]
        w["W2"][:, K] = 0.0
    return w


def attribution(w):
    A2 = np.abs(w["W3"])                       # 16 × 256
    A1 = np.abs(w["W3"]) @ np.abs(w["W2"])     # 16 × 256 路径权重
    out = {}
    for name, A in (("L2", A2), ("L1", A1)):
        tot = A.sum(0)
        out[name] = {"r": A[RIGHT].sum(0) / tot, "l": A[LEFT].sum(0) / tot}
    return out


# ---------- 仿真 ----------
_G = {}


def _init(model_path):
    import torch
    torch.set_num_threads(1)
    from humanoid_rl.full_body.gait_diag import load
    model, venv = load(model_path)
    _G.update(model=model, venv=venv)


def _eval(job):
    from humanoid_rl.full_body.gait_diag import run
    name, w, episodes = job

    def fn(obs):
        a, _, _ = forward(w, obs[0])
        return np.clip(a, -1, 1)[None, :].astype(np.float32)

    s = run(_G["model"], _G["venv"], episodes, policy_fn=fn)
    return name, s


def collect_means(model_path, w, steps=3000):
    """正常行走中两层隐藏单元的平均激活。"""
    _init(model_path)
    venv = _G["venv"]
    venv.seed(123)
    obs = venv.reset()
    H1, H2 = [], []
    for _ in range(steps):
        a, h1, h2 = forward(w, obs[0])
        H1.append(h1); H2.append(h2)
        obs, _, done, _ = venv.step(np.clip(a, -1, 1)[None, :].astype(np.float32))
    return np.mean(H1, 0), np.mean(H2, 0), np.std(H1, 0), np.std(H2, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/full_body_walk_robust3/walk")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40])
    p.add_argument("--random-reps", type=int, default=3)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", default="checkpoints/unit_analysis")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    w = load_weights(a.model)
    att = attribution(w)
    mean1, mean2, std1, std2 = collect_means(a.model, w)

    # ---- 1. 权重归属分布 ----
    print("=" * 70)
    print("权重归属：右腿占比 r_k 分布(均匀基线 ≈ 0.19)")
    for L in ("L2", "L1"):
        r, l = att[L]["r"], att[L]["l"]
        q = np.percentile(r, [5, 25, 50, 75, 95, 99])
        print(f"  {L}: r_k 分位数 5/25/50/75/95/99 = {np.round(q, 3)}  max {r.max():.3f}")
        print(f"      r_k>0.4: {int((r > 0.4).sum())} 个   r_k>0.5: {int((r > 0.5).sum())} 个"
              f"   corr(r,l) = {np.corrcoef(r, l)[0, 1]:+.3f}")
    # 第二层几乎不活动的单元(std 很小)对消融不敏感，单独报告
    print(f"  L2 激活 std<0.05 的单元: {int((std2 < 0.05).sum())}/256")

    # ---- 2. 成组消融 ----
    rng = np.random.default_rng(0)
    jobs = [("baseline", w, a.episodes)]
    groups = {}
    for L, layer in (("L2", 2), ("L1", 1)):
        r, l = att[L]["r"], att[L]["l"]
        for n in a.sizes:
            groups[f"{L}_topR_{n}"] = (layer, np.argsort(-r)[:n])
            groups[f"{L}_topL_{n}"] = (layer, np.argsort(-l)[:n])
            for rep in range(a.random_reps):
                groups[f"{L}_rand{rep}_{n}"] = (layer, rng.choice(256, n, replace=False))
    for name, (layer, K) in groups.items():
        jobs.append((name, ablate(w, layer, K, mean1, mean2), a.episodes))

    print(f"\n成组消融：{len(jobs)} 个配置 × {a.episodes} 回合，{a.workers} 进程并行 ...")
    with Pool(a.workers, initializer=_init, initargs=(a.model,)) as pool:
        res = dict(pool.map(_eval, jobs))

    keys = ["len", "dist", "R_steps", "L_steps", "R_hipROM", "L_hipROM",
            "R_clear", "L_clear", "R_chat", "L_chat"]
    print(f"\n{'配置':16s} " + " ".join(f"{k:>8s}" for k in keys))
    order = ["baseline"] + sorted(groups, key=lambda s: (s[:2], int(s.split("_")[-1]), s))
    for name in order:
        s = res[name]
        print(f"{name:16s} " + " ".join(f"{s[k]:8.3f}" for k in keys))

    # 随机组取平均，便于对比
    print("\n随机组平均(同层同规模)：")
    for L in ("L2", "L1"):
        for n in a.sizes:
            rs = [res[f"{L}_rand{i}_{n}"] for i in range(a.random_reps)]
            m = {k: np.mean([x[k] for x in rs]) for k in keys}
            print(f"{L}_rand_avg_{n:<4d}   " + " ".join(f"{m[k]:8.3f}" for k in keys))

    json.dump({
        "results": res,
        "groups": {k: [int(v[0]), [int(i) for i in v[1]]] for k, v in groups.items()},
        "attribution": {L: {k: v.tolist() for k, v in att[L].items()} for L in att},
        "std2": std2.tolist(), "std1": std1.tolist(),
    }, open(os.path.join(a.out, "unit_analysis.json"), "w"), ensure_ascii=False, indent=1)

    _plot(att, std2, os.path.join(a.out, "attribution.png"))
    print(f"\n结果: {a.out}/unit_analysis.json  图: {a.out}/attribution.png")


def _plot(att, std2, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import humanoid_rl.full_body.plot_gait_cmp  # noqa: F401  复用中文字体设置
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, L in enumerate(("L2", "L1")):
        ax[i].hist(att[L]["r"], bins=40, color="r", alpha=0.6, label="右腿占比 r_k")
        ax[i].hist(att[L]["l"], bins=40, color="b", alpha=0.6, label="左腿占比 l_k")
        ax[i].axvline(3 / 16, color="k", ls="--", lw=1, label="均匀基线 3/16")
        ax[i].set_title(f"{L} 隐藏单元：腿部权重占比分布"); ax[i].legend()
    ax[2].scatter(att["L2"]["l"], att["L2"]["r"], c=std2, cmap="viridis", s=12)
    ax[2].set_xlabel("左腿占比 l_k"); ax[2].set_ylabel("右腿占比 r_k")
    ax[2].set_title("L2 单元：右腿 vs 左腿占比(颜色=激活 std)")
    fig.tight_layout(); fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
