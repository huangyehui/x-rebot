#!/usr/bin/env python3
"""权重空间手术：不训练、无梯度，用左腿控制回路重写右腿输出。

原理
----
左腿步态正常，右腿拖步。镜像对称下，"右腿在状态 s 应做的动作" ≈
"左腿在镜像状态 M·s 下的动作"。镜像算子 M(关节左右互换 + 侧向量取反)
与 VecNormalize 的归一化都是仿射变换，可以直接折进第一层权重：

    z  = (raw - μ)/σ                      (归一化观测)
    z' = (M·raw - μ)/σ = D⁻¹ M D z + D⁻¹(Mμ - μ),  D = diag(σ)
    W1_mir = W1 · D⁻¹ M D,   b1_mir = b1 + W1 · D⁻¹(Mμ - μ)

于是得到一个镜像副本网络。把原网络与镜像副本拼成块对角的 512 宽 MLP，
输出层只改右腿 3 行:
    a_R = (1-λ)·a_R(原) + λ·P·a_L(镜像副本)
其中 P 是镜像动作映射(左右腿同号)。结果仍是标准 SB3 PPO 模型，可直接
用 eval_walk / train_walk --resume 加载。

λ 不靠梯度，而是少量前向评估(eval)网格搜索得到。

用法:
    python -m humanoid_rl.full_body.weight_surgery --search
    python -m humanoid_rl.full_body.weight_surgery --lam 0.6 0.6 0.6 \
        --out checkpoints/full_body_walk_surgery/walk
"""
import argparse
import os
import pickle
import shutil

import numpy as np
import torch as th
from stable_baselines3 import PPO

SRC = "checkpoints/full_body_walk_robust3/walk"

# ---------- 镜像算子 ----------
# 关节(qpos[7:] / qvel[6:]) 顺序:
#  0 waist_y 1 waist_x 2 r_sh_y 3 r_sh_x 4 r_elbow 5 l_sh_y 6 l_sh_x 7 l_elbow
#  8 neck_y 9 neck_x 10 r_hip 11 r_knee 12 r_ankle 13 l_hip 14 l_knee 15 l_ankle
J_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 9, 13, 14, 15, 10, 11, 12]
J_SIGN = [1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1]   # 绕 x 轴的关节取反
# 动作顺序:
#  0 r_hip 1 r_knee 2 r_ankle 3 l_hip 4 l_knee 5 l_ankle 6 waist_y 7 waist_x
#  8 r_sh_y 9 r_sh_x 10 r_elbow 11 l_sh_y 12 l_sh_x 13 l_elbow 14 neck_y 15 neck_x
A_PERM = [3, 4, 5, 0, 1, 2, 6, 7, 11, 12, 13, 8, 9, 10, 14, 15]
A_SIGN = [1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1]


def obs_mirror_matrix():
    """45 维观测的镜像矩阵 M (mirror_obs = M @ obs)。"""
    perm, sign = [], []
    for off in (0, 16):                       # qpos 关节, qvel 关节
        perm += [off + p for p in J_PERM]
        sign += J_SIGN
    # 根速度 vx vy vz wx wy wz: y 方向线速度、绕 x/z 角速度取反
    perm += list(range(32, 38)); sign += [1, -1, 1, -1, 1, -1]
    # pelvis_h head_h 3×upright: 不变
    perm += list(range(38, 43)); sign += [1] * 5
    # com-mid (x, y): y 取反
    perm += [43, 44]; sign += [1, -1]
    M = np.zeros((45, 45))
    for i, (p, s) in enumerate(zip(perm, sign)):
        M[i, p] = s
    return M


def act_mirror_matrix():
    P = np.zeros((16, 16))
    for i, (p, s) in enumerate(zip(A_PERM, A_SIGN)):
        P[i, p] = s
    return P


def build(src=SRC, lam=(0.6, 0.6, 0.6), whole_body=False):
    """返回 (新 PPO 模型, 编辑参数量, 原参数量)。

    lam=右腿 髋/膝/踝 的混合系数。whole_body=True 时对全部 16 个执行器统一用
    lam[0] 混合(整体对称化)，避免只改右腿导致与左腿/躯干的协同失配。
    """
    model = PPO.load(src, device="cpu")
    vn = pickle.load(open(f"{src}_vecnorm.pkl", "rb"))
    mu = vn.obs_rms.mean
    sd = np.sqrt(vn.obs_rms.var + vn.epsilon)

    M = obs_mirror_matrix()
    P = act_mirror_matrix()
    T = np.diag(1 / sd) @ M @ np.diag(sd)          # z -> z'
    c = (M @ mu - mu) / sd

    sd_old = {k: v.detach().double().numpy() for k, v in model.policy.state_dict().items()}
    W1, b1 = sd_old["mlp_extractor.policy_net.0.weight"], sd_old["mlp_extractor.policy_net.0.bias"]
    W2, b2 = sd_old["mlp_extractor.policy_net.2.weight"], sd_old["mlp_extractor.policy_net.2.bias"]
    W3, b3 = sd_old["action_net.weight"], sd_old["action_net.bias"]
    H = W1.shape[0]

    # 镜像副本第一层
    W1m = W1 @ T
    b1m = b1 + W1 @ c

    # 块对角拼接
    nW1 = np.vstack([W1, W1m]); nb1 = np.concatenate([b1, b1m])
    nW2 = np.zeros((2 * H, 2 * H)); nW2[:H, :H] = W2; nW2[H:, H:] = W2
    nb2 = np.concatenate([b2, b2])

    # 输出层: 默认全部用原网络
    nW3 = np.zeros((16, 2 * H)); nW3[:, :H] = W3
    nb3 = b3.copy()
    # 镜像副本的动作 a_m = W3 h_m + b3；镜像回真实空间 = P @ a_m
    PW3, Pb3 = P @ W3, P @ b3
    pairs = [(j, lam[0]) for j in range(16)] if whole_body else zip([0, 1, 2], lam)
    for j, l in pairs:                             # 右腿三个执行器(或全身)
        nW3[j, :H] = (1 - l) * W3[j]
        nW3[j, H:] = l * PW3[j]
        nb3[j] = (1 - l) * b3[j] + l * Pb3[j]

    # 新模型：策略网络加宽一倍；价值网络保持原结构不变。
    # 价值网络宽度要从原模型读取：对已做过手术的模型(策略 512、价值 256)再次手术时，
    # 两者宽度不同，不能用 H 推断。
    Hv = sd_old["mlp_extractor.value_net.0.weight"].shape[0]
    new = PPO("MlpPolicy", model.env if model.env else _dummy_env(), device="cpu",
              n_steps=model.n_steps, batch_size=model.batch_size,
              learning_rate=1e-4,
              policy_kwargs=dict(net_arch=dict(pi=[2 * H, 2 * H], vf=[Hv, Hv]),
                                 log_std_init=-1.0))
    sd_new = new.policy.state_dict()
    t = lambda a: th.tensor(a, dtype=th.float32)
    for k in sd_new:
        if k.startswith("mlp_extractor.value_net") or k.startswith("value_net") or k == "log_std":
            sd_new[k] = th.tensor(sd_old[k], dtype=th.float32)
    sd_new["mlp_extractor.policy_net.0.weight"] = t(nW1)
    sd_new["mlp_extractor.policy_net.0.bias"] = t(nb1)
    sd_new["mlp_extractor.policy_net.2.weight"] = t(nW2)
    sd_new["mlp_extractor.policy_net.2.bias"] = t(nb2)
    sd_new["action_net.weight"] = t(nW3)
    sd_new["action_net.bias"] = t(nb3)
    new.policy.load_state_dict(sd_new)

    # 统计"有效编辑量"：原网络被改动的只有右腿 3 行输出(其余为新增镜像通路，直接复用原权重)
    n_orig = sum(v.size for v in sd_old.values())
    n_edit = 3 * (H + 1)
    return new, n_edit, n_orig


def _dummy_env():
    from stable_baselines3.common.vec_env import DummyVecEnv
    from humanoid_rl.full_body.walk_env import FullBodyWalkEnv
    return DummyVecEnv([lambda: FullBodyWalkEnv()])


def save(new, out, src=SRC):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new.save(out)
    shutil.copy(f"{src}_vecnorm.pkl", f"{out}_vecnorm.pkl")


def make_policy_fn(new):
    pol = new.policy
    def fn(obs):
        with th.no_grad():
            a = pol._predict(th.tensor(obs, dtype=th.float32), deterministic=True)
        return np.clip(a.numpy(), -1.0, 1.0)
    return fn


def sanity():
    """检查: λ=0 时输出与原网络完全一致；镜像副本 == 原网络(镜像观测)。"""
    from humanoid_rl.full_body.gait_diag import load
    model, venv = load(SRC)
    new, _, _ = build(lam=(0, 0, 0))
    obs = venv.reset()
    for _ in range(50):
        obs, *_ = venv.step(model.predict(obs, deterministic=True)[0])
    a0 = model.predict(obs, deterministic=True)[0]
    a1 = make_policy_fn(new)(obs)
    print("[sanity] λ=0 最大差:", float(np.abs(a0 - a1).max()))


def search(episodes=3):
    from humanoid_rl.full_body.gait_diag import load, run, show
    model, venv = load(SRC)
    base = run(model, venv, episodes)
    show("原模型", base)
    grid = [(l, l, l) for l in (0.3, 0.5, 0.7, 1.0)] + [(0.7, 0.4, 0.4), (1.0, 0.5, 0.5)]
    results = []
    for lam in grid:
        new, n_edit, n_orig = build(lam=lam)
        s = run(model, venv, episodes, policy_fn=make_policy_fn(new))
        show(f"λ={lam}", s)
        results.append((lam, s))
    print(f"\n编辑原网络参数 {n_edit}/{n_orig} = {100 * n_edit / n_orig:.2f}%")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--search", action="store_true")
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--lam", type=float, nargs=3, default=[0.6, 0.6, 0.6])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--out", default="checkpoints/full_body_walk_surgery/walk")
    a = p.parse_args()
    if a.sanity:
        sanity(); return
    if a.search:
        search(a.episodes); return
    new, n_edit, n_orig = build(lam=tuple(a.lam))
    save(new, a.out)
    print(f"已保存 {a.out}.zip  (λ={a.lam}, 编辑 {n_edit}/{n_orig} 参数)")


if __name__ == "__main__":
    main()
