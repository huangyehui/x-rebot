#!/usr/bin/env python3
"""完整人形站立 PPO 训练（带实时网页可视化）

在稳定下半身基座上叠加躯干+双臂+脖子+头后，验证仍能学会稳定站立。

用法:
    python -m humanoid_rl.full_body.train_full_body                 # 训练 + 网页可视化
    python -m humanoid_rl.full_body.train_full_body --steps 1000000
    python -m humanoid_rl.full_body.train_full_body --no-live       # 关闭可视化
"""
import argparse
import os

import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from humanoid_rl.full_body.full_body_env import FullBodyStandEnv


class PushCurriculumCallback(BaseCallback):
    """课程式推力：训练进度从 0→1，推力从 start_force 线性增到 max_force。

    温和地让策略先适应小扰动再逐步加大，避免一上来大推力把已学能力训崩。
    """

    def __init__(self, start_force, max_force, total_steps, verbose=0):
        super().__init__(verbose)
        self.start_force = start_force
        self.max_force = max_force
        self.total_steps = max(total_steps, 1)

    def _set_all_envs(self, force):
        self.training_env.env_method("set_push_force", force)

    def _on_training_start(self):
        self._set_all_envs(self.start_force)

    def _on_step(self):
        progress = min(1.0, self.num_timesteps / self.total_steps)
        force = self.start_force + progress * (self.max_force - self.start_force)
        # 每 2048 步更新一次即可
        if self.num_timesteps % 2048 == 0:
            self._set_all_envs(force)
        return True


class PeriodicSaveCallback(BaseCallback):
    """每隔 save_freq 步覆盖式保存一次模型 + VecNormalize 统计。

    固定文件名(覆盖)而非带步数后缀，避免占用大量磁盘；模型与其观测归一化
    统计配套保存，保证中途停止后也能正确加载续训/评估。
    """

    def __init__(self, save_freq, save_dir, name="stand", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.name = name
        self._last_save = 0

    def _on_step(self):
        if self.num_timesteps - self._last_save >= self.save_freq:
            self._last_save = self.num_timesteps
            path = os.path.join(self.save_dir, self.name)
            self.model.save(path)
            vecnorm = self.model.get_vec_normalize_env()
            if vecnorm is not None:
                vecnorm.save(f"{path}_vecnorm.pkl")
            if self.verbose:
                print(f"[定期保存] {self.num_timesteps} 步 -> {path}.zip (覆盖)")
        return True


class LiveCallback(BaseCallback):
    """推送训练指标和渲染帧到网页。单环境下才能同步机器人画面。"""

    def __init__(self, live_server, render_env, frame_interval=50, verbose=0):
        super().__init__(verbose)
        self.live_server = live_server
        self.render_env = render_env
        self.frame_interval = frame_interval
        self._ep_reward = 0.0
        self._ep_len = 0
        self._ep_count = 0
        self._frame_ctr = 0

    def _get_train_env(self):
        venv = self.model.get_env()
        while hasattr(venv, "venv"):
            venv = venv.venv
        if hasattr(venv, "envs"):
            return venv.envs[0]
        return None

    def _sync_and_render(self):
        train_env = self._get_train_env()
        if train_env is None or self.render_env is None:
            return
        base = getattr(train_env, "unwrapped", train_env)
        np.copyto(self.render_env.data.qpos, base.data.qpos)
        np.copyto(self.render_env.data.qvel, base.data.qvel)
        mujoco.mj_forward(self.render_env.model, self.render_env.data)
        frame = self.render_env.render()
        if frame is not None:
            self.live_server.push_frame(frame)

    def _on_step(self):
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])

        self._frame_ctr += 1
        if self.live_server and self._frame_ctr % self.frame_interval == 0:
            try:
                self._sync_and_render()
            except Exception:
                pass

        for r, d, info in zip(rewards, dones, infos):
            self._ep_reward += r
            self._ep_len += 1
            if d:
                if self.live_server:
                    self.live_server.push_metrics({
                        "stage": 0,
                        "episode": self._ep_count,
                        "reward": float(self._ep_reward),
                        "length": int(self._ep_len),
                        "timestep": int(self.num_timesteps),
                        "torso_height": float(info.get("pelvis_height", 0.0)),
                        "upright": float(info.get("torso_upright", 0.0)),
                        "com_err": float(info.get("com_err", 0.0)),
                    })
                self._ep_reward = 0.0
                self._ep_len = 0
                self._ep_count += 1
        return True


def main():
    parser = argparse.ArgumentParser(description="完整人形站立 PPO 训练")
    parser.add_argument("--steps", type=int, default=1_500_000)
    parser.add_argument("--save-dir", type=str, default="checkpoints/full_body")
    parser.add_argument("--no-live", action="store_true", help="关闭网页可视化")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default="",
                        help="加载已有全身模型续训(路径不含扩展名，如 checkpoints/full_body/stand)")
    parser.add_argument("--disturb", action="store_true",
                        help="启用抗扰训练(随机外力推 + 更大初始随机化)")
    parser.add_argument("--push-force", type=float, default=120.0,
                        help="推力量级(N)，作用于躯干")
    parser.add_argument("--push-interval", type=int, default=100,
                        help="每隔多少步随机推一次")
    parser.add_argument("--reset-noise", type=float, default=0.05,
                        help="reset 随机化幅度(抗扰建议 0.05)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="学习率(续训抗扰建议调小到 1e-4 保护已学能力)")
    parser.add_argument("--push-curriculum", action="store_true",
                        help="课程式推力：训练中推力从小到大线性递增")
    parser.add_argument("--save-freq", type=int, default=100_000,
                        help="每隔多少步覆盖式保存一次模型(0=关闭，仅训练结束时保存)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    live_server = None
    render_env = None
    if not args.no_live:
        from humanoid_rl.visualization.live_server import LiveVisServer
        live_server = LiveVisServer(port=args.port)
        live_server.start()
        live_server.push_stage_change(0, "抗扰平衡" if args.disturb else "完整人形站立")
        render_env = FullBodyStandEnv(render_mode="rgb_array")
        render_env.reset()
        print(f"实时可视化: http://localhost:{args.port}")

    env_kwargs = dict(
        disturb=args.disturb,
        push_force=args.push_force,
        push_interval=args.push_interval,
        reset_noise=(args.reset_noise if args.disturb else 0.02),
    )

    def make_env():
        return Monitor(FullBodyStandEnv(**env_kwargs))

    vec_env = DummyVecEnv([make_env])

    resume_path = args.resume
    if resume_path:
        # 续训：观测/动作维度不变时可整段加载旧模型继续训练(finetune)。
        # 先恢复 VecNormalize 统计，再加载 PPO 权重接着训。
        vn_path = f"{resume_path}_vecnorm.pkl"
        if os.path.exists(vn_path):
            vec_env = VecNormalize.load(vn_path, vec_env)
            vec_env.training = True
            print(f"[续训] 已加载观测归一化统计: {vn_path}")
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
            print(f"[续训] 未找到 vecnorm，使用新归一化: {vn_path}")
        model = PPO.load(resume_path, env=vec_env, device="cpu")
        model.tensorboard_log = os.path.join(args.save_dir, "tb_logs")
        # 续训用更小学习率保护已学的站立能力(避免灾难性遗忘)
        model.learning_rate = args.lr
        model.lr_schedule = lambda _: args.lr
        print(f"[续训] 已加载模型继续训练: {resume_path} (lr={args.lr})")
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        model = PPO(
            policy="MlpPolicy", env=vec_env,
            learning_rate=args.lr, n_steps=2048, batch_size=256, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
            vf_coef=0.5, max_grad_norm=0.5, verbose=1, seed=args.seed,
            tensorboard_log=os.path.join(args.save_dir, "tb_logs"),
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               log_std_init=-1.0),
        )

    callbacks = []
    if args.save_freq > 0:
        callbacks.append(PeriodicSaveCallback(
            save_freq=args.save_freq, save_dir=args.save_dir, name="stand"))
        print(f"[定期保存] 每 {args.save_freq} 步覆盖保存到 "
              f"{os.path.join(args.save_dir, 'stand')}.zip")
    if live_server:
        callbacks.append(LiveCallback(live_server, render_env))
    if args.disturb and args.push_curriculum:
        # 课程式推力：从 1/4 目标力起步，线性增到 push_force
        callbacks.append(PushCurriculumCallback(
            start_force=max(20.0, args.push_force * 0.25),
            max_force=args.push_force,
            total_steps=args.steps,
        ))
        print(f"[课程] 推力将从 {max(20.0, args.push_force*0.25):.0f}N "
              f"线性增至 {args.push_force:.0f}N")
    callback = callbacks if callbacks else None

    print("=" * 60)
    print(f"完整人形站立训练: {args.steps} 步")
    print("=" * 60)
    model.learn(total_timesteps=args.steps, callback=callback,
                tb_log_name="full_body_stand",
                reset_num_timesteps=not bool(resume_path))

    model_path = os.path.join(args.save_dir, "stand")
    model.save(model_path)
    vec_env.save(f"{model_path}_vecnorm.pkl")
    print(f"模型已保存: {model_path}")

    if render_env is not None:
        render_env.close()
    if live_server:
        live_server.stop()


if __name__ == "__main__":
    main()
