#!/usr/bin/env python3
"""完整人形行走 PPO 训练（带实时网页可视化）

从站立模型热启动，训练直线行走。支持课程式提速。

用法:
    # 从站立模型热启动，慢速起步
    python -m humanoid_rl.full_body.train_walk --resume checkpoints/full_body/stand \
        --target-speed 0.3 --speed-curriculum --steps 1000000
"""
import argparse
import os

import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from humanoid_rl.full_body.walk_env import FullBodyWalkEnv


class PeriodicSaveCallback(BaseCallback):
    """每隔 save_freq 步覆盖式保存一次模型 + VecNormalize 统计。

    固定文件名(覆盖)而非带步数后缀，避免占用大量磁盘；模型与其观测归一化
    统计配套保存，保证中途停止后也能正确加载续训/评估。
    """

    def __init__(self, save_freq, save_dir, name="walk", verbose=1, keep_snapshots=False):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.name = name
        self.keep_snapshots = keep_snapshots   # True: 额外保存带步数后缀的快照(不覆盖)
        self._last_save = 0

    def _on_step(self):
        if self.num_timesteps - self._last_save >= self.save_freq:
            self._last_save = self.num_timesteps
            path = os.path.join(self.save_dir, self.name)
            self.model.save(path)
            vecnorm = self.model.get_vec_normalize_env()
            if vecnorm is not None:
                vecnorm.save(f"{path}_vecnorm.pkl")
            if self.keep_snapshots:
                snap = os.path.join(self.save_dir, "snapshots", f"{self.name}_{self.num_timesteps}")
                os.makedirs(os.path.dirname(snap), exist_ok=True)
                self.model.save(snap)
                if vecnorm is not None:
                    vecnorm.save(f"{snap}_vecnorm.pkl")
            if self.verbose:
                print(f"[定期保存] {self.num_timesteps} 步 -> {path}.zip (覆盖)")
        return True


class SpeedCurriculumCallback(BaseCallback):
    """课程式提速：目标速度从 start→max 线性增长。"""

    def __init__(self, start_speed, max_speed, total_steps, verbose=0):
        super().__init__(verbose)
        self.start_speed = start_speed
        self.max_speed = max_speed
        self.total_steps = max(total_steps, 1)

    def _set(self, spd):
        self.training_env.env_method("set_target_speed", spd)

    def _on_training_start(self):
        self._set(self.start_speed)

    def _on_step(self):
        if self.num_timesteps % 2048 == 0:
            p = min(1.0, self.num_timesteps / self.total_steps)
            self._set(self.start_speed + p * (self.max_speed - self.start_speed))
        return True


class LiveCallback(BaseCallback):
    """推送训练指标和渲染帧到网页。"""

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
                        "com_err": float(info.get("forward_vel", 0.0)),
                    })
                self._ep_reward = 0.0
                self._ep_len = 0
                self._ep_count += 1
        return True


def main():
    parser = argparse.ArgumentParser(description="完整人形行走 PPO 训练")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--save-dir", type=str, default="checkpoints/full_body_walk")
    parser.add_argument("--resume", type=str, default="",
                        help="站立模型路径(整段加载热启动)")
    parser.add_argument("--target-speed", type=float, default=0.5,
                        help="目标前进速度 m/s(课程式时为上限)")
    parser.add_argument("--speed-curriculum", action="store_true",
                        help="课程式提速：目标速度从 0.1 线性增到 target-speed")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-freq", type=int, default=100_000,
                        help="每隔多少步覆盖式保存一次模型(0=关闭，仅训练结束时保存)")
    parser.add_argument("--task", choices=["walk", "march"], default="walk",
                        help="walk=直线行走, march=原地踏步(左右交替抬脚)")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="并行环境数(>1 用 SubprocVecEnv 多进程)。每轮样本 = n_envs*n_steps")
    parser.add_argument("--n-steps", type=int, default=2048, help="每个环境每轮采样步数")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO minibatch 大小")
    parser.add_argument("--hip-sym-coef", type=float, default=0.0,
                        help="左右髋摆幅对称惩罚系数(仅 walk 任务，0=关闭)")
    parser.add_argument("--step-sym-coef", type=float, default=0.0,
                        help="左右步长对称惩罚系数(仅 walk 任务，0=关闭)")
    parser.add_argument("--imitate-delay", action="store_true",
                        help="镜像模仿改用半周期延迟的左腿角(修正旧版把右腿往后拉的偏差)")
    parser.add_argument("--keep-snapshots", action="store_true",
                        help="定期保存时额外留下带步数的快照(save-dir/snapshots/)，用于分析修复所需步数")
    args = parser.parse_args()

    if args.task == "march":
        from humanoid_rl.full_body.march_env import FullBodyMarchEnv
        EnvCls, name = FullBodyMarchEnv, "march"
    else:
        EnvCls, name = FullBodyWalkEnv, "walk"

    os.makedirs(args.save_dir, exist_ok=True)

    live_server = None
    render_env = None
    if not args.no_live:
        from humanoid_rl.visualization.live_server import LiveVisServer
        live_server = LiveVisServer(port=args.port)
        live_server.start()
        live_server.push_stage_change(0, "原地踏步" if name == "march" else "行走")
        render_env = EnvCls(render_mode="rgb_array",
                            target_speed=args.target_speed)
        render_env.reset()
        print(f"实时可视化: http://localhost:{args.port}")

    env_kwargs = {"target_speed": args.target_speed}
    if name == "walk":
        env_kwargs["hip_sym_coef"] = args.hip_sym_coef
        env_kwargs["step_sym_coef"] = args.step_sym_coef
        env_kwargs["imitate_delay"] = args.imitate_delay

    def make_env():
        return Monitor(EnvCls(**env_kwargs))

    if args.n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        vec_env = SubprocVecEnv([make_env for _ in range(args.n_envs)])
        if live_server:
            print("[提示] n-envs>1 时实时画面无法同步训练环境，只推送指标曲线")
    else:
        vec_env = DummyVecEnv([make_env])
    print(f"[并行] {args.n_envs} 环境 × n_steps {args.n_steps} = "
          f"每轮 {args.n_envs * args.n_steps} 样本, minibatch {args.batch_size}")

    if args.resume:
        vn_path = f"{args.resume}_vecnorm.pkl"
        if os.path.exists(vn_path):
            vec_env = VecNormalize.load(vn_path, vec_env)
            vec_env.training = True
            print(f"[热启动] 已加载观测归一化统计: {vn_path}")
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        # 显式覆盖采样/batch 参数：环境数变了，n_steps/batch_size 也要随之生效，
        # PPO.load 会据此重建 rollout buffer
        model = PPO.load(args.resume, env=vec_env, device="cpu",
                         custom_objects={"n_steps": args.n_steps,
                                         "batch_size": args.batch_size})
        model.tensorboard_log = os.path.join(args.save_dir, "tb_logs")
        model.learning_rate = args.lr
        model.lr_schedule = lambda _: args.lr
        print(f"[热启动] 已加载站立模型: {args.resume} (lr={args.lr})")
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        model = PPO(
            policy="MlpPolicy", env=vec_env,
            learning_rate=args.lr, n_steps=args.n_steps,
            batch_size=args.batch_size, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
            vf_coef=0.5, max_grad_norm=0.5, verbose=1, seed=args.seed,
            tensorboard_log=os.path.join(args.save_dir, "tb_logs"),
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               log_std_init=-1.0),
        )

    callbacks = []
    if args.save_freq > 0:
        callbacks.append(PeriodicSaveCallback(
            save_freq=args.save_freq, save_dir=args.save_dir, name=name,
            keep_snapshots=args.keep_snapshots))
        print(f"[定期保存] 每 {args.save_freq} 步覆盖保存到 "
              f"{os.path.join(args.save_dir, name)}.zip")
    if live_server:
        callbacks.append(LiveCallback(live_server, render_env))
    if args.speed_curriculum:
        callbacks.append(SpeedCurriculumCallback(
            start_speed=0.1, max_speed=args.target_speed, total_steps=args.steps))
        print(f"[课程] 目标速度将从 0.1 线性增至 {args.target_speed} m/s")
    callback = callbacks if callbacks else None

    print("=" * 60)
    print(f"完整人形训练[{name}]: {args.steps} 步, 目标速度 {args.target_speed} m/s")
    print("=" * 60)
    model.learn(total_timesteps=args.steps, callback=callback,
                tb_log_name=f"full_body_{name}",
                # 换任务(march)时从 0 计步：PPO.load 会带入旧任务的 ep_info_buffer，
                # 不重置的话前 100 回合的 ep_rew_mean 是旧任务数据，会误导判断
                reset_num_timesteps=(not bool(args.resume))
                or (name == "march" and "march" not in os.path.basename(args.resume)))

    model_path = os.path.join(args.save_dir, name)
    model.save(model_path)
    vec_env.save(f"{model_path}_vecnorm.pkl")
    print(f"模型已保存: {model_path}")

    if render_env is not None:
        render_env.close()
    if live_server:
        live_server.stop()


if __name__ == "__main__":
    main()
