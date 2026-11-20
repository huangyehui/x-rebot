#!/usr/bin/env python3
"""PD 位置控制 + 相位时钟 + 参考步态跟踪 的 PPO 训练

环境见 walk_env_pd.py。动作含义与旧的力矩环境不同，只能从头训练，
或从本脚本训练出的 PD 模型续训。

用法:
    export PYTHONPATH=$PWD/.pythonpath
    python -m humanoid_rl.full_body.train_walk_pd --steps 5000000 --n-envs 8
    # 续训
    python -m humanoid_rl.full_body.train_walk_pd --resume checkpoints/full_body_walk_pd/walk_pd
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from humanoid_rl.full_body.train_walk import PeriodicSaveCallback, SpeedCurriculumCallback
from humanoid_rl.full_body.walk_env_pd import FullBodyWalkPDEnv


def main():
    p = argparse.ArgumentParser(description="PD + 相位时钟 行走训练")
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--save-dir", type=str, default="checkpoints/full_body_walk_pd")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-freq", type=int, default=200_000)
    p.add_argument("--target-speed", type=float, default=0.5)
    p.add_argument("--speed-curriculum", action="store_true",
                   help="目标速度从 0.1 线性增到 target-speed")
    p.add_argument("--gait-freq", type=float, default=1.0, help="步态周期频率 Hz")
    p.add_argument("--hip-amp", type=float, default=0.3)
    p.add_argument("--knee-amp", type=float, default=0.6)
    p.add_argument("--ankle-amp", type=float, default=0.15)
    p.add_argument("--arm-amp", type=float, default=0.25)
    p.add_argument("--no-residual", action="store_true",
                   help="动作为相对站立姿态的偏移(默认: 相对参考步态的修正)")
    p.add_argument("--action-filter", type=float, default=0.6, help="低通系数 β，1=不滤波")
    args = p.parse_args()

    env_kwargs = dict(target_speed=args.target_speed, gait_freq=args.gait_freq,
                      hip_amp=args.hip_amp, knee_amp=args.knee_amp,
                      ankle_amp=args.ankle_amp, arm_amp=args.arm_amp,
                      residual=not args.no_residual, action_filter=args.action_filter)
    make_env = lambda: Monitor(FullBodyWalkPDEnv(**env_kwargs))
    vec_env = (SubprocVecEnv([make_env] * args.n_envs) if args.n_envs > 1
               else DummyVecEnv([make_env]))
    os.makedirs(args.save_dir, exist_ok=True)
    name = "walk_pd"

    if args.resume:
        vec_env = VecNormalize.load(f"{args.resume}_vecnorm.pkl", vec_env)
        vec_env.training = True
        model = PPO.load(args.resume, env=vec_env, device="cpu",
                         custom_objects={"n_steps": args.n_steps, "batch_size": args.batch_size,
                                         "learning_rate": args.lr})
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        model = PPO(
            policy="MlpPolicy", env=vec_env, learning_rate=args.lr,
            n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
            vf_coef=0.5, max_grad_norm=0.5, verbose=1, seed=args.seed, device="cpu",
            tensorboard_log=os.path.join(args.save_dir, "tb_logs"),
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               log_std_init=-1.0),
        )

    callbacks = []
    if args.save_freq > 0:
        callbacks.append(PeriodicSaveCallback(args.save_freq, args.save_dir, name=name))
    if args.speed_curriculum:
        callbacks.append(SpeedCurriculumCallback(0.1, args.target_speed, args.steps))

    print(f"[PD 行走] {args.steps} 步, {args.n_envs} 环境, 目标速度 {args.target_speed} m/s, "
          f"步频 {args.gait_freq} Hz, residual={not args.no_residual}")
    model.learn(total_timesteps=args.steps, callback=callbacks or None,
                tb_log_name="walk_pd", reset_num_timesteps=not bool(args.resume))
    path = os.path.join(args.save_dir, name)
    model.save(path)
    vec_env.save(f"{path}_vecnorm.pkl")
    print(f"模型已保存: {path}")


if __name__ == "__main__":
    main()
