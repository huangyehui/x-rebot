#!/usr/bin/env python3
"""用固定正弦动作序列驱动 full_body 模型，验证参考步态是否物理可行。

不使用任何 RL 策略，纯粹用手工设计的 ctrl 信号驱动 MuJoCo 仿真，
直接观察机器人能否走起来、步态是否自然对称。

用法(需要系统有显示器或 X11):
    python -m humanoid_rl.full_body.play_ref_gait
    python -m humanoid_rl.full_body.play_ref_gait --freq 1.0 --hip-amp 0.4
    python -m humanoid_rl.full_body.play_ref_gait --save-gif ref_gait.gif

调参说明:
    --freq      步频 Hz (默认 1.2, 正常人走路约 1.0-1.5 Hz)
    --hip-amp   髋关节摆动幅度 rad (默认 0.35, 约 20 度)
    --knee-amp  膝关节弯曲幅度 rad (默认 0.5, 约 28 度)
    --ankle-amp 踝关节踩地幅度 rad (默认 0.2)
    --arm-amp   肩关节摆臂幅度 rad (默认 0.25)
    --steps     仿真步数 (默认 3000)
"""
import argparse
import os
import numpy as np
import mujoco
import mujoco.viewer


# 执行器顺序(见 full_body.xml actuator 定义):
# 0  right_hip_y,  1  right_knee,   2  right_ankle_y
# 3  left_hip_y,   4  left_knee,    5  left_ankle_y
# 6  waist_y,      7  waist_x
# 8  right_shoulder_y, 9  right_shoulder_x, 10 right_elbow
# 11 left_shoulder_y,  12 left_shoulder_x,  13 left_elbow
# 14 neck_y,       15 neck_x
R_HIP, R_KNEE, R_ANKLE = 0, 1, 2
L_HIP, L_KNEE, L_ANKLE = 3, 4, 5
WAIST_Y, WAIST_X = 6, 7
R_SH_Y, R_SH_X, R_ELBOW = 8, 9, 10
L_SH_Y, L_SH_X, L_ELBOW = 11, 12, 13
NECK_Y, NECK_X = 14, 15


def compute_ctrl(t, freq, hip_amp, knee_amp, ankle_amp, arm_amp):
    """在时刻 t 计算各关节的控制信号([-1, 1] 范围，对应 ctrl)。

    步态相位:
      左腿摆动相 = sin(phase) > 0  (phase = 2π·freq·t)
      右腿摆动相 = sin(phase+π) > 0 (反相)

    髋关节:正弦摆腿(核心)，左右反相
    膝关节:只在摆动相弯曲(半波整流)，支撑相伸直
    踝关节:支撑相末期向下踩蹬地
    肩关节:与同侧髋反相(对角摆臂)
    其余关节:保持 0(自然位置)
    """
    phase = 2 * np.pi * freq * t

    # 左腿摆动包络(摆动相为正)
    l_swing = np.sin(phase)
    r_swing = np.sin(phase + np.pi)   # 右腿反相

    ctrl = np.zeros(16)

    # 髋关节：正弦摆腿
    ctrl[L_HIP] = hip_amp * l_swing
    ctrl[R_HIP] = hip_amp * r_swing

    # 膝关节：摆动相弯曲(半波整流)，支撑相伸直
    # 膝关节正方向=屈膝，摆动相弯曲助抬脚
    ctrl[L_KNEE] = knee_amp * max(0.0, l_swing)
    ctrl[R_KNEE] = knee_amp * max(0.0, r_swing)

    # 踝关节：支撑相末期向下蹬(与摆动相反的相位)
    # 支撑脚蹬地 = 摆动包络为负时踝向下
    ctrl[L_ANKLE] = ankle_amp * max(0.0, -l_swing)
    ctrl[R_ANKLE] = ankle_amp * max(0.0, -r_swing)

    # 肩关节：与同侧髋反相(对角摆臂：右臂配左腿)
    ctrl[L_SH_Y] = arm_amp * (-l_swing)   # 左臂与左腿反相
    ctrl[R_SH_Y] = arm_amp * (-r_swing)

    # 肘关节：保持自然微屈(约 0.3 rad，对应 ctrl 约 0.2)
    ctrl[R_ELBOW] = 0.2
    ctrl[L_ELBOW] = 0.2

    # 腰关节：小幅跟随髋摆动，增加自然感
    ctrl[WAIST_Y] = 0.05 * l_swing

    return np.clip(ctrl, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="固定正弦步态仿真验证")
    parser.add_argument("--freq", type=float, default=1.2, help="步频 Hz")
    parser.add_argument("--hip-amp", type=float, default=0.35, help="髋摆幅 rad")
    parser.add_argument("--knee-amp", type=float, default=0.5, help="膝弯幅 rad")
    parser.add_argument("--ankle-amp", type=float, default=0.2, help="踝踩幅 rad")
    parser.add_argument("--arm-amp", type=float, default=0.25, help="肩摆幅 rad")
    parser.add_argument("--steps", type=int, default=3000, help="仿真步数")
    parser.add_argument("--save-gif", type=str, default="",
                        help="保存 gif 路径(空=不保存，直接弹窗显示)")
    parser.add_argument("--no-viewer", action="store_true",
                        help="不弹窗，只保存 gif(需要 --save-gif)")
    args = parser.parse_args()

    xml_path = os.path.join(os.path.dirname(__file__), "full_body.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    frame_skip = 5
    dt = model.opt.timestep * frame_skip
    print(f"模型: nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"步频 {args.freq}Hz  髋幅 {args.hip_amp}rad  膝幅 {args.knee_amp}rad")
    print(f"仿真 {args.steps} 步 = {args.steps * dt:.1f}s")

    # 初始化：站立姿态
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.88   # 骨盆高度

    frames = []
    renderer = None
    if args.save_gif or args.no_viewer:
        renderer = mujoco.Renderer(model, height=480, width=640)

    t = 0.0
    x_start = float(data.xpos[1][0])  # 骨盆初始 x

    def step_sim():
        nonlocal t
        ctrl = compute_ctrl(t, args.freq, args.hip_amp,
                            args.knee_amp, args.ankle_amp, args.arm_amp)
        data.ctrl[:] = ctrl
        for _ in range(frame_skip):
            mujoco.mj_step(model, data)
        t += dt

    if args.no_viewer or (args.save_gif and not args.no_viewer is False):
        # 无窗口模式：只渲染到 gif
        if not args.save_gif:
            print("[警告] --no-viewer 需要配合 --save-gif 使用")
            return
        for i in range(args.steps):
            step_sim()
            if i % 3 == 0:   # 每 3 步取一帧，减少 gif 大小
                renderer.update_scene(data, camera="track")
                frames.append(renderer.render().copy())
        x_end = float(data.xpos[1][0])
        pelvis_h = float(data.xpos[1][2])
        print(f"前进 {x_end - x_start:.2f}m  骨盆高度 {pelvis_h:.3f}m")
    else:
        # 交互窗口模式
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0.9]
            viewer.cam.distance = 3.5
            viewer.cam.elevation = -15
            step_count = 0
            while viewer.is_running() and step_count < args.steps:
                step_sim()
                if args.save_gif and step_count % 3 == 0:
                    renderer.update_scene(data, camera="track")
                    frames.append(renderer.render().copy())
                viewer.sync()
                step_count += 1
            x_end = float(data.xpos[1][0])
            pelvis_h = float(data.xpos[1][2])
            print(f"前进 {x_end - x_start:.2f}m  骨盆高度 {pelvis_h:.3f}m")

    if renderer:
        renderer.close()

    if frames and args.save_gif:
        import imageio
        imageio.mimsave(args.save_gif, frames, fps=30)
        print(f"已保存: {args.save_gif} ({len(frames)} 帧)")


if __name__ == "__main__":
    main()
