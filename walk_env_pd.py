"""PD 位置控制 + 相位时钟 + 参考步态跟踪的行走环境

与 walk_env.FullBodyWalkEnv 的区别:
1. 动作含义: 策略输出的是"相对参考步态的关节角修正"(residual=True，默认)
   或"相对站立姿态的关节角偏移"(residual=False)，再由 PD 控制器换算成力矩。
   PD 在每个物理子步都重新计算一次。
2. 相位时钟: phase 按 gait_freq 匀速推进，sin/cos(phase) 进入观测。
   右腿用 phase，左腿用 phase+π；sin(p)>0 为摆动期，sin(p)<0 为支撑期
   (与 play_ref_gait.py 的相位约定一致)。
3. 奖励: 以参考关节角跟踪、按相位的接触时序、抬脚高度、速度跟踪为主体，
   去掉了原环境里"右腿模仿左腿 / 髋摆幅对称 / 步长对称"等补丁项。
   - 接触时序、抬脚高度对两只脚分别打分后取较差的一只，防止只用一条腿踏步；
   - 跟踪/接触/抬脚三项乘以速度门控 gate = 0.2 + 0.8·(1 秒平均速度/目标速度)，
     原地踏步最多只能拿到这几项的 20%；
   - 停滞终止看过去 2 秒的骨盆总位移(< 0.15 m 即终止)，不看瞬时速度。

关节方向(已在仿真中实测, 2026-11):
  hip_y      +  = 腿向后(伸髋)，  −  = 腿向前(屈髋)
  knee       +  = 屈膝
  ankle_y    +  = 脚尖向下(跖屈)
  shoulder_y +  = 手臂向后
观测 45 → 65 维: 原 45 维 + sin/cos(phase) 2 维 + 双脚接触 2 维 + 上一步动作 16 维。
动作维度不变(16)，但含义已变，不能从旧的力矩策略热启动。
"""
import os
from collections import deque

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

# 执行器顺序(见 full_body.xml)
R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE = 0, 1, 2, 3, 4, 5
WAIST_Y, WAIST_X = 6, 7
R_SH_Y, R_SH_X, R_ELBOW, L_SH_Y, L_SH_X, L_ELBOW = 8, 9, 10, 11, 12, 13
NECK_Y, NECK_X = 14, 15

# 每个执行器的 PD 增益与动作幅度(按执行器顺序)。XML 已给每个关节 damping=5，Kd 只是额外阻尼。
#               r_hip r_knee r_ank l_hip l_knee l_ank waist_y waist_x r_sh_y r_sh_x r_elb l_sh_y l_sh_x l_elb neck_y neck_x
KP = np.array([100., 100., 40., 100., 100., 40., 60., 60., 20., 20., 15., 20., 20., 15., 8., 8.])
KD = np.array([2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2])
ACT_SCALE = np.array([0.5, 0.5, 0.4, 0.5, 0.5, 0.4, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3])
# 跟踪奖励中各关节的权重: 腿为主，手臂次之，躯干/颈只做弱约束
TRACK_W = np.array([1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 0.3, 0.3, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1])


class FullBodyWalkPDEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, target_speed=0.5, gait_freq=1.0,
                 hip_amp=0.3, knee_amp=0.6, ankle_amp=0.15, arm_amp=0.25,
                 swing_height=0.08, residual=True, action_filter=0.6):
        super().__init__()
        self.render_mode = render_mode
        self.target_speed = float(target_speed)
        self.gait_freq = float(gait_freq)
        self.hip_amp, self.knee_amp = float(hip_amp), float(knee_amp)
        self.ankle_amp, self.arm_amp = float(ankle_amp), float(arm_amp)
        self.swing_height = float(swing_height)
        self.residual = bool(residual)
        # 一阶低通: a_used = β·a + (1-β)·a_prev，β=1 表示不滤波
        self.action_filter = float(action_filter)

        model_path = os.path.join(os.path.dirname(__file__), "full_body.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 5
        self.dt = self.model.opt.timestep * self.frame_skip
        self.max_episode_steps = 1000
        self.init_pelvis_height = 0.88
        # 过去 2 秒骨盆前进不足 stall_min_disp 米就终止回合
        self.stall_min_disp = 0.15
        self._stall_hist = deque(maxlen=int(round(2.0 / self.dt)) + 1)
        # 最近 1 秒骨盆 x，用于计算平均速度(步态项打折)
        self._x_hist = deque(maxlen=int(round(1.0 / self.dt)) + 1)

        bid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
        gid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
        self._pelvis_id, self._torso_id, self._head_id = bid("pelvis"), bid("torso"), bid("head")
        self._lfoot_id, self._rfoot_id = bid("left_foot"), bid("right_foot")
        self._floor_gid = gid("floor")
        self._foot_gid = {"r": gid("right_foot_geom"), "l": gid("left_foot_geom")}

        self.n_actuators = self.model.nu
        # 执行器 → 关节的 qpos / qvel 地址(观测里关节顺序与执行器顺序不同，必须显式映射)
        jids = self.model.actuator_trnid[:, 0]
        self._act_qadr = self.model.jnt_qposadr[jids].copy()
        self._act_vadr = self.model.jnt_dofadr[jids].copy()
        self._gear = self.model.actuator_gear[:, 0].copy()
        self._jrange = self.model.jnt_range[jids].copy()

        # 站立姿态(全 0 即直立)，肘关节微屈
        self._q0 = np.zeros(self.n_actuators)
        self._q0[R_ELBOW] = self._q0[L_ELBOW] = 0.3

        self.phase = 0.0
        self._prev_action = np.zeros(self.n_actuators)
        self._a_used = np.zeros(self.n_actuators)
        self._renderer = None
        self.step_count = 0

        self.action_space = spaces.Box(-1.0, 1.0, (self.n_actuators,), dtype=np.float32)
        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float64)

    # ---------- 工具 ----------
    def set_target_speed(self, speed):
        self.target_speed = float(speed)

    def _upright(self, body_id):
        return float(self.data.xmat[body_id].reshape(3, 3)[2, 2])

    def _foot_contacts(self):
        on = {"r": 0.0, "l": 0.0}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (c.geom1, c.geom2)
            if self._floor_gid not in pair:
                continue
            for side, g in self._foot_gid.items():
                if g in pair:
                    on[side] = 1.0
        return on

    def leg_phases(self, phase=None):
        p = self.phase if phase is None else phase
        return {"r": p, "l": p + np.pi}

    def q_ref(self, phase=None):
        """参考关节角(执行器顺序)。摆动期 sin(p)>0: 腿由后向前摆、屈膝、脚尖略抬。"""
        ref = self._q0.copy()
        legs = {"r": (R_HIP, R_KNEE, R_ANKLE, R_SH_Y), "l": (L_HIP, L_KNEE, L_ANKLE, L_SH_Y)}
        for side, p in self.leg_phases(phase).items():
            hip, knee, ankle, sh = legs[side]
            swing = max(0.0, np.sin(p))
            # 髋: p=0 在最后方(+A)，p=π 摆到最前方(-A)，支撑期再回到后方；略带前屈偏置
            ref[hip] = self.hip_amp * np.cos(p) - 0.05
            ref[knee] = self.knee_amp * swing + 0.05
            # 踝: 摆动期脚尖上抬(负)，支撑末期蹬地(正)
            ref[ankle] = -0.5 * self.ankle_amp * swing + self.ankle_amp * max(0.0, -np.sin(p - 0.6))
            # 肩: 与同侧腿反向摆(同侧腿向前时手臂向后)
            ref[sh] = -self.arm_amp * np.cos(p)
        return np.clip(ref, self._jrange[:, 0], self._jrange[:, 1])

    def foot_height_ref(self, side, phase=None):
        p = self.leg_phases(phase)[side]
        return self._foot_z0 + self.swing_height * max(0.0, np.sin(p))

    def _joint_q(self):
        return self.data.qpos[self._act_qadr].copy()

    def _joint_qd(self):
        return self.data.qvel[self._act_vadr].copy()

    def _get_obs(self):
        com = self.data.subtree_com[0][:2]
        lf = self.data.xpos[self._lfoot_id][:2]
        rf = self.data.xpos[self._rfoot_id][:2]
        on = self._foot_contacts()
        return np.concatenate([
            self.data.qpos[7:].copy(),
            self.data.qvel[6:].copy(),
            self.data.qvel[:6].copy(),
            [self.data.xpos[self._pelvis_id][2], self.data.xpos[self._head_id][2],
             self._upright(self._pelvis_id), self._upright(self._torso_id),
             self._upright(self._head_id)],
            com - 0.5 * (lf + rf),
            [np.sin(self.phase), np.cos(self.phase)],
            [on["r"], on["l"]],
            self._prev_action,
        ])

    # ---------- gym 接口 ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = self.init_pelvis_height
        self.phase = float(self.np_random.uniform(0.0, 2 * np.pi))
        # 从参考步态的当前相位附近开始，避免一开始就与参考相差很大
        q = self.q_ref() + self.np_random.uniform(-0.02, 0.02, self.n_actuators)
        self.data.qpos[self._act_qadr] = q
        self.data.qvel[:] = self.np_random.uniform(-0.02, 0.02, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        # 让最低的脚刚好着地
        lowest = min(self.data.xpos[self._lfoot_id][2], self.data.xpos[self._rfoot_id][2])
        self.data.qpos[2] += 0.05 - lowest
        mujoco.mj_forward(self.model, self.data)
        self._foot_z0 = 0.05
        x0 = float(self.data.xpos[self._pelvis_id][0])
        self._stall_hist.clear(); self._stall_hist.append(x0)
        self._x_hist.clear(); self._x_hist.append(x0)
        self.step_count = 0
        self._prev_action = np.zeros(self.n_actuators)
        self._a_used = np.zeros(self.n_actuators)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        b = self.action_filter
        self._a_used = b * action + (1.0 - b) * self._a_used
        base = self.q_ref() if self.residual else self._q0
        q_target = np.clip(base + ACT_SCALE * self._a_used,
                           self._jrange[:, 0], self._jrange[:, 1])

        x_before = float(self.data.xpos[self._pelvis_id][0])
        torque_sq = 0.0
        for _ in range(self.frame_skip):
            tau = KP * (q_target - self._joint_q()) - KD * self._joint_qd()
            ctrl = np.clip(tau / self._gear, -1.0, 1.0)
            self.data.ctrl[:] = ctrl
            torque_sq += float(np.sum(ctrl ** 2))
            mujoco.mj_step(self.model, self.data)
        x_after = float(self.data.xpos[self._pelvis_id][0])
        self.step_count += 1
        self.phase = (self.phase + 2 * np.pi * self.gait_freq * self.dt) % (2 * np.pi)

        # ---- 状态量 ----
        forward_vel = (x_after - x_before) / self.dt
        pelvis_h = float(self.data.xpos[self._pelvis_id][2])
        head_h = float(self.data.xpos[self._head_id][2])
        pelvis_up = self._upright(self._pelvis_id)
        torso_up = self._upright(self._torso_id)
        head_up = self._upright(self._head_id)
        com = self.data.subtree_com[0][:2]
        lf, rf = self.data.xpos[self._lfoot_id], self.data.xpos[self._rfoot_id]
        lateral_dev = abs(com[1] - 0.5 * (lf[1] + rf[1]))
        on = self._foot_contacts()
        foot_z = {"r": float(rf[2]), "l": float(lf[2])}

        # ---- 奖励 ----
        q = self._joint_q()
        track_err = float(np.sum(TRACK_W * (q - self.q_ref()) ** 2))
        r_track = np.exp(-2.0 * track_err)

        # 接触时序: 与相位一致 +1，不一致 -1，交界处不计分。
        # (试训发现若"不一致"只给 0 分，双脚一直着地也能拿一半分，策略会停在"站着不动")
        # 两只脚分别打分，取较差的那只(第二轮试训: 取平均时策略只用右脚踏步，左脚不动)
        contact_score, swing_score = {}, {}
        for side, p in self.leg_phases().items():
            s = np.sin(p)
            if s < -0.1:                       # 应支撑
                contact_score[side] = 1.0 if on[side] else -1.0
            elif s > 0.1:                      # 应摆动
                contact_score[side] = -1.0 if on[side] else 1.0
            else:
                contact_score[side] = 0.0
            swing_score[side] = np.exp(-200.0 * (foot_z[side] - self.foot_height_ref(side)) ** 2)
        r_contact = min(contact_score.values())
        r_swing = min(swing_score.values())

        # 速度: 按目标速度的比例线性给分，站着不动得 0 分；超速轻微回拉
        r_vel = float(np.clip(forward_vel / max(self.target_speed, 1e-6), 0.0, 1.0))
        if forward_vel > self.target_speed:
            r_vel = 1.0 - 0.5 * float(np.clip(
                (forward_vel - self.target_speed) / max(self.target_speed, 1e-6), 0.0, 1.0))
        # 步态项按"最近 1 秒平均速度"打折: 原地踏步最多只拿 20%。
        # 用 1 秒平均而不是瞬时速度，避免一步之内骨盆前后晃动带来的噪声。
        self._x_hist.append(x_after)
        avg_vel = (self._x_hist[-1] - self._x_hist[0]) / max((len(self._x_hist) - 1) * self.dt, self.dt)
        gate = 0.2 + 0.8 * float(np.clip(avg_vel / max(self.target_speed, 1e-6), 0.0, 1.0))
        r_height = np.exp(-10.0 * (pelvis_h - self.init_pelvis_height) ** 2)
        action_rate = float(np.sum((action - self._prev_action) ** 2))

        reward = (0.3
                  + gate * (3.0 * r_track + 1.0 * r_contact + 1.0 * r_swing)
                  + 2.0 * r_vel
                  + 0.5 * max(0.0, torso_up) + 0.5 * max(0.0, head_up) + 0.5 * r_height
                  - 0.02 * action_rate
                  - 0.001 * torque_sq / self.frame_skip
                  - 1.0 * lateral_dev)
        self._prev_action = action.copy()

        # 停滞终止: 看过去 2 秒骨盆的总位移，而不是瞬时速度。
        # (第二轮试训: 原地踏步时骨盆前后晃动，瞬时速度会反复越过阈值，计数器一直被清零)
        self._stall_hist.append(x_after)
        stalled = (len(self._stall_hist) == self._stall_hist.maxlen
                   and self._stall_hist[-1] - self._stall_hist[0] < self.stall_min_disp)
        terminated = bool(pelvis_h < 0.55 or pelvis_up < 0.5 or torso_up < 0.5 or head_up < 0.5
                          or stalled)
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "pelvis_height": pelvis_h, "head_height": head_h, "torso_height": pelvis_h,
            "torso_upright": torso_up, "head_upright": head_up, "upright": torso_up,
            "forward_vel": forward_vel, "lateral_dev": lateral_dev, "distance": x_after,
            "phase": self.phase, "r_track": r_track, "track_err": track_err,
            "r_contact": r_contact, "r_swing": r_swing, "r_vel": r_vel,
            "avg_vel": avg_vel, "gate": gate, "stalled": stalled,
            "contact_r": on["r"], "contact_l": on["l"],
            "foot_z_r": foot_z["r"], "foot_z_l": foot_z["l"],
            "foot_x_r": float(rf[0]), "foot_x_l": float(lf[0]),
            "action_rate": action_rate, "reward": reward, "length": self.step_count,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="track")
            return self._renderer.render()
        return None
