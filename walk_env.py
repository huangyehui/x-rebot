"""完整人形行走环境（方案A：沿用现有矢状面自由度结构）

在已稳定站立的全身模型上训练直线行走。复用 full_body.xml，
奖励改为：前进速度 + 保持直立/头竖直 + 存活，去掉"静止"奖励(走路要动)。
从站立模型热启动续训(观测/动作维度与站立环境一致，可整段加载)。
"""
import os
from collections import deque

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class FullBodyWalkEnv(gym.Env):
    """完整人形行走任务环境"""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, target_speed=0.3, hip_sym_coef=0.0,
                 step_sym_coef=0.0, imitate_delay=False):
        super().__init__()
        self.render_mode = render_mode
        # 髋摆幅对称惩罚系数：0=关闭(与 ft3 奖励完全一致，用于对照实验)
        self.hip_sym_coef = float(hip_sym_coef)
        # 左右步长对称惩罚系数：0=关闭
        self.step_sym_coef = float(step_sym_coef)
        # 镜像模仿目标：False=旧版 r_hip→-l_hip(会把右腿往后拉)；True=半周期延迟的左腿角
        self.imitate_delay = bool(imitate_delay)
        self.target_speed = target_speed   # 目标前进速度 m/s（沿 +x）

        model_path = os.path.join(os.path.dirname(__file__), "full_body.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = 5
        self.dt = self.model.opt.timestep * self.frame_skip
        self.max_episode_steps = 1000
        self.step_count = 0

        self.init_pelvis_height = 0.88
        self.init_head_height = 1.33

        bid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
        self._pelvis_id = bid("pelvis")
        self._torso_id = bid("torso")
        self._head_id = bid("head")
        self._lfoot_id = bid("left_foot")
        self._rfoot_id = bid("right_foot")

        # 膝关节 qpos 地址(用于步态姿态奖励：支撑腿应伸直，膝角接近0)
        jadr = lambda n: self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        self._rknee_adr = jadr("right_knee")
        self._lknee_adr = jadr("left_knee")
        # 左右腿完整关节地址(用于右腿追踪左腿镜像奖励)
        self._rhip_adr    = jadr("right_hip_y")
        self._lhip_adr    = jadr("left_hip_y")
        self._rankle_adr  = jadr("right_ankle_y")
        self._lankle_adr  = jadr("left_ankle_y")

        self._nq_joint = self.model.nq - 7
        self._nv_joint = self.model.nv - 6

        self.n_actuators = self.model.nu
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_actuators,), dtype=np.float32)

        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float64)

        self._renderer = None
        self._prev_action = np.zeros(self.n_actuators)

        # 停滞检测：连续 _stall_limit 步几乎不前进就终止回合(约 2 秒)
        self._stall_steps = 0
        self._stall_limit = 120

        # 单脚抬脚高度窗口(约 1 秒)：每只脚分别记录高度，取窗口内 max-min 作为抬脚幅度
        self._swing_win = int(round(1.0 / self.dt))
        self._lz_hist = deque(maxlen=self._swing_win)
        self._rz_hist = deque(maxlen=self._swing_win)
        # 髋角窗口(同 1 秒)：用于左右髋摆幅对称惩罚
        self._lhip_hist = deque(maxlen=self._swing_win)
        self._rhip_hist = deque(maxlen=self._swing_win)

        # 步长对称：每只脚着地瞬间记录"领先另一只脚的 x 距离"，各保留最近 3 步
        self._ground_z = 0.09            # 脚 body 原点低于此高度视为着地(实测贴地约 0.08)
        self._l_steps = deque(maxlen=3)
        self._r_steps = deque(maxlen=3)
        self._l_on = self._r_on = True
        # 半周期延迟的左腿关节角(步频约 1Hz → 0.5s)，用于修正后的镜像模仿目标
        self._half_period = int(round(0.5 / self.dt))
        self._lleg_hist = deque(maxlen=self._half_period + 1)

    def _upright(self, body_id):
        return float(self.data.xmat[body_id].reshape(3, 3)[2, 2])

    def set_target_speed(self, speed):
        """课程式提速用。"""
        self.target_speed = float(speed)

    def _get_obs(self):
        """观测：与站立环境完全相同的 45 维，以便整段加载站立模型热启动。

        前进速度已包含在 qvel[:6](qvel[0]=沿x线速度)，无需额外维度；
        目标速度在单次训练内固定，不进观测，保证维度与站立模型一致。
        """
        com = self.data.subtree_com[0][:2]
        lf = self.data.xpos[self._lfoot_id][:2]
        rf = self.data.xpos[self._rfoot_id][:2]
        mid = 0.5 * (lf + rf)
        pelvis_h = self.data.xpos[self._pelvis_id][2]
        head_h = self.data.xpos[self._head_id][2]
        return np.concatenate([
            self.data.qpos[7:].copy(),
            self.data.qvel[6:].copy(),
            self.data.qvel[:6].copy(),
            [pelvis_h, head_h,
             self._upright(self._pelvis_id),
             self._upright(self._torso_id),
             self._upright(self._head_id)],
            com - mid,
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = self.init_pelvis_height
        self.data.qpos[7:] += self.np_random.uniform(-0.02, 0.02, self._nq_joint)
        self.data.qvel[:] += self.np_random.uniform(-0.02, 0.02, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self._stall_steps = 0
        self._prev_action = np.zeros(self.n_actuators)
        self._prev_action2 = np.zeros(self.n_actuators)   # 上上步动作(算二阶差分/抖动)
        self._lz_hist.clear()
        self._rz_hist.clear()
        self._lhip_hist.clear()
        self._rhip_hist.clear()
        self._l_steps.clear()
        self._r_steps.clear()
        self._l_on = self._r_on = True
        self._lleg_hist.clear()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action
        x_before = float(self.data.xpos[self._pelvis_id][0])
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        x_after = float(self.data.xpos[self._pelvis_id][0])
        self.step_count += 1

        forward_vel = (x_after - x_before) / self.dt
        pelvis_h = self.data.xpos[self._pelvis_id][2]
        head_h = self.data.xpos[self._head_id][2]
        pelvis_up = self._upright(self._pelvis_id)
        torso_up = self._upright(self._torso_id)
        head_up = self._upright(self._head_id)
        com = self.data.subtree_com[0][:2]
        lf = self.data.xpos[self._lfoot_id][:2]
        rf = self.data.xpos[self._rfoot_id][:2]
        lf_z = float(self.data.xpos[self._lfoot_id][2])
        rf_z = float(self.data.xpos[self._rfoot_id][2])
        rknee = float(self.data.qpos[self._rknee_adr])   # 0=伸直, 正=向后屈膝
        lknee = float(self.data.qpos[self._lknee_adr])
        # 侧向(y)偏移惩罚：鼓励走直线、抑制侧倒
        lateral_dev = abs(com[1] - 0.5 * (lf[1] + rf[1]))

        # 前进奖励：线性正比于速度(占目标的比例)，封顶 1.0。
        # 关键：速度≤0 得 0 分，不给"原地不动"任何安慰分，逼策略真正迈步前进。
        forward_r = float(np.clip(forward_vel / max(self.target_speed, 1e-6), 0.0, 1.0))
        # 超速轻微回拉，避免为刷前进分而冲刺失稳
        if forward_vel > self.target_speed:
            forward_r = 1.0 - 0.5 * np.clip(
                (forward_vel - self.target_speed) / max(self.target_speed, 1e-6), 0.0, 1.0)
        # 原地不动惩罚：速度远低于目标时持续扣分，打破"站着不动"的局部最优
        stall_pen = max(0.0, 0.3 * self.target_speed - max(forward_vel, 0.0))
        # 姿态奖励：保持直立/头竖直/高度(维持站立能力)
        height_r = np.exp(-10.0 * (pelvis_h - self.init_pelvis_height) ** 2)
        torso_up_r = max(0.0, torso_up)
        head_up_r = max(0.0, head_up)

        # === 步态形状约束(治"腿软")===
        # 支撑腿伸直：着地那条腿(脚更低)的膝盖应接近伸直(膝角接近0)
        ground_foot = "r" if rf_z < lf_z else "l"
        support_knee = rknee if ground_foot == "r" else lknee
        support_straight_r = np.exp(-4.0 * support_knee ** 2)   # 支撑膝越直越高
        # 抬脚：两脚分别计分、取较差那只。旧版 |lf_z-rf_z|/0.12 只抬一只脚也能拿满分，
        # 导致"右腿贴地当拐杖、左腿独自迈步"。改为每只脚最近 1 秒内的抬脚幅度
        # (max-min)各自归一化，取 min，逼两条腿都要抬。
        foot_clearance = abs(lf_z - rf_z)
        self._lz_hist.append(lf_z)
        self._rz_hist.append(rf_z)
        l_lift = max(self._lz_hist) - min(self._lz_hist)
        r_lift = max(self._rz_hist) - min(self._rz_hist)
        swing_r = float(np.clip(min(l_lift, r_lift) / 0.06, 0.0, 1.0))
        # 软腿惩罚：两腿同时过度屈膝(半蹲走)时惩罚(屈膝为正角)
        crouch_pen = max(0.0, (rknee + lknee) - 0.3)

        lateral_pen = 1.0 * lateral_dev
        energy = 0.003 * np.sum(action ** 2)
        # 一阶差分(帧间变化)：抑制动作突变，鼓励平滑。保持温和(0.01)，
        # 避免与下面的 jerk 叠加过猛而把策略逼成"少动"(实测会退回拖步并失稳)。
        jitter = 0.01 * np.sum((action - self._prev_action) ** 2)
        # 二阶差分(加加速度/jerk)：专门惩罚"来回反复抖"的高频振荡。
        # a - 2*a_prev + a_prev2 在方向频繁反转时很大、在平滑单向摆动时接近0，
        # 因此能打击腿部哆嗦又不伤正常摆腿。系数取 0.01 轻推，避免破坏抬脚/平衡。
        jerk = 0.01 * np.sum((action - 2 * self._prev_action
                              + self._prev_action2) ** 2)
        self._prev_action2 = self._prev_action.copy()
        self._prev_action = action.copy()

        # === 右腿追踪左腿镜像(治"右腿碎步") ===
        # 左腿走得好(~1Hz 大步)，右腿碎步(~4Hz 小步)。
        # 实时读左腿髋/膝/踝，取反相作为右腿目标，奖励右腿逼近目标。
        # 髋:两腿反相(负号)；膝和踝:两腿应该一样(同号，摆动相同侧弯曲)。
        l_hip   = float(self.data.qpos[self._lhip_adr])
        l_knee  = float(self.data.qpos[self._lknee_adr])
        l_ankle = float(self.data.qpos[self._lankle_adr])
        r_hip   = float(self.data.qpos[self._rhip_adr])
        r_knee  = float(self.data.qpos[self._rknee_adr])
        r_ankle = float(self.data.qpos[self._rankle_adr])
        # === 髋摆幅对称(治"左大步、右小步") ===
        # 最近 1 秒内两侧髋角极差(max-min)之差的绝对值；ft3 实测左 0.97 / 右 0.35。
        # 只比较摆幅，不强制相位，避免与 right_imitate 的反相约束冲突。
        self._lhip_hist.append(l_hip)
        self._rhip_hist.append(r_hip)
        l_hip_rom = max(self._lhip_hist) - min(self._lhip_hist)
        r_hip_rom = max(self._rhip_hist) - min(self._rhip_hist)
        hip_sym_pen = self.hip_sym_coef * abs(l_hip_rom - r_hip_rom)

        # === 左右步长对称(治"右脚迈不到身体前面") ===
        # 每只脚"离地→着地"的瞬间，记录它领先另一只脚的 x 距离(步长)。
        # ft5 实测：左步长约 +0.59m，右步长约 −0.09m(右脚落地仍在左脚后面)。
        # 惩罚 = coef × |左最近 3 步均值 − 右最近 3 步均值|，每步都扣，两侧都有数据才生效。
        lf_x = float(self.data.xpos[self._lfoot_id][0])
        rf_x = float(self.data.xpos[self._rfoot_id][0])
        l_on, r_on = lf_z < self._ground_z, rf_z < self._ground_z
        if l_on and not self._l_on:
            self._l_steps.append(lf_x - rf_x)
        if r_on and not self._r_on:
            self._r_steps.append(rf_x - lf_x)
        self._l_on, self._r_on = l_on, r_on
        if self._l_steps and self._r_steps:
            l_step = float(np.mean(self._l_steps))
            r_step = float(np.mean(self._r_steps))
            step_sym_pen = self.step_sym_coef * abs(l_step - r_step)
        else:
            l_step = r_step = 0.0
            step_sym_pen = 0.0

        # 镜像模仿目标
        if self.imitate_delay:
            # 修正版：右腿 = 半个步态周期之前的左腿(同号)。正常行走两腿都围绕
            # "略微前屈"的角度摆动，只差半个周期；旧版 -l_hip 会把右髋均值推到
            # 左髋均值的相反数，等于把右腿整体往后拉(ft5 右髋均值 −0.04 vs 左 −0.29)。
            self._lleg_hist.append((l_hip, l_knee, l_ankle))
            r_hip_target, r_knee_target, r_ankle_target = self._lleg_hist[0]
        else:
            r_hip_target   = -l_hip     # 旧版：反相(有偏差，见上)
            r_knee_target  =  l_knee
            r_ankle_target =  l_ankle
        right_imitate_r = np.exp(-5.0 * (
            (r_hip   - r_hip_target)   ** 2
          + (r_knee  - r_knee_target)  ** 2 * 0.5
          + (r_ankle - r_ankle_target) ** 2 * 0.5
        ))

        # 前进奖励权重提到 8.0(原 4.0)，并降低纯姿态项权重，
        # 让"站着不动能拿的分"显著低于"迈步前进能拿的分"。
        # swing_r 权重 1.0→2.0：更强奖励清晰抬脚，摆脱拖步、让步态更利落优雅
        reward = (0.5 + 8.0 * forward_r
                  + 1.0 * height_r + 1.5 * torso_up_r + 1.0 * head_up_r
                  + 1.0 * support_straight_r + 2.0 * swing_r
                  + 3.5 * right_imitate_r   # 1.5→3.5：加强右腿跟随左腿镜像
                  - 3.0 * stall_pen
                  - lateral_pen - 1.0 * crouch_pen - energy - jitter - jerk
                  - hip_sym_pen - step_sym_pen)

        # 累计前进距离与"停滞步数"：长时间不前进就提前终止，断掉原地站死的局部最优
        if forward_vel < 0.1 * self.target_speed:
            self._stall_steps += 1
        else:
            self._stall_steps = 0

        terminated = bool(pelvis_h < 0.55 or pelvis_up < 0.5
                          or torso_up < 0.5 or head_up < 0.5
                          or self._stall_steps >= self._stall_limit)
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "pelvis_height": pelvis_h,
            "head_height": head_h,
            "torso_upright": torso_up,
            "head_upright": head_up,
            "upright": torso_up,
            "forward_vel": forward_vel,
            "lateral_dev": lateral_dev,
            "distance": x_after,
            "support_knee": support_knee,
            "foot_clearance": foot_clearance,
            "l_lift": l_lift,
            "r_lift": r_lift,
            "right_imitate": right_imitate_r,
            "l_hip_rom": l_hip_rom,
            "r_hip_rom": r_hip_rom,
            "l_step": l_step,
            "r_step": r_step,
            "step_sym_pen": step_sym_pen,
            "stall_steps": self._stall_steps,
            "reward": reward,
            "length": self.step_count,
            "torso_height": pelvis_h,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="track")
            return self._renderer.render()
        return None
