"""原地踏步环境：先单独练会"左右交替、等高、等节奏"的抬脚，再去学前进。

观测与 FullBodyWalkEnv 完全相同(45 维)，可整段加载站立模型热启动，
练完后也能直接接回行走训练。

奖励以"落脚事件"计分，不靠连续惩罚(前几轮连续惩罚总被策略钻空子)：
- 一只脚抬起离地 >= lift_clear 后落下，且与上一步不是同一只脚 → 有效步 +step_bonus
- 同一只脚连续踩两次 → 扣分(单腿跺脚拿不到分)
- 两次有效步间隔 < min_step_interval → 不计分(高频碎步无效)
- 连续 stall_limit 步没有有效步 → 终止回合(堵死"站着不动")
- 两脚同时离地扣分(防蹦跳)；骨盆离开起点扣分(必须原地)
"""
import numpy as np
import mujoco

from humanoid_rl.full_body.walk_env import FullBodyWalkEnv


class FullBodyMarchEnv(FullBodyWalkEnv):
    """原地踏步任务环境"""

    def __init__(self, render_mode=None, target_speed=0.0):
        super().__init__(render_mode=render_mode, target_speed=0.0)
        self.air_margin = 0.01        # 脚底离地多少且无接触才算"离地"(m)
        # 有效步门槛 0.07→0.035：实测从站立出发只能抬到 4~5cm，硬门槛 7cm
        # 一次都触发不了，踏步奖励从未出现；改为低门槛 + 按高度分级给分
        self.lift_clear = 0.035       # 有效步要求的最低抬脚高度(m)
        self.lift_target = 0.08       # 抬到此高度拿满分(m)
        self.min_step_interval = 20   # 两次有效步最小间隔(步)，20*0.015=0.3s
        self.max_swing_steps = 40     # 抬脚引导只在摆动前 0.6s 内给，防单腿站着不放
        # 踏步奖励 10→30：原来站姿每步 ~5 分，踏步摊到每步仅 ~0.3 分，
        # 抬脚有摔倒风险收益却小，策略宁可站着等终止
        self.step_bonus = 30.0
        self.wrong_foot_pen = 5.0
        self._stall_limit = 200       # 3s 无有效步则终止(原 2s 太紧)

    def set_target_speed(self, speed):
        """踏步任务不前进，忽略速度课程。"""
        self.target_speed = 0.0

    def _sole_clear(self, geom_id):
        """脚底离地高度：box 中心 z - 半高 - 站稳时的静置偏移。站稳时 ≈0。

        静置偏移 ≈0.02m 来自接触 margin(0.01) 与软接触穿透，是站稳后
        实测的常数(_rest_offset)，而不是 reset 瞬间的脚高。
        """
        return (float(self.data.geom_xpos[geom_id][2])
                - self._sole_half_h - self._rest_offset)

    def _foot_contact(self, geom_id):
        """该脚 geom 是否与地面有接触(物理接触，比高度阈值可靠)。"""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 == geom_id and c.geom2 == self._floor_gid) or \
               (c.geom2 == geom_id and c.geom1 == self._floor_gid):
                return True
        return False

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        gid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
        self._foot_gid = {"l": gid("left_foot_geom"), "r": gid("right_foot_geom")}
        self._floor_gid = gid("floor")
        self._sole_half_h = float(self.model.geom_size[self._foot_gid["l"]][2])
        # 站稳后脚底中心比"半高"再高出约 0.02(接触 margin+软接触)，固定标定
        self._rest_offset = 0.0198
        # 注：不再用 reset 时刻的脚高作"地面基线"——reset 后几帧内脚踝会从
        # 0.050 沉降到 0.080，旧基线让平放的脚被判成"离地 3cm"(98% 时间误判腾空)
        self._start_xy = self.data.xpos[self._pelvis_id][:2].copy()
        self._air = {"l": False, "r": False}
        self._peak = {"l": 0.0, "r": 0.0}             # 本次摆动的最大离地高度
        self._swing_steps = {"l": 0, "r": 0}
        self._last_side = None                        # 上一次有效步是哪只脚
        self._last_valid_step = -10 ** 6
        self._steps_since_valid = 0
        self.step_counts = {"l": 0, "r": 0}           # 有效步计数(评估用)
        self.peak_log = {"l": [], "r": []}            # 有效步抬脚高度(评估用)
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        pelvis_h = float(self.data.xpos[self._pelvis_id][2])
        pelvis_up = self._upright(self._pelvis_id)
        torso_up = self._upright(self._torso_id)
        head_up = self._upright(self._head_id)
        clear = {s: self._sole_clear(self._foot_gid[s]) for s in ("l", "r")}
        z = clear   # 支撑腿判定用：哪只脚更低
        # 离地 = 无地面接触 且 脚底高于 air_margin(双重条件，滤掉接触抖动)
        air_now = {s: (not self._foot_contact(self._foot_gid[s]))
                   and clear[s] > self.air_margin for s in ("l", "r")}

        # 期望抬起的脚 = 上一步的对侧脚(第一步两只都可以)
        expected = ("l", "r") if self._last_side is None else \
            ("r",) if self._last_side == "l" else ("l",)

        # --- 落脚事件：判定有效步 ---
        step_r = 0.0
        for s in ("l", "r"):
            if air_now[s]:
                self._swing_steps[s] += 1
                self._peak[s] = max(self._peak[s], clear[s])
            elif self._air[s]:                        # 空中 → 触地
                interval = self.step_count - self._last_valid_step
                if s in expected and self._peak[s] >= self.lift_clear \
                        and interval >= self.min_step_interval:
                    # 分级：刚过门槛 ≈0.3 分，抬到 lift_target 拿满
                    quality = float(np.clip(
                        0.3 + 0.7 * (self._peak[s] - self.lift_clear)
                        / (self.lift_target - self.lift_clear), 0.3, 1.0))
                    step_r += self.step_bonus * quality
                    self._last_side = s
                    self._last_valid_step = self.step_count
                    self._steps_since_valid = 0
                    self.step_counts[s] += 1
                    self.peak_log[s].append(self._peak[s])
                elif s not in expected and self._peak[s] >= self.lift_clear:
                    step_r -= self.wrong_foot_pen     # 同一只脚连踩
                self._peak[s] = 0.0
                self._swing_steps[s] = 0
            self._air[s] = air_now[s]
        self._steps_since_valid += 1

        # --- 稠密引导：期望的那只脚抬起来(只在摆动前段给) ---
        lift_r = 0.0
        for s in expected:
            if self._swing_steps[s] <= self.max_swing_steps:
                lift_r = max(lift_r, float(np.clip(clear[s] / self.lift_target, 0.0, 1.0)))

        # --- 姿态 ---
        height_r = np.exp(-10.0 * (pelvis_h - self.init_pelvis_height) ** 2)
        rknee = float(self.data.qpos[self._rknee_adr])
        lknee = float(self.data.qpos[self._lknee_adr])
        support_knee = rknee if z["r"] < z["l"] else lknee
        support_straight_r = np.exp(-4.0 * support_knee ** 2)

        # --- 惩罚 ---
        both_air_pen = 3.0 if (air_now["l"] and air_now["r"]) else 0.0
        drift = float(np.linalg.norm(self.data.xpos[self._pelvis_id][:2] - self._start_xy))
        drift_pen = 2.0 * drift
        energy = 0.003 * np.sum(action ** 2)
        jitter = 0.01 * np.sum((action - self._prev_action) ** 2)
        self._prev_action = action.copy()

        # 站姿项整体降到约一半(原每步 ~5 分压过踏步收益)，抬脚引导 2.0→3.0
        reward = (0.3 + 0.5 * height_r + 0.8 * max(0.0, torso_up)
                  + 0.5 * max(0.0, head_up) + 0.3 * support_straight_r
                  + 3.0 * lift_r + step_r
                  - both_air_pen - drift_pen - energy - jitter)

        terminated = bool(pelvis_h < 0.55 or pelvis_up < 0.5
                          or torso_up < 0.5 or head_up < 0.5
                          or self._steps_since_valid >= self._stall_limit)
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "pelvis_height": pelvis_h,
            "torso_upright": torso_up,
            "head_upright": head_up,
            "forward_vel": 0.0,
            "drift": drift,
            "steps_l": self.step_counts["l"],
            "steps_r": self.step_counts["r"],
            "length": self.step_count,
        }
        return self._get_obs(), reward, terminated, truncated, info
