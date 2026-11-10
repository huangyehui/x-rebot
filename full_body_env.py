"""完整人形站立环境

在已验证可稳定站立的下半身基座上，叠加躯干+双臂+脖子+头(共 16 自由度)。
目标：加上半身重量后，PPO 仍能学会稳定站立且上半身直立、头部稳定。
观测/奖励在下半身基础上增加躯干直立和头部高度项。
"""
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class FullBodyStandEnv(gym.Env):
    """完整人形站立任务环境"""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, disturb=False,
                 push_interval=100, push_force=120.0, reset_noise=0.02):
        super().__init__()
        self.render_mode = render_mode

        # 抗扰训练开关(默认关，保持原站立环境不变)：
        #   disturb        是否启用外力扰动
        #   push_interval  每隔多少步随机推一次
        #   push_force     推力量级(N)，作用于躯干，方向随机(水平面)
        #   reset_noise    reset 时的关节/速度随机化幅度(越大起始姿态越多样)
        self.disturb = disturb
        self.push_interval = push_interval
        self.push_force = push_force
        self.reset_noise = reset_noise
        self._push_steps_left = 0   # 当前推力剩余作用步数

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

        # freejoint 占 qpos 前 7 维、qvel 前 6 维
        self._nq_joint = self.model.nq - 7
        self._nv_joint = self.model.nv - 6

        self.n_actuators = self.model.nu
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_actuators,), dtype=np.float32)

        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float64)

        self._renderer = None
        self._prev_action = np.zeros(self.n_actuators)

    def _upright(self, body_id):
        return float(self.data.xmat[body_id].reshape(3, 3)[2, 2])

    def _get_obs(self):
        """观测：关节角/速 + 骨盆/躯干姿态 + 高度 + 质心相对支撑中心偏移"""
        com = self.data.subtree_com[0][:2]
        lf = self.data.xpos[self._lfoot_id][:2]
        rf = self.data.xpos[self._rfoot_id][:2]
        mid = 0.5 * (lf + rf)
        pelvis_h = self.data.xpos[self._pelvis_id][2]
        head_h = self.data.xpos[self._head_id][2]
        return np.concatenate([
            self.data.qpos[7:].copy(),          # 关节角
            self.data.qvel[6:].copy(),          # 关节角速度
            self.data.qvel[:6].copy(),          # 骨盆线速度+角速度
            [pelvis_h, head_h,
             self._upright(self._pelvis_id),
             self._upright(self._torso_id),
             self._upright(self._head_id)],
            com - mid,                          # 质心相对支撑中心(xy)
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = self.init_pelvis_height
        # 抗扰训练用更大的 reset 随机化(domain randomization)，增强泛化
        nq_noise = self.reset_noise
        self.data.qpos[7:] += self.np_random.uniform(-nq_noise, nq_noise, self._nq_joint)
        self.data.qvel[:] += self.np_random.uniform(-nq_noise, nq_noise, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self._push_steps_left = 0
        self.data.xfrc_applied[:] = 0.0
        self._prev_action = np.zeros(self.n_actuators)
        return self._get_obs(), {}

    def set_push_force(self, force):
        """外部(课程式回调)动态调整推力量级。"""
        self.push_force = float(force)

    def _apply_disturbance(self):
        """每隔 push_interval 步，对躯干施加一次随机方向的水平推力(持续数步)。"""
        if not self.disturb:
            return
        if self._push_steps_left > 0:
            self._push_steps_left -= 1
            if self._push_steps_left == 0:
                self.data.xfrc_applied[self._torso_id, :3] = 0.0
        elif self.step_count > 0 and self.step_count % self.push_interval == 0:
            theta = float(self.np_random.uniform(0, 2 * np.pi))
            mag = float(self.np_random.uniform(0.5, 1.0)) * self.push_force
            fx, fy = mag * np.cos(theta), mag * np.sin(theta)
            self.data.xfrc_applied[self._torso_id, :3] = [fx, fy, 0.0]
            self._push_steps_left = 5   # 推力持续 5 个 env step

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._apply_disturbance()
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        pelvis_h = self.data.xpos[self._pelvis_id][2]
        head_h = self.data.xpos[self._head_id][2]
        pelvis_up = self._upright(self._pelvis_id)
        torso_up = self._upright(self._torso_id)
        head_up = self._upright(self._head_id)
        com = self.data.subtree_com[0][:2]
        lf = self.data.xpos[self._lfoot_id][:2]
        rf = self.data.xpos[self._rfoot_id][:2]
        com_err = float(np.linalg.norm(com - 0.5 * (lf + rf)))

        # 奖励：活着 + 骨盆高度 + 头部高度 + 骨盆/躯干/头部直立 + 质心居中 + 静止 - 能耗/抖动
        # 强化上身姿态：头部始终竖直(head_up 权重最高)、躯干挺直、头部到位
        height_r = np.exp(-10.0 * (pelvis_h - self.init_pelvis_height) ** 2)
        head_r = np.exp(-10.0 * (head_h - self.init_head_height) ** 2)
        pelvis_up_r = max(0.0, pelvis_up)
        torso_up_r = max(0.0, torso_up)
        head_up_r = max(0.0, head_up)
        com_r = np.exp(-20.0 * com_err ** 2)
        joint_vel = self.data.qvel[6:]
        still_r = np.exp(-0.05 * np.sum(joint_vel ** 2))
        energy = 0.003 * np.sum(action ** 2)
        jitter = 0.01 * np.sum((action - self._prev_action) ** 2)
        self._prev_action = action.copy()

        reward = (1.0 + 2.0 * height_r + 3.0 * head_r
                  + 2.0 * pelvis_up_r + 4.0 * torso_up_r + 4.0 * head_up_r
                  + 2.0 * com_r + 1.0 * still_r - energy - jitter)

        # 终止：骨盆过低、躯干/骨盆/头部明显倾倒(头部必须保持竖直)
        terminated = bool(pelvis_h < 0.55 or pelvis_up < 0.5
                          or torso_up < 0.5 or head_up < 0.5)
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "pelvis_height": pelvis_h,
            "head_height": head_h,
            "pelvis_upright": pelvis_up,
            "torso_upright": torso_up,
            "head_upright": head_up,
            "upright": torso_up,
            "com_err": com_err,
            "reward": reward,
            "length": self.step_count,
            "torso_height": pelvis_h,  # 兼容可视化回调
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="track")
            return self._renderer.render()
        return None
