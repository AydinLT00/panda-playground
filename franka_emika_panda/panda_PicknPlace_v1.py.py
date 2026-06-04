import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import warp as wp
import mujoco
import mujoco_warp as mjw

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

wp.init()
wp.config.quiet = True
device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
wp.set_device(device_str)
torch_device = torch.device(device_str)

class PandaPickPlaceVecEnv(VecEnv):
    """GPU-Parallelized MuJoCo Environment for multi-object color-based Pick and Place."""
    def __init__(self, num_envs=2048, xml_path="/home/aidin/B/mujoco_project/panda_scripts/franka_emika_panda/scene_boxes.xml", max_steps=400):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.device = torch_device
        self.frame_skip = 10 

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.dt = self.frame_skip * self.mj_model.opt.timestep 

        # Obs Space = 42:
        # Arm qpos(7), Arm qvel(7), Fingers qpos(2), EE pos(3), Active Pick Pos(3), Active Place Pos(3), 
        # Red block(3), Green block(3), Blue block(3), Last Action(8)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32)
        # 8 actions: 7 for joints, 1 for continuous gripper control [-1 (open) to 1 (close)]
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)

        self.mjw_model = mjw.put_model(self.mj_model)
        self.mjw_data = mjw.put_data(self.mj_model, self.mj_data, nworld=self.num_envs, njmax=400, nconmax=150)

        # Lookup IDs
        self.hand_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.red_block_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
        self.green_block_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "green_block")
        self.blue_block_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "blue_block")
        self.black_pad_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "black_pad")
        self.white_pad_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "white_pad")

        self.key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        home_qpos_np = self.mj_model.key_qpos[self.key_id].copy()
        home_ctrl_np = self.mj_model.key_ctrl[self.key_id].copy()
        self.home_qpos_torch = torch.tensor(home_qpos_np, device=self.device, dtype=torch.float32)
        self.home_ctrl_torch = torch.tensor(home_ctrl_np, device=self.device, dtype=torch.float32)

        # Map Zero-Copy Tensors
        self.qpos_tensor = wp.to_torch(self.mjw_data.qpos)
        self.qvel_tensor = wp.to_torch(self.mjw_data.qvel)
        self.ctrl_tensor = wp.to_torch(self.mjw_data.ctrl)
        self.xpos_tensor = wp.to_torch(self.mjw_data.xpos)  # (nworld, nbody, 3)

        self.joint_limits_lower = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], device=self.device, dtype=torch.float32)
        self.joint_limits_upper = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], device=self.device, dtype=torch.float32)

        # Goal commands
        self.pick_target_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.place_target_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.command_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.resample_interval_steps = 300
        
        self.episode_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.current_action_tensor = torch.zeros((self.num_envs, 8), device=self.device, dtype=torch.float32)
        self.last_action_tensor = torch.zeros((self.num_envs, 8), device=self.device, dtype=torch.float32)
        self.last_dof_vel = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        self.action_scale = 0.08 

        self._resample_commands(torch.arange(self.num_envs, device=self.device))

        # Initial steps inside the captured graph
        mjw.step(self.mjw_model, self.mjw_data)
        print("Capturing CUDA simulation graph...")
        with wp.ScopedCapture() as capture:
            for _ in range(self.frame_skip):
                mjw.step(self.mjw_model, self.mjw_data)
        self.advance_sim_graph = capture.graph
        print("Graph Capture Complete!")

    def _resample_commands(self, env_ids):
        num_resample = len(env_ids)
        # Random pick: 0 = red, 1 = green, 2 = blue
        self.pick_target_ids[env_ids] = torch.randint(0, 3, (num_resample,), device=self.device)
        # Random place: 0 = black pad, 1 = white pad
        self.place_target_ids[env_ids] = torch.randint(0, 2, (num_resample,), device=self.device)
        self.command_steps[env_ids] = 0

    def step_async(self, actions):
        action_tensor = torch.tensor(actions, device=self.device, dtype=torch.float32)
        current_joint_angles = self.qpos_tensor[:, 0:7]
        target_angles = current_joint_angles + (action_tensor[:, 0:7] * self.action_scale)
        target_angles = torch.clamp(target_angles, self.joint_limits_lower, self.joint_limits_upper)
        
        # Arm control
        self.ctrl_tensor[:, :7] = target_angles
        
        # Gripper control: map [-1, 1] to [0, 255]
        self.ctrl_tensor[:, 7] = (action_tensor[:, 7] + 1.0) * 127.5
        
        self.last_action_tensor = self.current_action_tensor.clone()
        self.current_action_tensor = action_tensor.clone()

    def step_wait(self):
        self.last_dof_vel = self.qvel_tensor[:, 0:7].clone()
        wp.capture_launch(self.advance_sim_graph)
        wp.synchronize_device() 

        # Fetch block and pad coordinate positions dynamically
        red_pos = self.xpos_tensor[:, self.red_block_id]
        green_pos = self.xpos_tensor[:, self.green_block_id]
        blue_pos = self.xpos_tensor[:, self.blue_block_id]
        blocks_pos = torch.stack([red_pos, green_pos, blue_pos], dim=1)

        black_pad = self.xpos_tensor[:, self.black_pad_id]
        white_pad = self.xpos_tensor[:, self.white_pad_id]
        pads_pos = torch.stack([black_pad, white_pad], dim=1)

        # Map to individual environment active targets
        batch_indices = torch.arange(self.num_envs, device=self.device)
        active_pick_pos = blocks_pos[batch_indices, self.pick_target_ids]
        active_place_pos = pads_pos[batch_indices, self.place_target_ids]

        ee_pos = self.xpos_tensor[:, self.hand_id]
        self.episode_steps += 1
        self.command_steps += 1

        # Resample commands if interval reached
        resample_mask = self.command_steps >= self.resample_interval_steps
        if resample_mask.any():
            self._resample_commands(torch.where(resample_mask)[0])

        # ----------------------------------------------------
        # STAGED REWARD SCHEME
        # ----------------------------------------------------
        dist_ee_to_block = torch.norm(active_pick_pos - ee_pos, dim=-1)
        dist_block_to_pad = torch.norm(active_pick_pos - active_place_pos, dim=-1)

        # Detect block lift (initial z is ~0.025m, >0.06m indicates a lift)
        is_lifted = (active_pick_pos[:, 2] > 0.06).float()

        # Multi-stage rewards
        r_reach = -dist_ee_to_block
        r_place = -dist_block_to_pad
        
        # Base rewards combining tracking, regularization and smooth acceleration
        joint_vel = self.qvel_tensor[:, 0:7]
        r_joint_vel = -1e-2 * torch.sum(torch.square(joint_vel), dim=-1)
        r_posture = -1e-1 * torch.sum(torch.square(self.qpos_tensor[:, 0:7] - self.home_qpos_torch[0:7]), dim=-1)

        rewards = (1.0 - is_lifted) * r_reach + is_lifted * (5.0 + 3.0 * r_place) + r_joint_vel + r_posture

        # NaNs protection
        is_exploded = torch.any(torch.isnan(self.qpos_tensor) | torch.isinf(self.qpos_tensor), dim=-1)
        terminated = is_exploded
        truncated = self.episode_steps >= self.max_steps
        needs_reset = terminated | truncated

        rewards -= 100.0 * terminated.float()

        obs = torch.cat([
            self.qpos_tensor[:, 0:7],       # 7
            joint_vel,                      # 7
            self.qpos_tensor[:, 7:9],       # 2 (fingers)
            ee_pos,                         # 3
            active_pick_pos,                # 3
            active_place_pos,               # 3
            red_pos,                        # 3
            green_pos,                      # 3
            blue_pos,                       # 3
            self.current_action_tensor      # 8
        ], dim=-1)

        infos = [{} for _ in range(self.num_envs)]
        if needs_reset.any():
            reset_indices = torch.where(needs_reset)[0]
            for idx in reset_indices:
                if self.episode_steps[idx] >= self.max_steps:
                    infos[idx]["TimeLimit.truncated"] = True

            # Reset robot to home configuration
            self.qpos_tensor[reset_indices, 0:9] = self.home_qpos_torch[0:9].clone()
            self.qvel_tensor[reset_indices] = 0.0

            # Randomize positions of the dynamic blocks on reset to prevent overfitting
            num_resets = len(reset_indices)
            for i, body_idx in enumerate([9, 16, 23]): # slices for block qpos freejoints
                rand_x = torch.rand(num_resets, device=self.device) * 0.2 + 0.35
                rand_y = torch.rand(num_resets, device=self.device) * 0.3 - 0.15
                self.qpos_tensor[reset_indices, body_idx:body_idx+3] = torch.stack([rand_x, rand_y, torch.ones_like(rand_x)*0.025], dim=-1)
                self.qpos_tensor[reset_indices, body_idx+3:body_idx+7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

            self.episode_steps[reset_indices] = 0
            self.current_action_tensor[reset_indices] = 0.0
            self.last_action_tensor[reset_indices] = 0.0
            self.last_dof_vel[reset_indices] = 0.0
            self._resample_commands(reset_indices)

            mjw.forward(self.mjw_model, self.mjw_data)

            # Re-fetch positions for observation construction
            red_pos_r = self.xpos_tensor[:, self.red_block_id]
            green_pos_r = self.xpos_tensor[:, self.green_block_id]
            blue_pos_r = self.xpos_tensor[:, self.blue_block_id]
            blocks_pos_r = torch.stack([red_pos_r, green_pos_r, blue_pos_r], dim=1)
            active_pick_pos_r = blocks_pos_r[batch_indices, self.pick_target_ids]

            obs = torch.cat([
                self.qpos_tensor[:, 0:7],
                self.qvel_tensor[:, 0:7],
                self.qpos_tensor[:, 7:9],
                self.xpos_tensor[:, self.hand_id],
                active_pick_pos_r,
                pads_pos[batch_indices, self.place_target_ids],
                red_pos_r,
                green_pos_r,
                blue_pos_r,
                self.current_action_tensor
            ], dim=-1)

        return obs.cpu().numpy(), rewards.cpu().numpy(), needs_reset.cpu().numpy(), infos

    def reset(self):
        mjw.reset_data(self.mjw_model, self.mjw_data)
        self.qpos_tensor[:, 0:9] = self.home_qpos_torch[0:9].clone()
        self.qvel_tensor[:, :] = 0.0
        self.episode_steps[:] = 0
        self.current_action_tensor[:, :] = 0.0
        self.last_action_tensor[:, :] = 0.0
        self.last_dof_vel[:, :] = 0.0
        self._resample_commands(torch.arange(self.num_envs, device=self.device))
        mjw.step(self.mjw_model, self.mjw_data)
        obs, _, _, _ = self.step_wait()
        return obs

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        if hasattr(self, attr_name):
            val = getattr(self, attr_name)
            return [val] * (self.num_envs if indices is None else len(indices))
        raise AttributeError(f"Attribute '{attr_name}' not found.")

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return [method(*method_args, **method_kwargs)] * (self.num_envs if indices is None else len(indices))
        raise AttributeError(f"Method '{method_name}' not found.")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * (self.num_envs if indices is None else len(indices))


if __name__ == "__main__":
    # Standard StableBaselines3 training script entry
    log_dir = "./tb_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("Initializing parallel envs...")
    raw_env = PandaPickPlaceVecEnv(num_envs=1024)
    env = VecMonitor(raw_env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=4096,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log=log_dir
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, 
        save_path="./models_pick_place/",
        name_prefix="panda_pick_place_ppo"
    )

    print("Starting Training...")
    model.learn(total_timesteps=15_000_000, callback=checkpoint_callback, tb_log_name="pick_place_run")
    model.save("panda_pick_place_model")