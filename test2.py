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

# 1. Initialize Warp on GPU
wp.init()
wp.config.quiet = True
device_str = "cuda:0"
wp.set_device(device_str)
torch_device = torch.device(device_str)


class PandaWarpVecEnv(VecEnv):
    """GPU-Parallelized MuJoCo Environment for Learned End-Effector Cartesian Target Tracking."""
    
    def __init__(self, num_envs=2048, xml_path="/home/aidin/B/mujoco_project/panda_scripts/franka_emika_panda/scene.xml", max_steps=500):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.device = torch_device
        self.frame_skip = 10  # 500Hz physics -> 50Hz control loop

        # 2. Load and Compile Model
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.dt = self.frame_skip * self.mj_model.opt.timestep  # 0.02s

        # Observation Space (Size 30)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)

        # 5. Bind Parallel Data on GPU Device
        self.mjw_model = mjw.put_model(self.mj_model)
        self.mjw_data = mjw.put_data(self.mj_model, self.mj_data, nworld=self.num_envs,njmax=280,nconmax=128)

        # Body tracking IDs
        self.hand_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        
        # Extract initial/home configurations
        home_qpos_np = self.mj_model.key_qpos[self.key_id].copy()
        home_ctrl_np = self.mj_model.key_ctrl[self.key_id].copy()
        self.home_qpos_torch = torch.tensor(home_qpos_np, device=self.device, dtype=torch.float32)
        self.home_ctrl_torch = torch.tensor(home_ctrl_np, device=self.device, dtype=torch.float32)

        # Zero-Copy PyTorch Tensors
        self.qpos_tensor = wp.to_torch(self.mjw_data.qpos)
        self.qvel_tensor = wp.to_torch(self.mjw_data.qvel)
        self.ctrl_tensor = wp.to_torch(self.mjw_data.ctrl)
        self.xpos_tensor = wp.to_torch(self.mjw_data.xpos)  # (nworld, nbody, 3)

        # Explicit limits of Panda's 7 hinges
        self.joint_limits_lower = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], device=self.device, dtype=torch.float32)
        self.joint_limits_upper = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], device=self.device, dtype=torch.float32)

        # Target (x, y, z) Commands & Resampling Tracker
        self.commands = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.command_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.resample_interval_steps = 200  # Resample target goal every 4 seconds (200 steps)
        
        # Environment Trackers
        self.episode_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.current_action_tensor = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        self.last_action_tensor = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        self.last_dof_vel = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        
        # Velocity scale modifier
        self.action_scale = 0.08 

        # Initial Goal Commands Generation
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

        # Capture Physics Step Loop with CUDA Graphs
        mjw.step(self.mjw_model, self.mjw_data)
        
        print("Capturing CUDA simulation graph...")
        with wp.ScopedCapture() as capture:
            for _ in range(self.frame_skip):
                mjw.step(self.mjw_model, self.mjw_data)
        self.advance_sim_graph = capture.graph
        print("CUDA Graph Capture Complete!")

    def _resample_commands(self, env_ids):
        """Generates random target coordinate command vectors."""
        num_resample = len(env_ids)
        r = torch.rand(num_resample, device=self.device) * 0.45 + 0.25 # radius between 0.25m and 0.70m
        theta = torch.rand(num_resample, device=self.device) * np.pi * 0.7
        phi = torch.rand(num_resample, device=self.device) * 2.0 * np.pi
        
        dx = r * torch.sin(theta) * torch.cos(phi)
        dy = r * torch.sin(theta) * torch.sin(phi)
        dz = r * torch.cos(theta)
        
        self.commands[env_ids, 0] = dx
        self.commands[env_ids, 1] = dy
        self.commands[env_ids, 2] = dz + 0.333
        
        self.commands[env_ids, 2] = torch.clamp(self.commands[env_ids, 2], min=0.1)
        self.command_steps[env_ids] = 0

    def step_async(self, actions):
        action_tensor = torch.tensor(actions, device=self.device, dtype=torch.float32)
        
        current_joint_angles = self.qpos_tensor[:, 0:7]
        target_angles = current_joint_angles + (action_tensor * self.action_scale)
        target_angles = torch.clamp(target_angles, self.joint_limits_lower, self.joint_limits_upper)
        
        self.ctrl_tensor[:, :7] = target_angles
        self.ctrl_tensor[:, 7] = 255.0  # Kept Closed
        
        self.last_action_tensor = self.current_action_tensor.clone()
        self.current_action_tensor = action_tensor.clone()

    def step_wait(self):
        self.last_dof_vel = self.qvel_tensor[:, 0:7].clone()

        wp.capture_launch(self.advance_sim_graph)
        wp.synchronize_device() 

        qpos = self.qpos_tensor
        qvel = self.qvel_tensor
        
        # Retrieve Cartesian coordinate tracking position of EE ("hand")
        ee_pos = self.xpos_tensor[:, self.hand_id]
        joint_vel = qvel[:, 0:7]
        self.episode_steps += 1
        self.command_steps += 1

        # Resample targets periodically
        resample_mask = self.command_steps >= self.resample_interval_steps
        if resample_mask.any():
            resample_ids = torch.where(resample_mask)[0]
            self._resample_commands(resample_ids)

        # ----------------------------------------------------
        # REWARD FUNCTIONS & PENALTIES (Re-scaled for smooth trajectories)
        # ----------------------------------------------------
        dist_vector = self.commands - ee_pos
        dist = torch.norm(dist_vector, p=2, dim=-1)
        r_dist = -dist
        
        # Precision exponential alignment reward
        r_tracking_ee = torch.exp(-torch.sum(torch.square(dist_vector), dim=-1) / 0.02)
        
        # Regularize kinetic velocity (scaled up 100x from 1e-4 -> 1e-2)
        r_joint_vel = -1e-2 * torch.sum(torch.square(joint_vel), dim=-1)
        
        # Joint acceleration penalty (scaled up 100x from 1e-7 -> 1e-5)
        r_joint_acc = -1e-5 * torch.sum(torch.square((joint_vel - self.last_dof_vel) / self.dt), dim=-1)
        
        # Smooth action trajectory penalty (scaled up 10x from 1e-3 -> 1e-2)
        r_action_rate = -1e-2 * torch.sum(torch.square(self.current_action_tensor - self.last_action_tensor), dim=-1)

        # Posture regularizer (encourages arm to remain near natural default configuration)
        r_posture = -1e-1 * torch.sum(torch.square(qpos[:, 0:7] - self.home_qpos_torch[0:7]), dim=-1)

        # Exception safety resets (NaN protection)
        is_exploded = torch.any(torch.isnan(qpos) | torch.isinf(qpos), dim=-1) | torch.any(torch.isnan(ee_pos), dim=-1)
        
        terminated = is_exploded
        truncated = self.episode_steps >= self.max_steps
        needs_reset = terminated | truncated

        r_termination = -100.0 * terminated.float()

        rewards = (
            5.0 * r_tracking_ee + 
            2.0 * r_dist + 
            r_joint_vel + 
            r_joint_acc + 
            r_action_rate +
            r_posture +
            r_termination
        )

        obs = torch.cat([
            qpos[:, 0:7],                      # (7) Current joint positions
            joint_vel,                         # (7) Current joint velocities
            ee_pos,                            # (3) Hand body positions
            self.commands,                     # (3) Target goal positions
            dist_vector,                       # (3) Vector distance to target
            self.current_action_tensor         # (7) Last actions taken
        ], dim=-1)

        infos = [{} for _ in range(self.num_envs)]
        
        if needs_reset.any():
            reset_indices = torch.where(needs_reset)[0]
            
            terminal_obs = obs.clone()
            for idx in reset_indices:
                infos[idx]["terminal_observation"] = terminal_obs[idx].cpu().numpy()
                if self.episode_steps[idx] >= self.max_steps:
                    infos[idx]["TimeLimit.truncated"] = True

            # Perform GPU State Reset updates
            self.qpos_tensor[reset_indices] = self.home_qpos_torch.clone()
            
            # Add positioning noise
            noise_pos = (torch.rand((len(reset_indices), 7), device=self.device) - 0.5) * 0.05
            self.qpos_tensor[reset_indices, 0:7] += noise_pos
            self.qpos_tensor[reset_indices, 7:9] = 0.04
            self.qvel_tensor[reset_indices] = (torch.rand((len(reset_indices), 9), device=self.device) - 0.5) * 0.02
            
            self.episode_steps[reset_indices] = 0
            self.current_action_tensor[reset_indices] = 0.0
            self.last_action_tensor[reset_indices] = 0.0
            self.last_dof_vel[reset_indices] = 0.0
            
            self._resample_commands(reset_indices)

            # --- FIX 1: Run kinematics pass on GPU to update self.xpos_tensor ---
            mjw.forward(self.mjw_model, self.mjw_data)

            # --- FIX 2: Removed "rewards[reset_indices] = 0.0" ---
            # Storing the actual computed terminal rewards allows the agent 
            # to learn from its terminations/failures.

            qpos_reset = self.qpos_tensor
            qvel_reset = self.qvel_tensor
            ee_pos_reset = self.xpos_tensor[:, self.hand_id]
            dist_vector_reset = self.commands - ee_pos_reset

            # Update observation block specifically for reset contexts
            obs = torch.cat([
                qpos_reset[:, 0:7],
                qvel_reset[:, 0:7],
                ee_pos_reset,
                self.commands,
                dist_vector_reset,
                self.current_action_tensor
            ], dim=-1)

        return (
            obs.cpu().numpy(),
            rewards.cpu().numpy(),
            needs_reset.cpu().numpy(),
            infos
        )

    def reset(self):
        mjw.reset_data(self.mjw_model, self.mjw_data)
        self.qpos_tensor[:, :] = self.home_qpos_torch.clone()
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
            num = self.num_envs if indices is None else len(indices)
            return [val] * num
        raise AttributeError(f"PandaWarpVecEnv has no attribute '{attr_name}'")

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            res = method(*method_args, **method_kwargs)
            num = self.num_envs if indices is None else len(indices)
            return [res] * num
        raise AttributeError(f"PandaWarpVecEnv has no method '{method_name}'")

    def env_is_wrapped(self, wrapper_class, indices=None):
        num = self.num_envs if indices is None else len(indices)
        return [False] * num


if __name__ == "__main__":
    log_dir = "./tb_logs/"
    os.makedirs(log_dir, exist_ok=True)

    num_parallel_envs = 2048
    print(f"Initializing {num_parallel_envs} parallel environments...")
    
    raw_env = PandaWarpVecEnv(num_envs=num_parallel_envs)
    env = VecMonitor(raw_env)

    # Clean multi-environment MLP Policy configuration
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,            # --- FIX 3: Increased from 32 to 128 for longer GAE horizon ---
        batch_size=4096,       
        n_epochs=5,
        gamma=0.98,            
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,        
        verbose=1,
        tensorboard_log=log_dir
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=150_000, 
        save_path="./models_panda/",
        name_prefix="panda_warp_ppo_ik"
    )

    print("Training Panda Cartesian coordinates tracking!")
    model.learn(
        total_timesteps=20_000_000,
        callback=checkpoint_callback,
        tb_log_name="panda_warp_ik_tracking"
    )

    model.save("panda_warp_ik_model2")
    print("Training completed.")