import os
import threading
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

# 1. Base Scene XML content
SCENE_XML_CONTENT = """<mujoco model="panda scene">
  <include file="panda.xml"/>

  <statistic center="0.3 0 0.4" extent="1"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""

def prepare_test_scene_xml(input_content=SCENE_XML_CONTENT, output_path="/home/aidin/B/mujoco_project/panda_scripts/franka_emika_panda/scene_test.xml"):
    # Inject a non-colliding mocap sphere to act as our real-time target visualizer
    target_body_xml = """
    <body name="target" pos="0 0 0.5" mocap="true">
      <geom type="sphere" size="0.02" rgba="0 1 0 0.6" contype="0" conaffinity="0"/>
    </body>
    """
    
    # Inject target body inside scene's <worldbody>
    xml_text = input_content.replace("<worldbody>", "<worldbody>" + target_body_xml)
        
    with open(output_path, "w") as f:
        f.write(xml_text)
    print(f"Created temporary testing scene XML at '{output_path}' with visual target tracker.")


class PandaSceneTesterApp:
    def __init__(self, model_path="panda_warp_ik_model2"):
        # Load the PPO policy
        print(f"Loading trained model from '{model_path}'...")
        self.model = PPO.load(model_path)
        
        # Prepare and load the modified scene XML (removed scene.xml overwrite)
        prepare_test_scene_xml()
        self.mj_model = mujoco.MjModel.from_xml_path("/home/aidin/B/mujoco_project/panda_scripts/franka_emika_panda/scene_test.xml")
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # Set to keyframe home position
        self.key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.key_id)
        
        # Identify tracking bodies
        self.hand_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.target_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.mocap_id = self.mj_model.body_mocapid[self.target_body_id]
        
        # Joint limit configurations (Joints 1 to 7)
        self.joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # Shared target state (x, y, z)
        self.target_x = 0.3
        self.target_y = 0.0
        self.target_z = 0.4
        
        # Simulation state trackers
        self.running = True
        self.action_scale = 0.08  # --- FIX: Changed from 0.05 to 0.08 to match training scale ---
        self.frame_skip = 10
        self.last_action = np.zeros(7, dtype=np.float32)

        # Launch the passive visual viewer
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        
        # Start the physics simulation loop in a background thread
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()
        
        # Initialize the Tkinter Control Window
        self._build_gui()

    def _simulation_loop(self):
        """Simulation and command tracking loop running in a background thread."""
        while self.running and self.viewer.is_running():
            step_start = time.time()
            
            # Retrieve current target coordinates
            tx, ty, tz = self.target_x, self.target_y, self.target_z
            target_pos = np.array([tx, ty, tz], dtype=np.float32)
            
            # --- FIX: Acquire the viewer lock before modifying physics data ---
            with self.viewer.lock():
                # 1. Update target marker position in MuJoCo
                self.mj_data.mocap_pos[self.mocap_id] = target_pos
                
                # 2. Extract state observations matching the training layout
                qpos = self.mj_data.qpos
                qvel = self.mj_data.qvel
                ee_pos = self.mj_data.xpos[self.hand_id]
                dist_vector = target_pos - ee_pos
                
                obs = np.concatenate([
                    qpos[0:7],            # (7) Current joint angles
                    qvel[0:7],            # (7) Current joint velocities
                    ee_pos,               # (3) Hand position
                    target_pos,           # (3) Target coordinates
                    dist_vector,          # (3) Delta distance vector
                    self.last_action      # (7) Action history
                ]).astype(np.float32)
                
                # 3. Model Inference (Deterministic prediction)
                action, _ = self.model.predict(obs, deterministic=True)
                self.last_action = action.copy()
                
                # 4. Integrate joint commands (Delta Target Control)
                current_angles = qpos[0:7]
                target_angles = current_angles + (action * self.action_scale)
                target_angles = np.clip(target_angles, self.joint_limits_lower, self.joint_limits_upper)
                
                # 5. Apply commands to MuJoCo Actuators
                self.mj_data.ctrl[0:7] = target_angles
                self.mj_data.ctrl[7] = 255.0  # Keep fingers closed
                
                # 6. Step the simulator frame_skip times
                for _ in range(self.frame_skip):
                    mujoco.mj_step(self.mj_model, self.mj_data)
                
            # Sync rendering
            self.viewer.sync()
            
            # --- FIX: Maintained 50Hz control loop speed (0.02s step size) to avoid CPU bottleneck ---
            elapsed = time.time() - step_start
            time.sleep(max(0.001, 0.02 - elapsed))

    def _build_gui(self):
        """Constructs a Tkinter slider window on the main thread."""
        self.root = tk.Tk()
        self.root.title("Franka Panda RL-IK Controller")
        self.root.geometry("400x350")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        style = ttk.Style()
        style.theme_use('clam')

        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Move Sliders to Direct the End-Effector", font=("Helvetica", 12, "bold")).pack(pady=10)

        # Slider configurations
        self._create_slider(main_frame, "Target X (Forward/Backward)", -0.5, 0.5, self.target_x, "x")
        self._create_slider(main_frame, "Target Y (Left/Right)", -0.5, 0.5, self.target_y, "y")
        self._create_slider(main_frame, "Target Z (Up/Down)", 0.1, 0.6, self.target_z, "z")

        # Live coordinates readout labels
        self.readout_label = ttk.Label(main_frame, text="EE Target: [0.30, 0.00, 0.40] | Current: [..., ..., ...]", font=("Courier", 9))
        self.readout_label.pack(pady=15)

        self._update_labels()
        self.root.mainloop()

    def _create_slider(self, parent, label_text, min_val, max_val, init_val, axis_name):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)

        ttk.Label(frame, text=label_text, width=28, anchor=tk.W).pack(side=tk.LEFT)
        
        # Display coordinate value adjacent to slider
        val_label = ttk.Label(frame, text=f"{init_val:.2f}", width=6)
        
        def on_slider_move(val):
            val_float = float(val)
            val_label.config(text=f"{val_float:.2f}")
            if axis_name == "x":
                self.target_x = val_float
            elif axis_name == "y":
                self.target_y = val_float
            elif axis_name == "z":
                self.target_z = val_float

        slider = ttk.Scale(frame, from_=min_val, to=max_val, value=init_val, orient=tk.HORIZONTAL, command=on_slider_move)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        val_label.pack(side=tk.RIGHT)

    def _update_labels(self):
        """Reads back coordinates from the physical state simulation to display tracking error."""
        if self.running and self.viewer.is_running():
            # --- FIX: Acquire viewer lock since we are reading from the Tkinter GUI thread ---
            with self.viewer.lock():
                ee_pos = self.mj_data.xpos[self.hand_id].copy()
                
            self.readout_label.config(
                text=f"Target: [{self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}]\n"
                     f"Current EE: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]"
            )
            self.root.after(100, self._update_labels)

    def on_close(self):
        """Clean shutdown operations."""
        print("Shutting down tester environment...")
        self.running = False
        self.viewer.close()
        self.root.destroy()
        # Clean up the generated visual model test file
        test_xml = "/home/aidin/B/mujoco_project/panda_scripts/franka_emika_panda/scene_test.xml"
        if os.path.exists(test_xml):
            os.remove(test_xml)


if __name__ == "__main__":
    app = PandaSceneTesterApp()