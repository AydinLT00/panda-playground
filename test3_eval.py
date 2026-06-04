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


class PhysicsInformedPandaTesterApp:
    def __init__(self, model_path="panda_physics_informed_model"):
        # Load the trained physics-informed policy
        print(f"Loading trained model from '{model_path}'...")
        self.model = PPO.load(model_path)
        
        # Prepare and load the modified scene XML
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
        
        # ----------------------------------------------------
        # PHYSICS-INFORMED EVALUATION PARAMETERS (Match Training Environment)
        # ----------------------------------------------------
        self.frame_skip = 10
        self.dt = self.frame_skip * self.mj_model.opt.timestep  # 0.02s (50Hz control loop)
        self.max_joint_acc = 15.0  # rad/s^2
        self.joint_vel_limits = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610]) # rad/s
        
        # Tracking states
        self.integrated_vel = np.zeros(7, dtype=np.float32)
        self.last_action = np.zeros(7, dtype=np.float32)
        
        # Shared target state (x, y, z)
        self.target_x = 0.3
        self.target_y = 0.0
        self.target_z = 0.4
        
        self.running = True

        # Launch passive viewer
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        
        # Start physics background loop
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()
        
        # Initialize GUI Control Window
        self._build_gui()

    def _simulation_loop(self):
        """Simulation and command tracking loop running in a background thread."""
        while self.running and self.viewer.is_running():
            step_start = time.time()
            
            tx, ty, tz = self.target_x, self.target_y, self.target_z
            target_pos = np.array([tx, ty, tz], dtype=np.float32)
            
            # Acquire viewer lock before reading/modifying physical state
            with self.viewer.lock():
                # 1. Update target marker coordinates
                self.mj_data.mocap_pos[self.mocap_id] = target_pos
                
                # 2. Extract physics observations (matching size 37 structure)
                qpos = self.mj_data.qpos
                qvel = self.mj_data.qvel
                ee_pos = self.mj_data.xpos[self.hand_id]
                dist_vector = target_pos - ee_pos
                
                obs = np.concatenate([
                    qpos[0:7],            # (7) Joint angles
                    qvel[0:7],            # (7) Joint velocities
                    self.integrated_vel,  # (7) Integrated velocity targets
                    ee_pos,               # (3) Hand position
                    target_pos,           # (3) Target coordinates
                    dist_vector,          # (3) Target distance vector
                    self.last_action      # (7) Last actions (accelerations)
                ]).astype(np.float32)
                
                # 3. Model Inference (Predict Acceleration command ddq)
                action, _ = self.model.predict(obs, deterministic=True)
                
                # 4. Physics-Informed Acceleration Integration: v_new = v_old + a * dt
                joint_acc = action * self.max_joint_acc
                new_joint_vel = self.integrated_vel + (joint_acc * self.dt)
                # Apply physical speed bounds
                new_joint_vel = np.clip(new_joint_vel, -self.joint_vel_limits, self.joint_vel_limits)
                
                # 5. Position Integration: q_new = q_old + v_new * dt
                current_angles = qpos[0:7]
                target_angles = current_angles + (new_joint_vel * self.dt)
                target_angles = np.clip(target_angles, self.joint_limits_lower, self.joint_limits_upper)
                
                # Save internal control parameters for next step
                self.integrated_vel = new_joint_vel.copy()
                self.last_action = action.copy()
                
                # 6. Apply target joint positions to MuJoCo
                self.mj_data.ctrl[0:7] = target_angles
                self.mj_data.ctrl[7] = 255.0  # Keep fingers closed
                
                # 7. Step physics frame_skip times
                for _ in range(self.frame_skip):
                    mujoco.mj_step(self.mj_model, self.mj_data)
                
            # Sync rendering
            self.viewer.sync()
            
            # Maintain 50Hz control loop speed
            elapsed = time.time() - step_start
            time.sleep(max(0.002, 0.02 - elapsed))

    def _build_gui(self):
        """Constructs a Tkinter slider window on the main thread."""
        self.root = tk.Tk()
        self.root.title("Physics-Informed Panda Controller")
        self.root.geometry("400x350")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        style = ttk.Style()
        style.theme_use('clam')

        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Move Sliders to Direct the End-Effector", font=("Helvetica", 12, "bold")).pack(pady=10)

        # Target Coordinate Sliders
        self._create_slider(main_frame, "Target X (Forward/Backward)", -0.6, 0.6, self.target_x, "x")
        self._create_slider(main_frame, "Target Y (Left/Right)", -0.6, 0.6, self.target_y, "y")
        self._create_slider(main_frame, "Target Z (Up/Down)", 0.1, 0.8, self.target_z, "z")

        # Live coordinate error readout
        self.readout_label = ttk.Label(main_frame, text="Target: [0.30, 0.00, 0.40] | Current: [..., ..., ...]", font=("Courier", 9))
        self.readout_label.pack(pady=15)

        self._update_labels()
        self.root.mainloop()

    def _create_slider(self, parent, label_text, min_val, max_val, init_val, axis_name):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)

        ttk.Label(frame, text=label_text, width=28, anchor=tk.W).pack(side=tk.LEFT)
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
        """Reads back coordinates with thread locks to display tracking error."""
        if self.running and self.viewer.is_running():
            with self.viewer.lock():
                ee_pos = self.mj_data.xpos[self.hand_id].copy()
                
            self.readout_label.config(
                text=f"Target: [{self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}]\n"
                     f"Current EE: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]"
            )
            self.root.after(100, self._update_labels)

    def on_close(self):
        """Clean shutdown operations."""
        print("Shutting down physics-informed tester environment...")
        self.running = False
        self.viewer.close()
        self.root.destroy()
        
        # Clean up temporary scene XML
        test_xml = "/home/aidin/B/mujoco_project/panda_scripts/franka_emika_panda/scene_test.xml"
        if os.path.exists(test_xml):
            os.remove(test_xml)


if __name__ == "__main__":
    app = PhysicsInformedPandaTesterApp()