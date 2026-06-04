import os
import threading
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

# Scene XML incorporating the physical table and red cylinder obstacle
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
    
    <!-- Table base the robot is mounted on -->
    <geom name="table" type="box" size="0.4 0.4 0.02" pos="0 0 0.01" rgba="0.3 0.3 0.3 1"/>
    
    <!-- Red Cylinder Obstacle in the Workspace -->
    <geom name="obstacle" type="cylinder" size="0.08 0.15" pos="0.3 0.2 0.15" rgba="0.8 0.1 0.1 1"/>
  </worldbody>
</mujoco>
"""

def prepare_test_scene_xml(input_content=SCENE_XML_CONTENT, output_path="/home/aidin/B/mujoco_project/panda_scripts/franka_emika_panda/scene_test.xml"):
    # Inject target body tracker inside worldbody
    target_body_xml = """
    <body name="target" pos="0 0 0.5" mocap="true">
      <geom type="sphere" size="0.02" rgba="0 1 0 0.6" contype="0" conaffinity="0"/>
    </body>
    """
    xml_text = input_content.replace("<worldbody>", "<worldbody>" + target_body_xml)
        
    with open(output_path, "w") as f:
        f.write(xml_text)
    print(f"Created temporary testing scene XML at '{output_path}' with visual target and obstacles.")


class SafetyConsciousPandaTester:
    def __init__(self, model_path="panda_safety_conscious_model_finetuned"):
        # Load the policy
        print(f"Loading trained safety-conscious model from '{model_path}'...")
        self.model = PPO.load(model_path)
        
        # Prepare and load modified scene XML
        prepare_test_scene_xml()
        self.mj_model = mujoco.MjModel.from_xml_path("/home/aidin/B/mujoco_project/panda_scripts/franka_emika_panda/scene_test.xml")
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # Reset to home keyframe
        self.key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.key_id)
        
        # Body and joint registrations
        self.hand_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.link1_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "link1")
        self.target_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.mocap_id = self.mj_model.body_mocapid[self.target_body_id]
        
        self.joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # Physics & double integrator variables
        self.frame_skip = 10
        self.dt = self.frame_skip * self.mj_model.opt.timestep  # 0.02s
        self.max_joint_acc = 15.0  
        self.joint_vel_limits = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
        
        # Integrator Tracking states
        self.integrated_vel = np.zeros(7, dtype=np.float32)
        self.last_action = np.zeros(7, dtype=np.float32)
        
        # Initial Target State
        self.target_x = 0.3
        self.target_y = -0.1  # Initial position offset from cylinder obstacle [0.3, 0.2]
        self.target_z = 0.4
        
        self.running = True

        # Launch the viewer
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        
        # Start simulation loop thread
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()
        
        # Build interactive interface
        self._build_gui()

    def _simulation_loop(self):
        """Simulation and command tracking loop running in a background thread."""
        while self.running and self.viewer.is_running():
            step_start = time.time()
            
            tx, ty, tz = self.target_x, self.target_y, self.target_z
            target_pos = np.array([tx, ty, tz], dtype=np.float32)
            
            # Lock the viewer state while reading and updating simulation variables
            with self.viewer.lock():
                # 1. Update target coordinate system marker
                self.mj_data.mocap_pos[self.mocap_id] = target_pos
                
                # 2. Extract physics observations (matching size 45 structure)
                qpos = self.mj_data.qpos
                qvel = self.mj_data.qvel
                ee_pos = self.mj_data.xpos[self.hand_id]
                dist_vector = target_pos - ee_pos
                
                # Dynamic clearances for observation layout
                link1_pos = self.mj_data.xpos[self.link1_id]
                dist_ee_link1 = np.linalg.norm(ee_pos - link1_pos)
                
                dist_to_lower = qpos[0:7] - self.joint_limits_lower
                dist_to_upper = self.joint_limits_upper - qpos[0:7]
                limit_clearance = np.minimum(dist_to_lower, dist_to_upper)
                
                obs = np.concatenate([
                    qpos[0:7],                      # (7) Joint angles
                    qvel[0:7],                      # (7) Joint velocities
                    self.integrated_vel,            # (7) Tracked joint target velocities
                    ee_pos,                         # (3) Hand position
                    target_pos,                     # (3) Target coordinates
                    dist_vector,                    # (3) Distance target vector
                    self.last_action,               # (7) Action history
                    limit_clearance,                # (7) New Joint Limit clearances
                    np.array([dist_ee_link1], dtype=np.float32) # (1) Clearance to base shoulder
                ]).astype(np.float32)
                
                # 3. Model Inference (Predict joint acceleration)
                action, _ = self.model.predict(obs, deterministic=True)
                
                # 4. Integrate acceleration command to velocity: v_new = v_old + a * dt
                joint_acc = action * self.max_joint_acc
                new_joint_vel = self.integrated_vel + (joint_acc * self.dt)
                new_joint_vel = np.clip(new_joint_vel, -self.joint_vel_limits, self.joint_vel_limits)
                
                # 5. Position Integration: q_new = q_old + v_new * dt
                current_angles = qpos[0:7]
                target_angles = current_angles + (new_joint_vel * self.dt)
                target_angles = np.clip(target_angles, self.joint_limits_lower, self.joint_limits_upper)
                
                # Update tracking histories
                self.integrated_vel = new_joint_vel.copy()
                self.last_action = action.copy()
                
                # 6. Apply target joint angles
                self.mj_data.ctrl[0:7] = target_angles
                self.mj_data.ctrl[7] = 255.0  # Keep fingers closed
                
                # 7. Step physics frame_skip times
                for _ in range(self.frame_skip):
                    mujoco.mj_step(self.mj_model, self.mj_data)
                
            # Sync rendering
            self.viewer.sync()
            
            # Maintain 50Hz control loop speed (0.02s step size)
            elapsed = time.time() - step_start
            time.sleep(max(0.001, 0.02 - elapsed))

    def _build_gui(self):
        """Constructs a Tkinter slider window on the main thread."""
        self.root = tk.Tk()
        self.root.title("Safety-Conscious Panda Controller")
        self.root.geometry("400x350")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        style = ttk.Style()
        style.theme_use('clam')

        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Command Target Near Red Cylinder Obstacle", font=("Helvetica", 11, "bold")).pack(pady=10)

        # Coordinate Sliders
        self._create_slider(main_frame, "Target X (Forward/Backward)", -0.6, 0.6, self.target_x, "x")
        self._create_slider(main_frame, "Target Y (Left/Right)", -0.6, 0.6, self.target_y, "y")
        self._create_slider(main_frame, "Target Z (Up/Down)", 0.1, 0.8, self.target_z, "z")

        # Readout text labels
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
        if self.running and self.viewer.is_running():
            with self.viewer.lock():
                ee_pos = self.mj_data.xpos[self.hand_id].copy()
                
            self.readout_label.config(
                text=f"Target: [{self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}]\n"
                     f"Current EE: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]"
            )
            self.root.after(100, self._update_labels)

    def on_close(self):
        print("Shutting down safety-conscious tester...")
        self.running = False
        self.viewer.close()
        self.root.destroy()
        
        test_xml = "/home/aidin/B/mujoco_project/panda_scripts/franka_emika_panda/scene_test.xml"
        if os.path.exists(test_xml):
            os.remove(test_xml)


if __name__ == "__main__":
    app = SafetyConsciousPandaTester()