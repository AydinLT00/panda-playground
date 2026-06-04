import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

# --- Import your custom Safety environment class from your main training script ---
# (Ensure your training script is saved as 'train_safety.py' in the same folder)
from test4 import SafetyConsciousPandaVecEnv

if __name__ == "__main__":
    log_dir = "./tb_logs/"
    os.makedirs(log_dir, exist_ok=True)

    num_parallel_envs = 2048
    print(f"Initializing {num_parallel_envs} parallel environments for fine-tuning...")
    
    # Instantiate the parallel environments
    raw_env = SafetyConsciousPandaVecEnv(
        num_envs=num_parallel_envs, 
        xml_path="/home/aidin/B/mujoco_project/panda_scripts/franka_emika_panda/scene.xml"
    )
    env = VecMonitor(raw_env)

    # Path to load the previously saved trained model
    base_model_path = "panda_safety_conscious_model.zip"
    
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"Base model '{base_model_path}' not found. Make sure the initial training has "
            "completed and saved the model file to this directory first."
        )

    print(f"Loading base model '{base_model_path}' for fine-tuning...")
    
    # Load the PPO agent and bind it to the new vector environment instance.
    # To perform gentle fine-tuning, we lower the learning rate using the 'custom_objects' parameter.
    model = PPO.load(
        base_model_path,
        env=env,
        custom_objects={
            "learning_rate": 3e-4,   # Slightly lowered from 3e-4 to allow gradual parameter updates
            "tensorboard_log": log_dir
        }
    )

    # Set up a checkpoint callback specifically for the fine-tuning phase
    checkpoint_callback = CheckpointCallback(
        save_freq=150_000, 
        save_path="./models_panda_finetuned/",
        name_prefix="safety_conscious_panda_finetune"
    )

    # Run fine-tuning for an additional 10 million steps
    additional_steps = 20_000_000
    print(f"Beginning fine-tuning process for an additional {additional_steps} steps...")
    
    model.learn(
        total_timesteps=additional_steps,
        callback=checkpoint_callback,
        tb_log_name="panda_warp_safety_finetuning",
        reset_num_timesteps=False  # Crucial! Keeps the Tensorboard graph timelines contiguous
    )

    # Save the updated model
    output_model_name = "panda_safety_conscious_model_finetuned"
    model.save(output_model_name)
    print(f"Fine-tuning complete. Updated model saved as '{output_model_name}'.")