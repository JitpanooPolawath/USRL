import numpy as np
from tqdm import tqdm
import gymnasium as gym
import ale_py

# Initialize atari 
gym.register_envs(ale_py)

# PARAMTER
NUM_STEPS = 1000
NUM_EPS = 1000

# Save image numpy
def save_numpy(filename, list_of_nump):
    with open (f"{filename}.npy", "wb") as f:
        np.save(f, list_of_nump)

# Loop through 3 envs
for env_name in ['Breakout-v5','Assault-v5', 'Pong-v5']:
    print(f"============ Creating: {env_name} dataset =============")
    env = gym.make(f'ALE/{env_name}')
    list_of_nump = []
    for episode in tqdm(range(NUM_EPS)):
        obs, info = env.reset()
        for step in range(NUM_STEPS):            
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            list_of_nump.append(obs)
            if terminated or truncated:
                break
    nump_list_nump = np.array(list_of_nump)
    save_numpy(f"{env_name}", nump_list_nump)
    env.close()