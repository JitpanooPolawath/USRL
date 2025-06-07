import numpy as np
from tqdm import tqdm
import gymnasium as gym
import ale_py
import os
import gc
from multiprocessing import Pool, TimeoutError


# PARAMTER
NUM_STEPS = 1000
NUM_EPS = 100

# Save image numpy
def save_numpy(filename, list_of_nump):
    with open (f"{filename}.npy", "wb") as f:
        np.save(f, list_of_nump)


def env_data(env_name):
    print(f"============ Creating: {env_name} dataset =============")
    env = gym.make(f'ALE/{env_name}')    
    obs, info = env.reset()
    obs_shape = obs.shape
    max_observations = NUM_EPS * NUM_STEPS
    temp_filename = f"{env_name}_temp.npy"
    
    memmap_array = np.memmap(temp_filename, dtype=obs.dtype, mode='w+', 
                            shape=(max_observations,) + obs_shape)
    
    observation_count = 0
    
    for episode in tqdm(range(NUM_EPS)):
        obs, info = env.reset()
        
        for step in range(NUM_STEPS):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            memmap_array[observation_count] = obs
            observation_count += 1
            
            if terminated or truncated:
                break
    
    final_array = memmap_array[:observation_count]
    np.save(f"{env_name}.npy", final_array)
    
    del final_array
    del memmap_array
    os.remove(temp_filename)
    env.close()



if __name__=="__main__":

    # Loop through 3 envs
    with Pool(processes=3) as pool:
        pool.map(env_data,['Breakout-v5', 'Assault-v5', 'Pong-v5'])

        # Clean up temp files after all processing is done
        print("Cleaning up temporary files...")
        gc.collect()