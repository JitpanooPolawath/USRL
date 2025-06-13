# gymnasium and helper
import gymnasium as gym
import ale_py

# pytorch
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# Model
from VAE import VariationalEncoder
from HRL import agent

device = "cuda" if torch.cuda.is_available() else "cpu"
num_steps = 1000
EPS = 10
LATENT_DIMS = 64
N_ACTIONS = 4
WH = 256
NUM_CLASS = 1

# Transforms Atari states to encoded
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((WH, WH)),
    transforms.ToTensor()
])

if __name__ == "__main__":
    name = "Breakout-v5"
    env = gym.make(f"ALE/{name}", render_mode="human")
    
    # Load agent
    neural_model = agent(LATENT_DIMS,N_ACTIONS,1).to(device)
    neural_model.load_state_dict(torch.load('models/agent_hyperneural_comp.pth'))
    neural_model.eval()

    # Load a encoder model
    loaded_E = VariationalEncoder(latent_dims=LATENT_DIMS).to(device)
    loaded_E.load_state_dict(torch.load('./models/VAE/breakout_encoder_model_all.pth'))
    loaded_E.eval()

    # Breakout = 0, centipede = 1, assault = 2 
    task_id = torch.tensor([0], device=device)
    task_embedding = F.one_hot(task_id.detach().clone(), num_classes=NUM_CLASS).float().to(device)

    for ep in range(EPS):
        obs, info = env.reset()
        for step in range(num_steps):
            if name == "Breakout-v5":
                if step % 10 != 0:
                    enc_obs = data_transform(obs).unsqueeze(0).to(device)
                    with torch.no_grad():
                        enc_obs, _ = loaded_E(enc_obs)
                    action_value = neural_model(enc_obs, task_embedding)
                    action = torch.argmax(action_value).item()
                else:
                    action = 2
                obs, rewards, termination, truncation, info = env.step(action-1)
                done = termination or truncation

                if done:
                    break
            else:
                enc_obs = data_transform(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    enc_obs, _ = loaded_E(enc_obs)
                action_value = neural_model(enc_obs, task_embedding)
                action = torch.argmax(action_value).item()
                obs, rewards, termination, truncation, info = env.step(action-1)
                done = termination
                
                if done:
                    break

    env.close()
