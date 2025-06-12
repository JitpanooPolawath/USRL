import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import gc
from tqdm import tqdm

# Model
from VAE import VariationalEncoder

# gymnasium
import gymnasium as gym
import minari

# --- Hyperparameters ---
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LATENT_DIMS = 32
EPOCHS = 200
WH = 128          # Using a power of 2 for size makes conv layers simpler
LEARNING_RATE = 1e-3 
N_ACTIONS = 18 # The biggest action
SAMP_EPS = 5 # amount to train per env

# --- Neural Network agent ---
# Hypernetworks 
class Hypernetworks(nn.Module):
    def __init__(self, task_emb, network_shape):
        super(Hypernetworks, self).__init__()
        self.task_emb = task_emb
        self.network_shape = network_shape
        self.layers = nn.Sequential(
            nn.Linear(self.task_emb, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            # Output layer produces the flattened vector of all weights and biases
            nn.Linear(512, self.network_shape)
        )

    def forward(self, task_emb):
        return self.layers(task_emb)

# Neural networks
class NeuralNetwork(nn.Module):
    def __init__(self, latent_dims, n_actions):
        super(NeuralNetwork, self).__init__()
        self.l1_shape = (128, latent_dims) # out_features, in_features (e.g., latent_dims=32)
        self.b1_shape = (128,)
        self.l2_shape = (256, 128)
        self.b2_shape = (256,)
        self.l3_shape = (n_actions, 256)   # out_features, in_features (e.g., n_actions=4)
        self.b3_shape = (n_actions,)
        

    def get_total_params_count(self):
        return (
            self.l1_shape[0] * self.l1_shape[1] + self.b1_shape[0] +
            self.l2_shape[0] * self.l2_shape[1] + self.b2_shape[0] +
            self.l3_shape[0] * self.l3_shape[1] + self.b3_shape[0]
        )

    def forward(self, x, hypernet):
        batch_size = x.shape[0]
        
        if hypernet.dim() == 1:
            hypernet = hypernet.unsqueeze(0).expand(batch_size, -1)

        # Track where the hypernetwork is located at
        current_pos = 0

         # Layer 1 parameters
        l1_size = self.l1_shape[0] * self.l1_shape[1]
        w1_batch = hypernet[:, current_pos:current_pos + l1_size].view(batch_size, *self.l1_shape)
        current_pos += l1_size
        b1_batch = hypernet[:, current_pos:current_pos + self.b1_shape[0]]
        current_pos += self.b1_shape[0]
        
        # Layer 2 parameters
        l2_size = self.l2_shape[0] * self.l2_shape[1]
        w2_batch = hypernet[:, current_pos:current_pos + l2_size].view(batch_size, *self.l2_shape)
        current_pos += l2_size
        b2_batch = hypernet[:, current_pos:current_pos + self.b2_shape[0]]
        current_pos += self.b2_shape[0]
        
        # Layer 3 parameters
        l3_size = self.l3_shape[0] * self.l3_shape[1]
        w3_batch = hypernet[:, current_pos:current_pos + l3_size].view(batch_size, *self.l3_shape)
        current_pos += l3_size
        b3_batch = hypernet[:, current_pos:current_pos + self.b3_shape[0]]

        # Perform the forward pass manually
        outputs = []
        for i in range(batch_size):
            # Forward pass for sample i
            out = F.relu(F.linear(x[i:i+1], w1_batch[i], b1_batch[i]))
            out = F.relu(F.linear(out, w2_batch[i], b2_batch[i]))
            out = F.linear(out, w3_batch[i], b3_batch[i])
            outputs.append(out)
            
        return torch.cat(outputs, dim=0)

# Agent
class agent(nn.Module):
    def __init__(self, latent_dims, n_actions,total_task_emb):
        super(agent,self).__init__()
        self.latent_dims = latent_dims
        self.n_actions = n_actions
        self.total_task_emb = total_task_emb
        self.neural_net = NeuralNetwork(self.latent_dims,self.n_actions).to(device)
        self.neural_parameters = self.neural_net.get_total_params_count()
        self.hypernet = Hypernetworks(total_task_emb,self.neural_parameters).to(device)

    def forward(self,x,task_emb):
        # Updated neural net
        gen_param = self.hypernet(task_emb)
        # A set of probabilities
        return self.neural_net(x, gen_param)

if __name__ == "__main__":
    
    # Transforms Atari states to encoded
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((WH, WH)),
        transforms.ToTensor()
    ])

    # Expert datasets
    print("Loading datasets...")
    datasets_name = ['breakout','centipede','assault']
    dataset1 = minari.load_dataset('atari/breakout/expert-v0', download=True)
    dataset1.set_seed(seed=10)
    dataset2 = minari.load_dataset('atari/centipede/expert-v0', download=True)
    dataset2.set_seed(seed=13)
    dataset3 = minari.load_dataset('atari/assault/expert-v0', download=True)
    dataset3.set_seed(seed=12)
    datasets = [dataset1,dataset2,dataset3]

    # Load a encoder model
    loaded_E = VariationalEncoder(latent_dims=LATENT_DIMS).to(device)
    loaded_E.load_state_dict(torch.load('./models/encoder_model_all.pth'))
    loaded_E.eval()

    # Initialize agent
    neural_model = agent(LATENT_DIMS,N_ACTIONS,len(datasets)).to(device)
    optimizer = torch.optim.Adam(neural_model.hypernet.parameters(), lr=LEARNING_RATE)

    # Statistic
    log_interval = 1000
    loss_values = []
    running_loss = 0.0
    step_count = 0

    print("Starting training")
    # Training on datasets
    for task_id, dataset in enumerate(datasets):
        # task_id encoding
        task_id = torch.tensor([task_id], device=device)
        task_embedding = F.one_hot(task_id.detach().clone(), num_classes=len(datasets)).float().to(device)
        # List of episodeData
        sample_data = dataset.sample_episodes(SAMP_EPS)
        n_action = dataset.spec.action_space.n
        # Loop through the episodeData:
        for episode_data in sample_data:
            for step in tqdm(range(len(episode_data.observations)-1), desc=f"Training {datasets_name[task_id]}: "):
                # print(episode_data.observations[step], episode_data.actions[step], episode_data.rewards[step])
                enc_obs = data_transform(episode_data.observations[step]).unsqueeze(0).to(device)
                with torch.no_grad():
                    enc_obs, _ = loaded_E(enc_obs)
                
                # Agent action
                action_value = neural_model(enc_obs, task_embedding)
                expect_action = torch.tensor([episode_data.actions[step]], dtype=torch.long).to(device)
                # loss
                loss = F.cross_entropy(action_value, expect_action)
                current_loss = loss.item()
                loss_values.append(current_loss)
                running_loss += current_loss
                step_count += 1

                if step_count % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    print(f"\nAverage loss over last {log_interval} steps: {avg_loss:.6f}")
                    running_loss = 0.0  # Reset for next interval

                # backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(neural_model.hypernet.parameters(), 100)
                optimizer.step()

                # Clear cache
                if step_count % 50 == 0:
                    torch.cuda.empty_cache()  # Clear GPU cache
                    gc.collect()  # Clear CPU cache

            print()

        torch.save(neural_model.state_dict(), f'agent_hyperneural_{datasets_name[task_id]}.pth')
        print(f"{datasets_name[task_id]} saved to agent_hyperneural_{datasets_name[task_id]}.pth")

    print("===== Training completed =====")

    torch.save(neural_model.state_dict(), 'agent_hyperneural_comp.pth')
    print("Fully trained agent is saved to agent_hyperneural_comp.pth")

    if len(loss_values) > 0:
        overall_avg = sum(loss_values) / len(loss_values)
        print(f"Overall average loss: {overall_avg:.6f}")
        print(f"Min loss: {min(loss_values):.6f}")
        print(f"Max loss: {max(loss_values):.6f}")