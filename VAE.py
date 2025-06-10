import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import gc

# OTHER
import minari
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib

# --- Hyperparameters ---
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LATENT_DIMS = 32
EPOCHS = 200
WH = 128          # Using a power of 2 for size makes conv layers simpler
LEARNING_RATE = 1e-3

def save_reconstruction(reconstructed_tensor, original_tensor, epoch):
    """Saves a comparison of the original and reconstructed image."""

    os.makedirs('./output_images/', exist_ok=True)

    # Convert tensors to numpy arrays for saving with PIL
    reconstructed_img = reconstructed_tensor[0].cpu().detach().permute(1, 2, 0).numpy()
    original_img = original_tensor[0].cpu().detach().permute(1, 2, 0).numpy()

    # Scale from [0, 1] to [0, 255] and convert to uint8
    reconstructed_img = (reconstructed_img * 255).astype(np.uint8)
    original_img = (original_img * 255).astype(np.uint8)

    # Handle single-channel (grayscale) images
    if reconstructed_img.shape[2] == 1:
        reconstructed_img = reconstructed_img.squeeze(2)
        original_img = original_img.squeeze(2)

    # Create a side-by-side comparison image
    comparison = np.hstack((original_img, reconstructed_img))

    # Save the image
    img = Image.fromarray(comparison)
    img.save(f'./output_images/reconstruction_epoch_{epoch}.png')


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, image_size=128):
        super(VariationalEncoder, self).__init__()
        self.image_size = image_size

        # Calculate number of downsampling layers needed
        # Each conv layer with stride=2 halves the spatial dimensions
        self.num_layers = int(torch.log2(torch.tensor(image_size // 4)).item())  # Down to 4x4

        # Build convolutional layers dynamically
        layers = []
        in_channels = 1
        out_channels = 32

        for i in range(self.num_layers):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            ])
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)  # Cap at 512 channels

        self.conv_layers = nn.Sequential(*layers)
        self.final_channels = in_channels

        # Calculate flattened dimension
        self.flattened_dim = self.final_channels * 4 * 4

        # Fully connected layers for latent space
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dims)
        self.fc_log_var = nn.Linear(self.flattened_dim, latent_dims)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)

        # Get mu and log_var
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dims, image_size=128):
        super(Decoder, self).__init__()
        self.latent_dims = latent_dims
        self.image_size = image_size

        # Calculate number of upsampling layers needed
        self.num_layers = int(torch.log2(torch.tensor(image_size // 4)).item())

        # Calculate starting channels (should match encoder's final channels)
        start_channels = 32
        for _ in range(self.num_layers - 1):
            start_channels = min(start_channels * 2, 512)

        self.start_channels = start_channels
        self.start_dim = self.start_channels * 4 * 4
        self.fc = nn.Linear(latent_dims, self.start_dim)

        # Build transposed convolutional layers dynamically
        layers = []
        in_channels = self.start_channels

        for i in range(self.num_layers - 1):
            out_channels = max(in_channels // 2, 32)
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            ])
            in_channels = out_channels

        # Final layer to output single channel
        layers.extend([
            nn.ConvTranspose2d(in_channels, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        ])

        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.start_channels, 4, 4)
        x = self.deconv_layers(x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, image_size=128):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, image_size)
        self.decoder = Decoder(latent_dims, image_size)

    def reparameterize(self, mu, log_var):
        """Reparameterization trick to allow backpropagation through a random node."""
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var

def loss_function(recon_x, x, mu, log_var):
    """
    VAE loss function.
    Combines Binary Cross Entropy for reconstruction and KL Divergence for regularization.
    """
    # Use BCE because our output is sigmoided, treating pixels as Bernoulli probabilities.
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL Divergence (regularization term, forces latent space to be close to a standard normal distribution)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + KLD

def train(autoencoder, data_loader, epochs=20):
    print("Starting training")
    opt = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    len_data = len(data_loader.dataset)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, _ in data_loader:
            x = x.to(device) # GPU
            opt.zero_grad()

            # Forward pass
            x_hat, mu, log_var = autoencoder(x)

            # Calculate loss
            loss = loss_function(x_hat, x, mu, log_var)

            # Backward pass and optimization
            loss.backward()
            opt.step()

            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)

            epoch_loss += loss.item()

        # Save a sample reconstruction every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_reconstruction(x_hat, x, epoch + 1)

        avg_loss = epoch_loss / len_data
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

    return autoencoder

def plotLatentSpace(autoencoder, data_loader, device='cuda'):
    """
    Visualize the 2D latent space of a VAE trained on Atari datasets.
    Color-codes points by game type to show how different games cluster.
    """
    
    mus_list = []
    game_labels_list = []
    
    with torch.no_grad():
        for x, game_labels in data_loader:
            x = x.to(device)
            
            x_hat, mu, log_var = autoencoder(x)
            
            # Store latent representations and game labels
            mus_list.append(mu.cpu())
            game_labels_list.append(game_labels.cpu())
    
    # Concatenate all batches
    mus = torch.cat(mus_list, dim=0).numpy()
    game_labels = torch.cat(game_labels_list, dim=0).numpy()
    
    # Handle case where game_labels might be multi-dimensional
    if game_labels.ndim > 1:
        if len(game_labels.shape) > 2:
            game_labels = game_labels.reshape(game_labels.shape[0], -1)  # Flatten to (batch, features)
        
        game_labels = game_labels.mean(axis=1)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot each game separately for better legend
    plt.scatter(mus[:, 0], mus[:, 1], 
                       c=game_labels, s=8, alpha=0.7)
    
    plt.title('Atari Games Latent Space Visualization', fontsize=16)
    plt.xlabel('Latent Dimension 1', fontsize=12)
    plt.ylabel('Latent Dimension 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

class ObservationDataset(Dataset):
    def __init__(self, observations, transform=None):
        self.observations = observations
        self.transform = transform

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = self.observations[idx]

        if self.transform:
            obs = self.transform(obs)

        return obs, obs

if __name__ == "__main__":

    training = False
    plotting = True
    testing = False

    # --- Define Transforms for Preprocessing ---
    # Convert numpy array -> PIL Image -> Grayscale -> Resize -> To Tensor (scales to [0,1])
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((WH, WH)),
        transforms.ToTensor()
    ])

    print("Loading datasets...")
    dataset1 = minari.load_dataset('atari/breakout/expert-v0', download=True)
    dataset1.set_seed(seed=123)
    dataset2 = minari.load_dataset('atari/centipede/expert-v0', download=True)
    dataset2.set_seed(seed=123)
    dataset3 = minari.load_dataset('atari/assault/expert-v0', download=True)
    dataset3.set_seed(seed=123)

    if training:
        # --- Collect all observations from multiple episodes ---
        all_observations = []
        for i,dataS in enumerate([dataset1, dataset2, dataset3]):
          if i == 2:
            episodes = dataS.sample_episodes(2) # Using 10 episodes for a decent dataset size
          else:
            episodes = dataS.sample_episodes(5) # Using 10 episodes for a decent dataset size
          for ep in episodes:
              all_observations.extend(ep.observations)
          print(f"Collected {len(all_observations)} total observations.")

        # --- Create Dataset and DataLoader ---
        training_dataset = ObservationDataset(all_observations, transform=data_transform)
        train_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        # --- Initialize and Train the Model ---
        model = VariationalAutoencoder(latent_dims=LATENT_DIMS).to(device)
        model = train(model, train_loader, epochs=EPOCHS)

        # --- Save the Models ---
        torch.save(model.state_dict(), 'vae_model_all.pth')
        print("Full VAE model state_dict saved to vae_model.pth")
        torch.save(model.encoder.state_dict(), 'encoder_model_all.pth')
        print("Encoder model state_dict saved to encoder_model.pth")

    # --- Example of Loading and Using the Model ---
    print("\n--- Loading model and generating a reconstruction ---")

    # Load a fresh model
    loaded_vae = VariationalAutoencoder(latent_dims=LATENT_DIMS).to(device)
    loaded_vae.load_state_dict(torch.load('vae_model_all.pth'))
    loaded_vae.eval()

    # Plotting
    if plotting:
        all_observations = []
        for i,dataS in enumerate([dataset1, dataset2, dataset3]):
          if i == 2:
            episodes = dataS.sample_episodes(1) # Using 10 episodes for a decent dataset size
          else:
            episodes = dataS.sample_episodes(2) # Using 10 episodes for a decent dataset size
          for ep in episodes:
              all_observations.extend(ep.observations)
          print(f"Collected {len(all_observations)} total observations.")

        # --- Create Dataset and DataLoader ---
        training_dataset = ObservationDataset(all_observations[:8000], transform=data_transform)
        train_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        plotLatentSpace(loaded_vae, train_loader)

    if testing:
        # Get a single observation from a different part of the dataset
        test_episode = dataset2[9]
        test_obs_np = test_episode.observations[100] # Pick an observation from the middle

        # Apply the same transformation
        test_obs_tensor = data_transform(test_obs_np).to(device)

        # Add a batch dimension (B, C, H, W)
        test_obs_tensor = test_obs_tensor.unsqueeze(0)

        # Generate reconstruction
        with torch.no_grad():
            reconstruction, _, _ = loaded_vae(test_obs_tensor)

        # Save the final test reconstruction
        save_reconstruction(reconstruction, test_obs_tensor, "final_test")
        print("Saved final test reconstruction to ./output_images/reconstruction_epoch_final_test.png")

