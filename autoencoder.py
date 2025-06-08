import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import gc
import os
import numpy as np
from PIL import Image
import numpy as np

# --- Step 1: Define Hyperparameters and Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

LATENT_DIM = 64
LEARNING_RATE = 4e-4
BATCH_SIZE = 64
EPOCHS = 100 # Adjust as needed
IMAGE_HEIGHT = 224 # New resized height
IMAGE_WIDTH = 160  # New resized width

# --- Step 2: Prepare Data and Transforms ---
# The transform now includes a Resize step
transform = transforms.Compose([
    transforms.ToPILImage(), # Necessary if your input is a numpy array
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(), # Scales to [0, 1] and changes to (C, H, W)
])


def saveImage(tensor, original):

    # Convert torch tensor to numpy
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 4:  # (batch, channels, height, width)
            array = tensor.squeeze(0).cpu().numpy()
        else:
            array = tensor.cpu().numpy()
        
        # If channels first (C, H, W), convert to channels last (H, W, C)
        if array.shape[0] == 3:  # RGB channels
            array = np.transpose(array, (1, 2, 0))  # (H, W, C)
    else:
        array = tensor
    
    # Make sure values are in the correct range (0-255) and data type
    for i, arr in enumerate([array, original]):
        if arr.dtype != np.uint8:
            # If values are in range 0-1, scale to 0-255
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        image = Image.fromarray(arr)
        os.makedirs('./outputimage/',exist_ok=True)
        if i == 0:
            image.save(f'./outputimage/testoutput.png')
        else:
            image.save(f'./outputimage/exampleoutput.png')


class MemoryEfficientDataset(Dataset):
    def __init__(self, filepaths, transform=None, train_split=0.9):
        self.filepaths = filepaths
        self.transform = transform
        
        # Use memory mapping instead of loading into RAM
        self.mmaps = []
        self.file_lengths = []
        self.cumulative_lengths = [0]
        
        for filepath in filepaths:
            # Memory map the file (doesn't load into RAM)
            mmap_data = np.load(filepath, mmap_mode='r')  # Read-only memory map
            
            # Calculate train split length
            total_length = len(mmap_data)
            train_length = int(total_length * train_split)
            
            self.mmaps.append(mmap_data)
            self.file_lengths.append(train_length)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + train_length)
        
        self.total_length = self.cumulative_lengths[-1]
        print(f"Total dataset size: {self.total_length} (memory mapped)")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which file this index belongs to
        file_idx = 0
        for i, cum_length in enumerate(self.cumulative_lengths[1:]):
            if idx < cum_length:
                file_idx = i
                break
        
        # Calculate local index within the file
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        # Access data from memory-mapped file (only loads this specific item)
        image = self.mmaps[file_idx][local_idx].copy()
        
        if self.transform:
            image = self.transform(image)
        return image, 0

# --- Step 3: Model Architectures (Refactored) ---

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            # Input: (N, 3, 224, 160)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # -> (N, 32, 112, 80)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (N, 64, 56, 40)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (N, 128, 28, 20)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# -> (N, 256, 14, 10)
            nn.ReLU(),
            nn.Flatten(),                                          # -> (N, 256 * 14 * 10) = (N, 35840)
            nn.Linear(256 * 14 * 10, latent_dim)
        )
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256 * 14 * 10),
            nn.ReLU(),
            nn.Unflatten(1, (256, 14, 10)),                          # -> (N, 256, 14, 10)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (N, 128, 28, 20)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (N, 64, 56, 40)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (N, 32, 112, 80)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # -> (N, 3, 224, 160)
            nn.Sigmoid() # Scale output to [0, 1]
        )
    def forward(self, x):
        return self.model(x)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

if __name__ == "__main__":
    training = True

    if training:
        # Create Datasets and DataLoaders
        filepaths = ['Breakout-v5.npy']
        train_dataset = MemoryEfficientDataset(filepaths, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # --- Step 4: Instantiate, Train, and Save ---
        model = Autoencoder(LATENT_DIM).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print("--- Start training ---")
        # Training Loop
        for epoch in range(EPOCHS):
            model.train()
            for batch_idx, (imgs, _) in enumerate(train_loader):
                imgs = imgs.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, imgs)
                loss.backward()
                optimizer.step()

                # Clear cache periodically
                if batch_idx % 50 == 0:  # Every 50 batches
                    torch.cuda.empty_cache()  # Clear GPU cache
                    gc.collect()  # Clear CPU cache

            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item()}")

        print("\n--- Training Finished ---")


        # --- SAVE THE AUTOENCODER ---
        torch.save(model.state_dict(), 'autoencoder.pth')
        print("Model state_dict saved to autoencoder.pth")

        # --- SAVE THE ENCODER ---
        # Access the encoder part and save its state dictionary
        torch.save(model.encoder.state_dict(), 'encoder_model.pth')
        print("Encoder model state_dict saved to encoder_model.pth")

    print("\n--- Loading and using the saved autoencoder ---")
    loaded_autoencoder = Autoencoder(latent_dim=LATENT_DIM).to(device)
    loaded_autoencoder.load_state_dict(torch.load('autoencoder.pth'))

    # Test saved autoencoder trained correctly
    loaded_autoencoder.eval()
    example_data = np.load('Breakout-v5.npy')
    example_data = example_data[17980]
    example_tensor = torch.from_numpy(example_data).float() # Convert to float32
    example_tensor = example_tensor.unsqueeze(0).to(device)   # Add batch: (1, 210, 160, 3)
    example_tensor = example_tensor.permute(0, 3, 1, 2)  # Reorder to: (1, 3, 210, 160)
    example_tensor = example_tensor / 255.0  # Normalize to [0, 1]
    with torch.no_grad():
        output = loaded_autoencoder(example_tensor)
    print("Saving image")
    saveImage(output, example_data)

    # --- Step 5: Load and Use the Saved Encoder ---
    print("\n--- Loading and using the saved encoder ---")

    loaded_encoder = Encoder(latent_dim=LATENT_DIM).to(device)
    loaded_encoder.load_state_dict(torch.load('encoder_model.pth'))
    loaded_encoder.eval()
    example_data = np.load('Breakout-v5.npy')
    example_data = example_data[17980]
    example_tensor = torch.from_numpy(example_data).float() # Convert to float32
    example_tensor = example_tensor.unsqueeze(0).to(device)   # Add batch: (1, 210, 160, 3)
    example_tensor = example_tensor.permute(0, 3, 1, 2)  # Reorder to: (1, 3, 210, 160)
    example_tensor = example_tensor / 255.0  # Normalize to [0, 1]
    print(f"Input tensor shape for encoder: {example_tensor.shape}")

    # Get the latent vector
    with torch.no_grad():
        latent_vector = loaded_encoder(example_tensor)

    print(f"Shape of the resulting latent vector: {latent_vector.shape}") # Should be (1, LATENT_DIM)
    print("Example latent vector:")
    print(latent_vector)