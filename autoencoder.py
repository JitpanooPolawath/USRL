import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import numpy as np

# --- Step 1: Define Hyperparameters and Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

LATENT_DIM = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 20 # Adjust as needed
IMAGE_HEIGHT = 224 # New resized height
IMAGE_WIDTH = 160  # New resized width

# --- Step 2: Prepare Data and Transforms ---
# The transform now includes a Resize step
transform = transforms.Compose([
    transforms.ToPILImage(), # Necessary if your input is a numpy array
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(), # Scales to [0, 1] and changes to (C, H, W)
])


def saveImage(array):
    
    # Assuming your array is called 'array'
    # Make sure values are in the correct range (0-255) and data type
    if array.dtype != np.uint8:
        # If values are in range 0-1, scale to 0-255
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)

    # Convert to PIL Image
    image = Image.fromarray(array)

    # Save or display
    image.save('output.png')


# Create a dummy dataset for demonstration
# In your project, replace this with your actual dataset loader
class MyCustomDataset(Dataset):
    def __init__(self, filepath, transform=None):
        # Generating random data in (H, W, C) format
        self.data = np.load(filepath)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        # We don't have labels, but DataLoader expects a tuple
        return image, 0

# Create Datasets and DataLoaders
train_dataset = MyCustomDataset('Breakout-v5.npy', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Step 3: Define the Model Architectures (Refactored) ---

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

    # --- Step 4: Instantiate, Train, and Save ---
    model = Autoencoder(LATENT_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

    print("\n--- Training Finished ---")

    # --- SAVE THE ENCODER ---
    # Access the encoder part and save its state dictionary
    torch.save(model.encoder.state_dict(), 'encoder_model.pth')
    print("Encoder model state_dict saved to encoder_model.pth")

    # --- Step 5: Load and Use the Saved Encoder ---
    print("\n--- Loading and using the saved encoder ---")

    # 1. Instantiate a new encoder object
    loaded_encoder = Encoder(latent_dim=LATENT_DIM).to(device)

    # 2. Load the saved weights
    loaded_encoder.load_state_dict(torch.load('encoder_model.pth'))

    # 3. Set to evaluation mode
    loaded_encoder.eval()

    # Create a single dummy image with the original size (210, 160, 3)
    single_image_np = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)

    # Preprocess it exactly as you did for training
    input_tensor = transform(single_image_np).unsqueeze(0).to(device) # Add batch dimension (1, C, H, W)
    print(f"Input tensor shape for encoder: {input_tensor.shape}")

    # Get the latent vector
    with torch.no_grad():
        latent_vector = loaded_encoder(input_tensor)

    print(f"Shape of the resulting latent vector: {latent_vector.shape}") # Should be (1, LATENT_DIM)
    print("Example latent vector:")
    print(latent_vector)