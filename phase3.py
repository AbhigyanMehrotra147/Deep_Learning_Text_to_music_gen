import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# =====================
# 1. Load Latent Representations
# =====================
# Load latent representations from file
# Load latent representations from file
latent_representations = np.load('latent_representations.npy', allow_pickle=True)

# Check and fix data type
if latent_representations.dtype == np.object_:
    latent_representations = np.array(latent_representations, dtype=np.float32)

print(f"Latent representations shape: {latent_representations.shape}")
print(f"Shape: {latent_representations.shape}")
print(f"Dtype: {latent_representations.dtype}")

# Convert to PyTorch tensor
latent_representations = torch.tensor(latent_representations, dtype=torch.float32)

# =====================
# 2. Process CSV Files for Precomputed Text Embeddings
# =====================
class VideoDataset(Dataset):
    def __init__(self, csv_folder_path, latent_representations):
        """
        Custom dataset to handle latent representations and precomputed text embeddings.

        Parameters:
            csv_folder_path (str): Path to the folder containing CSV files.
            latent_representations (torch.Tensor): Latent representations for each video.
        """
        self.csv_folder_path = csv_folder_path
        self.latent_representations = latent_representations

        # List and sort the CSV files by creation time
        self.csv_files = sorted(os.listdir(csv_folder_path), key=lambda x: os.path.getctime(os.path.join(csv_folder_path, x)))
        print(len(self.csv_files)) 
        print(len(latent_representations))
        # Ensure the number of CSV files matches the latent representations
        assert len(self.csv_files) == len(latent_representations), "Mismatch between CSV files and latent representations"

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        latent = self.latent_representations[idx]

        # Load precomputed text embeddings from CSV
        csv_path = os.path.join(self.csv_folder_path, csv_file)
        text_embeddings = pd.read_csv(csv_path, header=None)

        # Ensure data is numeric and fill NaNs if necessary
        text_embeddings = text_embeddings.apply(pd.to_numeric, errors='coerce').fillna(0).values

        # Convert to PyTorch tensor and apply mean pooling
        text_embedding_tensor = torch.tensor(text_embeddings, dtype=torch.float32).mean(dim=0)

        return latent, text_embedding_tensor

csv_folder_path = './embeddings/'  # Path to folder containing 300 CSV files
dataset = VideoDataset(csv_folder_path, latent_representations)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# =====================
# 3. Define the Denoising Model
# =====================
class Denoiser(nn.Module):
    def __init__(self, latent_dim, text_embedding_dim):
        super(Denoiser, self).__init__()
        self.fc1 = nn.Linear(latent_dim + text_embedding_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, noisy_latent, text_embedding):
        x = torch.cat((noisy_latent, text_embedding), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# =====================
# 4. Add Noise Function
# =====================
def add_noise(latent, noise_factor=0.1):
    """
    Adds Gaussian noise to latent representations.
    """
    noise = noise_factor * torch.randn_like(latent)
    return latent + noise

# =====================
# 5. Train the Denoising Model
# =====================
latent_dim = latent_representations.shape[1]
text_embedding_dim = pd.read_csv(os.path.join(csv_folder_path, os.listdir(csv_folder_path)[0]), header=None).shape[1]  # Get embedding dimension from first CSV
learning_rate = 0.001
epochs = 10

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denoiser = Denoiser(latent_dim, text_embedding_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(denoiser.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    denoiser.train()
    total_loss = 0.0
    for latents, texts in dataloader:
        latents, texts = latents.to(device), texts.to(device)

        # Add noise to latent representations
        noisy_latents = add_noise(latents, noise_factor=0.1)

        # Forward pass
        denoised_latents = denoiser(noisy_latents, texts)
        loss = criterion(denoised_latents, latents)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# =====================
# 6. Save the Trained Model
# =====================
torch.save(denoiser.state_dict(), "denoiser_model.pth")
print("Denoiser model saved as 'denoiser_model.pth'.")

# =====================
# 7. Evaluate the Denoising Model
# =====================
# Switch to evaluation mode
denoiser.eval()

# Test with a single batch
test_latents, test_texts = next(iter(dataloader))
test_latents, test_texts = test_latents.to(device), test_texts.to(device)

# Add noise
noisy_latents = add_noise(test_latents, noise_factor=0.1)

# Denoise
with torch.no_grad():
    reconstructed_latents = denoiser(noisy_latents, test_texts)

# Print original, noisy, and reconstructed latents
print("\nOriginal Latent:", test_latents[0])
print("\nNoisy Latent:", noisy_latents[0])
print("\nReconstructed Latent:", reconstructed_latents[0])
print(len(test_latents))
print(len(noisy_latents))
print(len(reconstructed_latents))