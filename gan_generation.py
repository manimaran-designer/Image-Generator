import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Hyperparameters
BATCH_SIZE = 64
LR = 0.0002
Z_DIM = 100
EPOCHS = 50
IMAGE_DIM = 28 * 28
REAL_LABEL_SMOOTH = 0.9
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# 1. Data Loading
class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # First column is label, rest are pixels
        self.pixels = self.data.iloc[:, 1:].values.astype('float32')
        # Normalize to [-1, 1] for Tanh activation in Generator
        self.pixels = (self.pixels - 127.5) / 127.5
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.pixels[idx])

# 2. Model Architecture

class Generator(nn.Module):
    def __init__(self, z_dim, image_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, image_dim),
            # nn.LeakyReLU(0.2),
            # nn.Linear(1024, image_dim),
            nn.Tanh() # Outputs in range [-1, 1]
        )
        
    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid() # Probability output
        )
        
    def forward(self, x):
        return self.disc(x)

# 3. Training Setup
def train():
    # Paths
    csv_path = 'section 4/archive/fashion-mnist_test.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    # Data
    dataset = FashionMNISTDataset(csv_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Models
    generator = Generator(Z_DIM, IMAGE_DIM).to(DEVICE)
    discriminator = Discriminator(IMAGE_DIM).to(DEVICE)
    
    # Optimizers
    opt_gen = optim.Adam(generator.parameters(), lr=LR)
    opt_disc = optim.Adam(discriminator.parameters(), lr=LR)
    
    # Loss
    criterion = nn.BCELoss()
    
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        for batch_idx, real in enumerate(loader):
            real = real.to(DEVICE)
            batch_size = real.shape[0]
            
            ### Train Discriminator: max log(D(real)) + log(1 - D(G(z)))
            noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
            fake = generator(noise)
            
            disc_real = discriminator(real).view(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real) * REAL_LABEL_SMOOTH)
            
            disc_fake = discriminator(fake.detach()).view(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            
            discriminator.zero_grad()
            loss_disc.backward()
            opt_disc.step()
            
            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
            output = discriminator(fake).view(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            
            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f}")
        
    print("Training Completed.")
    
    # 4. Generate and Save Images
    save_images(generator)

def save_images(generator, num_images=16):
    generator.eval()
    noise = torch.randn(num_images, Z_DIM).to(DEVICE)
    with torch.no_grad():
        generated_images = generator(noise).cpu().numpy()
        
    # Rescale to [0, 1]
    generated_images = (generated_images * 0.5) + 0.5
    generated_images = generated_images.reshape(-1, 28, 28)
    
    # Plotting
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i], cmap='gray')
        ax.axis('off')
        
    output_path = 'section 4/generated_images_gan.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Generated images saved to {output_path}")

if __name__ == "__main__":
    train()
