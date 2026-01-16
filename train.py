import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json
import csv

HOME = os.getcwd()
OUTPUT_DIR = f"{HOME}/autoencoder_outputs_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_path = f"{OUTPUT_DIR}/training_log.csv"
csv_file = Path(csv_path)

if not csv_file.is_file():
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "learning_rate",
        ])

# Model hyperparameters
LATENT_DIM = 128
INPUT_CHANNELS = 1  # Grayscale spectrogram
ENCODER_CHANNELS = [32, 64, 128]
KERNEL_SIZE = 3
STRIDE = 2

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 3
VALIDATION_SPLIT = 0.15

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class SpectrogramDataset(Dataset):
    """Dataset loader for log-mel spectrograms"""
    
    def __init__(self, data_root, normalize=True):
        self.data_root = data_root
        self.normalize = normalize
        self.file_paths = []
        self.labels = []
        self.genre_to_idx = {}
        
        # Collect all .npy files
        genres = sorted([g for g in os.listdir(data_root) 
                        if os.path.isdir(os.path.join(data_root, g))])
        
        for idx, genre in enumerate(genres):
            self.genre_to_idx[genre] = idx
            genre_dir = os.path.join(data_root, genre)
            npy_files = [os.path.join(genre_dir, f) 
                        for f in os.listdir(genre_dir) if f.endswith('.npy')]
            self.file_paths.extend(npy_files)
            self.labels.extend([idx] * len(npy_files))
        
        print(f"Dataset: {len(self.file_paths)} spectrograms from {len(genres)} genres")
        
        # Compute global statistics for normalization
        if self.normalize:
            self._compute_stats()
    
    def _compute_stats(self, sample_size=500):
        """Compute mean and std from random sample for normalization"""
        print("🔍 Computing global statistics...")
        indices = np.random.choice(len(self.file_paths), 
                                   min(sample_size, len(self.file_paths)), 
                                   replace=False)
        
        samples = []
        for idx in tqdm(indices, desc="Sampling"):
            spec = np.load(self.file_paths[idx])
            samples.append(spec)
        
        samples = np.stack(samples)
        self.global_mean = float(np.mean(samples))
        self.global_std = float(np.std(samples))
        
        print(f"   Mean: {self.global_mean:.3f}, Std: {self.global_std:.3f}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load spectrogram
        spec = np.load(self.file_paths[idx]).astype(np.float32)
        
        # Normalize using global statistics
        if self.normalize:
            spec = (spec - self.global_mean) / (self.global_std + 1e-8)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        spec = spec[np.newaxis, ...]
        
        return torch.from_numpy(spec), self.labels[idx]


def crop_to_match(x, ref):
    _, _, h, w = ref.shape
    return x[:, :, :h, :w]

# -------------------------------
# Encoder
# -------------------------------
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        s1 = self.enc1(x)   # 128×130
        s2 = self.enc2(s1)  # 64×65
        z  = self.enc3(s2)  # 32×33
        return z, (s1, s2)

# -------------------------------
# Decoder
# -------------------------------
class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_shape=(128, 130)):
        super().__init__()

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        self.output_shape = output_shape

    def forward(self, z, skips):
        s1, s2 = skips

        x = self.dec3(z)
        x = crop_to_match(x, s2)
        x = x + s2                     # skip 2

        x = self.dec2(x)
        x = x + s1                     # skip 1

        x = self.dec1(x)

        # Ensure exact size
        return x[:, :, :self.output_shape[0], :self.output_shape[1]]

# -------------------------------
# Autoencoder
# -------------------------------
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128, output_shape=(128, 130)):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim, output_shape)

    def forward(self, x):
        z, skips = self.encoder(x)
        recon = self.decoder(z, skips)
        return recon, z

    def encode(self, x):
        z, _ = self.encoder(x)
        return z

    def decode(self, z, skips):
        return self.decoder(z, skips)

class ReconstructionLoss(nn.Module):
    def __init__(self, l1_weight=0.6, mse_weight=0.3, energy_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.energy_weight = energy_weight

    def forward(self, output, target):
        l1_loss = self.l1(output, target)
        mse_loss = self.mse(output, target)

        energy_loss = torch.abs(
            torch.mean(output ** 2) - torch.mean(target ** 2)
        )

        return (
            self.l1_weight * l1_loss +
            self.mse_weight * mse_loss +
            self.energy_weight * energy_loss
        )

# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        
        optimizer.zero_grad()
        reconstruction, latent = model(data)
        loss = criterion(reconstruction, data)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for data, _ in pbar:
            data = data.to(device)
            reconstruction, latent = model(data)
            loss = criterion(reconstruction, data)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def visualize_reconstruction(model, dataset, device, save_path, num_samples=4):
    """Visualize original vs reconstructed spectrograms"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2*num_samples))
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            data, _ = dataset[idx]
            data = data.unsqueeze(0).to(device)
            
            reconstruction, _ = model(data)
            
            original = data.cpu().squeeze().numpy()
            recon = reconstruction.cpu().squeeze().numpy()
            
            axes[i, 0].imshow(original, aspect='auto', origin='lower')
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(recon, aspect='auto', origin='lower')
            axes[i, 1].set_title('Reconstructed')
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def train():
    print("AUTOENCODER TRAINING FOR LOG-MEL SPECTROGRAMS")
    
    train_dataset_path = f"{HOME}/logspectrograms/train"
    val_dataset_path = f"{HOME}/logspectrograms/val"
    
    train_dataset = SpectrogramDataset(data_root=train_dataset_path,normalize=True)
    val_dataset = SpectrogramDataset(data_root=val_dataset_path,normalize=True)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=2, pin_memory=True)
    
    # Get sample to determine input shape
    sample, _ = train_dataset[0]
    input_shape = sample.shape[1:]  # Remove channel dimension
    print(f"Input shape: {input_shape}")
    
    # Initialize model
    model = Autoencoder(
        latent_dim=LATENT_DIM,
        output_shape=input_shape
    ).to(DEVICE)
    
    criterion = ReconstructionLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Save config
    config = {
        'latent_dim': LATENT_DIM,
        'encoder_channels': ENCODER_CHANNELS,
        'input_shape': input_shape,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
    }
    
    with open(f"{OUTPUT_DIR}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...\n")
    
    # Training loop
    pbar = tqdm(range(NUM_EPOCHS), desc="Training Progress")
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate_epoch(model, val_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Log metrics to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                float(train_loss),
                float(val_loss),
                optimizer.param_groups[0]['lr'],
            ])
        
        scheduler.step(val_loss)
        
        # Update progress bar
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}'
        })

        tqdm.write(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
                   f"Train Loss: {train_loss:.4f} | "
                   f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, f"{OUTPUT_DIR}/best.pt")
            tqdm.write(f"   ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Save last checkpoint
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, f"{OUTPUT_DIR}/last.pt")
        
        # Visualize every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualize_reconstruction(
                model, val_dataset, DEVICE,
                f"{OUTPUT_DIR}/reconstruction_epoch{epoch+1}.png"
            )
        
        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            tqdm.write(f"\nEarly stopping triggered (patience={EARLY_STOP_PATIENCE})")
            break
    
    # Final checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config
    }, f"{OUTPUT_DIR}/last.pt")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Checkpoints saved to: {OUTPUT_DIR}/")
    print("=" * 70)

if __name__ == "__main__":
    train()