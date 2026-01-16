import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from train_v1 import Autoencoder,SpectrogramDataset
from math import log10

#Configuration
HOME = os.getcwd()
OUTPUT_DIR = f"{HOME}/autoencoder_outputs_v1"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load configuration from training output
CONFIG_PATH = f"{OUTPUT_DIR}/config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}. Run the training script first.")

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

LATENT_DIM = config['latent_dim']
ENCODER_CHANNELS = config['encoder_channels']
INPUT_SHAPE = tuple(config['input_shape'])
BATCH_SIZE = 16 

KERNEL_SIZE = 3
STRIDE = 2

# --- Evaluation Function ---
def evaluate_model(model, dataloader, device):
    model.eval()

    metrics = {
        "mse": [],
        "l1": [],
        "rmse": [],
        "relative_error": [],
        "psnr": [],
        "spectral_energy_error": [],
        "latent_mean": [],
        "latent_std": []
    }

    print("Starting evaluation...")
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for data, _ in pbar:
            data = data.to(device)

            reconstruction, latent = model(data)

            # --- Reconstruction metrics ---
            mse = torch.mean((reconstruction - data) ** 2)
            l1 = torch.mean(torch.abs(reconstruction - data))
            rmse = torch.sqrt(mse)

            rel_err = torch.norm(reconstruction - data) / torch.norm(data)

            # PSNR
            psnr = 10 * log10(1.0 / mse.item()) if mse.item() > 0 else 100.0

            # Spectral energy
            orig_energy = torch.mean(data ** 2)
            recon_energy = torch.mean(reconstruction ** 2)
            energy_err = torch.abs(orig_energy - recon_energy) / orig_energy

            # Latent statistics
            metrics["latent_mean"].append(latent.mean().item())
            metrics["latent_std"].append(latent.std().item())

            # Store metrics
            metrics["mse"].append(mse.item())
            metrics["l1"].append(l1.item())
            metrics["rmse"].append(rmse.item())
            metrics["relative_error"].append(rel_err.item())
            metrics["psnr"].append(psnr)
            metrics["spectral_energy_error"].append(energy_err.item())

    # Averageed metric
    final_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return final_metrics

def main_evaluate():
    """Main function for model evaluation"""
    print("=" * 50)
    print("AUTOENCODER EVALUATION ON TEST SET")
    print("=" * 50)
    
    # 1. Load the Best Model Checkpoint
    checkpoint_path = f"{OUTPUT_DIR}/best.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best model checkpoint not found at {checkpoint_path}. Ensure training was successful.")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE,weights_only=False)
    print(f"Loaded model from epoch {checkpoint['epoch']+1} with Val Loss: {checkpoint['val_loss']:.4f}")
    
    # 2. Instantiate Model and Load Weights
    model = Autoencoder(
        latent_dim=LATENT_DIM,
        output_shape=INPUT_SHAPE
    ).to(DEVICE)

    # Create a dummy tensor matching the expected input shape: (1, 1, H, W)
    dummy_input = torch.randn(1, 1, INPUT_SHAPE[0], INPUT_SHAPE[1]).to(DEVICE)
    with torch.no_grad():
        model(dummy_input)  # Run a forward pass to initialize the self.fc layer

    #load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Successfully loaded model state dictionary.")

    # 4. Load the Test Dataset and DataLoader
    test_dataset_path = f"{HOME}/logspectrograms/test" 

    test_dataset = SpectrogramDataset(
        data_root=test_dataset_path, 
        normalize=True,
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    # 5. Run Evaluation
    metrics = evaluate_model(model, test_loader, DEVICE)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (TEST SET)")
    print("=" * 60)

    print(f"MSE:                    {metrics['mse']:.6f}")
    print(f"L1 (MAE):               {metrics['l1']:.6f}")
    print(f"RMSE:                  {metrics['rmse']:.6f}")
    print(f"Relative Error:         {metrics['relative_error']:.4f}")
    print(f"PSNR (dB):              {metrics['psnr']:.2f}")
    print(f"Spectral Energy Error:  {metrics['spectral_energy_error']:.4f}")

    print("\nLATENT SPACE STATISTICS")
    print(f"   Latent Mean:            {metrics['latent_mean']:.4f}")
    print(f"   Latent Std:             {metrics['latent_std']:.4f}")

    print("=" * 60)

    #Visualize reconstruction on test samples
    visualize_reconstruction(
        model, 
        test_dataset, 
        DEVICE, 
        f"{OUTPUT_DIR}/test_reconstruction_final.png", 
        num_samples=8
    )
    print(f"Test reconstructions saved to {OUTPUT_DIR}/test_reconstruction_final.png")

def visualize_reconstruction(model, dataset, device, save_path, num_samples=4):
    """Visualize original vs reconstructed spectrograms (copied from training script)"""
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

# --- Run Evaluation ---
if __name__ == "__main__":
    main_evaluate()