import torch
import torchaudio
from torch.utils.data import DataLoader
from _dataloading import VCTK_DEMAND_Dataset


# ------------------------------
# Dataset & DataLoader
# ------------------------------
trainset_dir = "../../dataset/VCTK_DEMAND/trainset"
train_ds = VCTK_DEMAND_Dataset(trainset_dir)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)


# ------------------------------
# Example batch
# ------------------------------
for clean, noisy, length in train_loader:
    # clean/noisy: (B, 1, T) → squeeze → (B, T)
    clean = torch.squeeze(clean, 1)
    noisy = torch.squeeze(noisy, 1)

    # Batch STFT
    stft_clean = torch.stft(
        clean, n_fft=512, hop_length=256, win_length=512, return_complex=True
    )   # (B, F, frames)

    stft_noisy = torch.stft(
        noisy, n_fft=512, hop_length=256, win_length=512, return_complex=True
    )   # (B, F, frames)

    print("STFT clean shape:", stft_clean.shape)   # (B, F, frames)
    print("STFT noisy shape:", stft_noisy.shape)   # (B, F, frames)

    # Batch ISTFT
    reconstructed = torch.istft(
        stft_clean, n_fft=512, hop_length=256, win_length=512, length=clean.shape[-1]
    )   # (B, T)

    print("Reconstructed shape:", reconstructed.shape)

    # Reconstruction error (per sample in batch)
    mse = torch.mean((clean - reconstructed) ** 2, dim=1)
    print("Batch MSE:", mse.tolist())
    break
