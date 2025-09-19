"""
VCTK-DEMAND Audio Dataset Loader
================================

This file implements a PyTorch-compatible dataset class and helper functions 
to prepare and load paired clean/noisy speech audio from the VCTK-DEMAND dataset. 
It is designed for speech enhancement and denoising tasks.

Main Features:
--------------
1. **On-the-fly preprocessing**
   - Loads paired clean and noisy speech waveforms.
   - Ensures all audio is resampled to 16 kHz.
   - For each file, a random 2-second segment is extracted.
   - Entropy check is applied to skip silent or low-information regions.
   - If the file is shorter than 2 seconds, it is zero-padded.

2. **Training-ready samples**
   - From the selected 2-second segment, a random 1-second crop is taken.
   - This 1-second segment is returned for both clean and noisy audio.

3. **Flexible dataloaders**
   - The `load_data()` function provides train and test dataloaders directly.
   - Supports multi-threaded loading, GPU-optimized transfers, and shuffling.

Expected Dataset Structure:
---------------------------
The dataset directory should contain the following subfolders:

    dataset_root/
        ├── trainset/
        │     ├── clean/
        │     └── noisy/
        └── testset/
              ├── clean/
              └── noisy/

Output shapes:
--------------
- clean: Tensor of shape [batch_size, 1, 16000]  → clean 1s waveform
- noisy: Tensor of shape [batch_size, 1, 16000]  → noisy 1s waveform
- length: Original length of the file before padding/cropping

"""

import os
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Global constants and settings
# -----------------------------
TARGET_SR = 16000             # Target sampling rate (Hz) for all audio
TARGET_LEN = TARGET_SR * 2    # 2 seconds segment length (32000 samples)
CROP_LEN = TARGET_SR          # 1 second crop for training (16000 samples)

MAX_TRIES = 5                 # Number of retries to find an informative segment


# -----------------------------
# Utility functions
# -----------------------------
def compute_entropy(wav, num_bins=64):
    """
    Compute entropy of a waveform based on histogram of values.

    Args:
        wav (Tensor): Input audio waveform [1, time].
        num_bins (int): Number of bins for histogram.

    Returns:
        entropy (float): Information entropy of the waveform.
    """
    hist = torch.histc(wav, bins=num_bins, min=-1.0, max=1.0)
    p = hist / torch.sum(hist)   # normalize to probability distribution
    p = p[p > 0]                 # avoid log(0)
    return -torch.sum(p * torch.log2(p))


def is_informative(wav, threshold=3.0):
    """
    Check if waveform has enough entropy (i.e., not silence).
    """
    return compute_entropy(wav) > threshold


# -----------------------------
# Dataset definition
# -----------------------------
class VCTK_DEMAND_Dataset(Dataset):
    """
    Custom PyTorch Dataset for VCTK + DEMAND dataset.
    Loads clean + noisy audio pairs and performs:
      - Resampling to 16 kHz
      - Random 2s segment extraction
      - Entropy check (skip silence)
      - Random 1s crop for training
    """

    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Directory containing 'clean' and 'noisy' subfolders.
        """
        self.clean_dir = os.path.join(data_dir, 'clean')
        self.noisy_dir = os.path.join(data_dir, 'noisy')

        # Collect all filenames from clean directory
        self.clean_wav_names = [f for f in os.listdir(self.clean_dir) if f.endswith(".wav")]
        self.clean_wav_names = sorted(self.clean_wav_names)

    def __len__(self):
        return len(self.clean_wav_names)

    def __getitem__(self, idx):
        # Get corresponding clean + noisy file paths
        clean_file = os.path.join(self.clean_dir, self.clean_wav_names[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_names[idx])

        # Load waveforms
        clean_wave, clean_sr = torchaudio.load(clean_file) 
        noisy_wave, sr = torchaudio.load(noisy_file)

        assert clean_sr == sr, "Clean and noisy sample rates do not match!"

        # If not in target SR, resample both
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            clean_wave = resampler(clean_wave)
            noisy_wave = resampler(noisy_wave)
         
        # Both clean and noisy must have same length
        assert clean_wave.shape == noisy_wave.shape        
        length = clean_wave.shape[-1]
      
        if (length < TARGET_LEN):
            # If too short → pad once (no retries needed)
            pad_size = TARGET_LEN - length    
            clean_w = F.pad(clean_wave, (0, pad_size))
            noisy_w = F.pad(noisy_wave, (0, pad_size))
        
        else:
            # Retry loop only for random cuts
            for _ in range(MAX_TRIES):
                # randomly cut 2-second segment
                start2s = random.randint(0, length - TARGET_LEN)
                clean_w = clean_wave[:, start2s:start2s + TARGET_LEN]
                noisy_w = noisy_wave[:, start2s:start2s + TARGET_LEN]
              
                if is_informative(clean_w):
                    break


        # Final selected 2s segments
        clean_wave = clean_w
        noisy_wave = noisy_w
 
        # From 2s, randomly crop 1s segment for training
        start1s = random.randint(0, TARGET_LEN - CROP_LEN)
        clean_wave = clean_wave[:, start1s:start1s+ CROP_LEN]
        noisy_wave = noisy_wave[:, start1s:start1s+ CROP_LEN]

        # Return tuple: (clean segment, noisy segment, original length)
        return clean_wave, noisy_wave, length
      

# -----------------------------
# Dataloader helper function
# -----------------------------
def load_data(ds_dir, batch_size=4, num_workers=2, pin_memory=True, shuffle=True):
    """
    Load VCTK-DEMAND dataset with train and test splits.
    
    Args:
        ds_dir (str): Root dataset directory containing 'trainset' and 'testset' subfolders.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker threads for DataLoader.
        pin_memory (bool): Speed up transfer to GPU.
        shuffle (bool): Shuffle training dataset.

    Returns:
        train_loader, test_loader (torch.utils.data.DataLoader)
    """
    train_dir = os.path.join(ds_dir, "trainset")
    test_dir = os.path.join(ds_dir, "testset")

    # Create dataset objects
    train_ds = VCTK_DEMAND_Dataset(train_dir)
    test_ds = VCTK_DEMAND_Dataset(test_dir)

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,  # keep test deterministic
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, test_loader
