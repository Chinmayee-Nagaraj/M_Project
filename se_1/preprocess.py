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

"""

import os
import random
import numpy as np
from typing import Tuple, Optional

import torch
import torchaudio
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler


# Global constants and settings
TARGET_SR = 16000             # Target sampling rate (Hz) for all audio
TARGET_LEN = TARGET_SR * 2    # 2 seconds segment length (32000 samples)
CROP_LEN = TARGET_SR          # 1 second crop for training (16000 samples)

MAX_TRIES = 5                 # Number of retries to find an informative segment


def compute_entropy(wav: Tensor, num_bins: int = 64) -> float:
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


def is_informative(wav: Tensor, threshold: float = 3.0) -> bool:
    """
    Check if waveform has enough entropy (i.e., not silence).
    """
    return compute_entropy(wav) > threshold


class VCTK_DEMAND_Dataset(Dataset):
    """
    Custom PyTorch Dataset for VCTK + DEMAND dataset.
    Loads clean + noisy audio pairs and performs:
      - Resampling to 16 kHz
      - Random 2s segment extraction
      - Entropy check (skip silence)
      - Random 1s crop for training
    """

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir (str): Directory containing 'clean' and 'noisy' subfolders.
        """
        self.clean_dir = os.path.join(data_dir, 'clean')
        self.noisy_dir = os.path.join(data_dir, 'noisy')

        # Collect all filenames from clean directory
        self.clean_wav_names = [f for f in os.listdir(self.clean_dir) if f.endswith(".wav")]
        self.clean_wav_names = sorted(self.clean_wav_names)

    def __len__(self) -> int:
        return len(self.clean_wav_names)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        # Get corresponding clean + noisy file paths
        clean_file = os.path.join(self.clean_dir, self.clean_wav_names[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_names[idx])

        # Load waveforms
        clean_wave, clean_sr = torchaudio.load(clean_file, normalize=True, channels_first=True) 
        noisy_wave, sr = torchaudio.load(noisy_file, normalize=True, channels_first=True)

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
        # clean: Tensor of shape [batch_size, 1, 16000]  → clean 1s waveform
        # noisy: Tensor of shape [batch_size, 1, 16000]  → noisy 1s waveform
        # length: Original length of the file before padding/cropping
        return clean_wave, noisy_wave, length
      

def load_data(
            ds_dir: str,
            split: float = 0.9,
            batch_size: int = 64,
            num_workers: int = 2,
            pin_memory: bool = True,
            shuffle: bool = True
            ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load VCTK-DEMAND dataset with train and test splits.
    
    Args:
        ds_dir (str): Root dataset directory containing 'trainset' and 'testset' subfolders.
        split (float): Fraction of training data used for training (rest for validation).
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker threads for DataLoader.
        pin_memory (bool): Speed up transfer to GPU.
        shuffle (bool): Shuffle training dataset.

    Returns:
        train_loader, val_loader, test_loader (torch.utils.data.DataLoader)
    """
    train_dir = os.path.join(ds_dir, "trainset")
    test_dir = os.path.join(ds_dir, "testset")

    # Create dataset objects
    full_train_ds = VCTK_DEMAND_Dataset(train_dir)
   
    # Split train/val
    train_size: int = int(split * len(full_train_ds))
    val_size: int = len(full_train_ds) - train_size
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])

   
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
   
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory, 
        drop_last=False
    )

    if os.path.exists(test_dir):
        test_ds = VCTK_DEMAND_Dataset(test_dir)
        test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,  # keep test deterministic
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False
        )
    else:
        print(f"No 'testset/' directory found in {ds_dir}. Skipping test_loader.")
        test_loader = None


    return train_loader, val_loader, test_loader
    
    
def load_data_with_subset_sampler(
            ds_dir: str,
            split: float = 0.9,
            batch_size: int = 64,
            num_workers: int = 2,
            pin_memory: bool = True,
            subset_size=20000
            ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_dir = os.path.join(ds_dir, "trainset")

    # Load full training dataset
    full_train_ds = VCTK_DEMAND_Dataset(train_dir)
    
    # Random subset
    all_indices = np.arange(len(full_train_ds))
    np.random.shuffle(all_indices)
    subset_indices = all_indices[:subset_size]

    # Split into train/val
    train_size = int(split * subset_size)
    train_idx = subset_indices[:train_size]
    val_idx = subset_indices[train_size:]

    # Samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    # Create dataloaders
    train_loader = DataLoader(
        full_train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
   
    val_loader = DataLoader(
        full_train_ds, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory, 
        drop_last=False
    )

    return train_loader, val_loader
            
	




