"""
STFT and ISTFT Processing Utilities
-----------------------------------

This file defines the STFTProcessor class, which provides a simple 
interface for converting audio waveforms into their time-frequency 
representations (magnitude + phase) and reconstructing them back 
to the time domain.

Key Features:
-------------
- Forward Short-Time Fourier Transform (STFT):
    * Converts a noisy waveform into its complex spectrogram.
    * Separates and returns magnitude and phase components.
    * Magnitude is typically used as input to neural networks.
    * Phase can be stored for reconstruction.

- Inverse STFT (ISTFT):
    * Reconstructs time-domain waveform from magnitude + phase.
    * Useful for evaluating enhanced magnitude spectrograms.

Typical Shapes:
---------------
- Input waveform: [B, 1, T] or [B, T]
- Magnitude:      [B, F, T_frames]
- Phase:          [B, F, T_frames]

"""

import torch
import torchaudio

# -----------------------------
# Global constants 
# -----------------------------
N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 256

T_length = 16000

class STFTProcessor:
    def __init__(self, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH):
        """
        STFT/ISTFT processor for waveform batches.

        Args:
            n_fft (int): FFT size.
            win_length (int): Window size.
            hop_length (int): Hop length between frames.
        """
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        # Complex STFT transform
        self.stft_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=None         # keep complex output   
        )

        # Inverse STFT
        self.istft_transform = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length
        )

    def stft(self, clean_wave, noisy_wave):
        """
        Compute STFT for clean (complex) and noisy (mag + phase).

        Args:
            clean_wave (Tensor): [B, 1, T]
            noisy_wave (Tensor): [B, 1, T]

        Returns:
            clean_mag: Magnitude [B, F, T_frames]
            noisy_mag: Magnitude [B, F, T_frames]
            noisy_phase: Phase [B, F, T_frames]
        """
        clean_spec = self.stft_transform(clean_wave.squeeze(1))
        noisy_spec = self.stft_transform(noisy_wave.squeeze(1))

        clean_mag = torch.abs(clean_spec)
        noisy_mag = torch.abs(noisy_spec)
        noisy_phase = torch.angle(noisy_spec)  # phase in radians

        return clean_mag, noisy_mag, noisy_phase

    def istft(self, noisy_mag, noisy_phase, length=T_length):
        """
        Reconstruct waveforms using ISTFT.

        Args:
            noisy_mag (Tensor): [B, F, T_frames] magnitude of noisy
            noisy_phase (Tensor): [B, F, T_frames] phase of noisy
            length (int, optional): Target waveform length

        Returns:
            rec_noisy: [B, 1, T] reconstructed noisy waveform
        """
        # Reconstruct noisy complex STFT
        noisy_spec = noisy_mag * torch.exp(1j * noisy_phase)
       
        rec_noisy = self.istft_transform(noisy_spec, length=length).unsqueeze(1)

        return rec_noisy
