import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Func

from stft_utils import STFTProcessor
from tfdense_net import TFDense_Net
from multiSD import MultiSpectrogramDiscriminator


class TFDense_GAN(nn.Module):
    def __init__(self, device, stft_cfgs=None):
        super().__init__()
        self.device = device
        self.generator = TFDense_Net().to(device)
        self.discriminator = MultiSpectrogramDiscriminator(device, stft_cfgs).to(device)
        self.stft_proc = STFTProcessor(device)

    def forward(self, noisy_audio):
        """
        Args:
            noisy_audio: [B, 1, T]
        Returns:
            denoised_audio: [B, 1, T]
        """
        # --- Time-Frequency domain ---
        mag, phase = self.stft_proc.compute_mag_phase(noisy_audio) # [B, F, T]
        mag_denoised = self.generator(mag.unsqueeze(1).permute(0,1,3,2))
        mag_denoised = mag_denoised.squeeze(1).permute(0,2,1) # back to [B, F, T]

        # inverse STFT to get waveform
        mag_denoised = torch.clamp(mag_denoised, min=1e-8, max=1e2) 
        audio_denoised = self.stft_proc.istft(mag_denoised, phase, length=16000)

        return audio_denoised, mag_denoised
