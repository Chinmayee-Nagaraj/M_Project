import torch
import torch.nn as nn
import torch.nn.functional as Func
import torchaudio

from stft_utils import STFTProcessor


class FrequencyDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        # progressively downsample the spectrogram
        self.layers = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)),  # 1/2
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.parametrizations.weight_norm(nn.Conv2d(32, 64, 3, stride=2, padding=1)),  # 1/4
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.parametrizations.weight_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)),  # 1/8
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.parametrizations.weight_norm(nn.Conv2d(128, 256, 3, stride=2, padding=1)),  # 1/16
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.parametrizations.weight_norm(nn.Conv2d(256, 1, 3, stride=1, padding=1)),  # score map
        )

    def forward(self, x):
        return self.layers(x)


class MultiSpectrogramDiscriminator(nn.Module):
    def __init__(self, device, stft_configs=None):
        super().__init__()

        # Default to 3 different STFT settings
        if stft_configs is None:
            stft_configs = [
                {'n_fft': 1024, 'hop_length': 256, 'win_length': 1024},
                {'n_fft': 2048, 'hop_length': 512, 'win_length': 2048},
                {'n_fft': 512,  'hop_length': 128, 'win_length': 512},
            ]

        self.stft_configs = stft_configs
        
        self.stft_processors = [ STFTProcessor(device, **cfg) for cfg in stft_configs ]
        self.discriminators = nn.ModuleList([
            FrequencyDiscriminator(in_channels=1) for _ in stft_configs
        ])

    def forward(self, audio):
        """
        Args:
            audio: [B, 1, T]
        Returns:
            List of discriminator outputs (one per STFT) 
        """
        outputs = []
        for stft_proc, disc in zip(self.stft_processors, self.discriminators):
            mag = stft_proc.compute_mag(audio)   
            out = disc(mag.unsqueeze(1))                      # pass magnitude to discriminator
            outputs.append(out)
        return outputs

