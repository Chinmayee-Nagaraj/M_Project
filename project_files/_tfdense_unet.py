import torch
import torch.nn as nn
import torchaudio
import os
import soundfile as sf
from _dense_block import ImprovedDenseBlock
from _bottleneck import TimeFreqGlobalBottleneck

# --------------------------------------------
# Encoder Block
# --------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, hidden_dim):
        super().__init__()
        self.dense = ImprovedDenseBlock(in_channels, growth_rate)
        out_ch = in_channels + len(self.dense.blocks) * growth_rate
        # Conv input = all dense outputs, output = hidden_dim
        self.conv = nn.Conv1d(out_ch, hidden_dim, kernel_size=3, padding=1)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.dense(x)           # (B, T, out_ch)
        x = x.transpose(1, 2)       # (B, out_ch, T)
        x = self.conv(x)            # (B, hidden_dim, T)
        x = self.act(x)
        return x.transpose(1, 2)    # (B, T, hidden_dim)


# --------------------------------------------
# Decoder Block
# --------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, hidden_dim):
        super().__init__()
        self.act = nn.PReLU()
        self.dense = ImprovedDenseBlock(in_channels, growth_rate)
        out_ch = in_channels + len(self.dense.blocks) * growth_rate
        self.conv = nn.Conv1d(out_ch, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.act(x)
        x = self.dense(x)           # (B, T, out_ch)
        x = x.transpose(1, 2)       # (B, out_ch, T)
        x = self.conv(x)            # (B, hidden_dim, T)
        x = x.transpose(1, 2)       # (B, T, hidden_dim)
        return x


# --------------------------------------------
# Encoder × N
# --------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels, growth_rate, hidden_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(in_channels if i == 0 else hidden_dim, growth_rate, hidden_dim)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# --------------------------------------------
# Decoder × N
# --------------------------------------------
class Decoder(nn.Module):
    def __init__(self, in_channels, growth_rate, hidden_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(in_channels if i == 0 else hidden_dim, growth_rate, hidden_dim)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# --------------------------------------------
# Full TFDense-UNet
# --------------------------------------------
class TFDenseUNet(nn.Module):
    def __init__(self, growth_rate=16,
                 num_heads=4, num_blocks=2, num_encoders=2, num_decoders=2):
        super().__init__()

        self.n_fft = 512
        self.hop_length = 256
        self.win_length = 512
        self.window = torch.hann_window(self.win_length)

        self.stft = lambda x: torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window.to(x.device),
            return_complex=True
        )
        self.istft = lambda X, length: torch.istft(
            X, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window.to(X.device),
            length=length
        )

        # STFT output channels
        self.in_channels = self.n_fft // 2 + 1  # 257

        # Hidden dim divisible by num_heads
        hidden_dim = 256

        # Encoder
        self.encoder = Encoder(self.in_channels, growth_rate, hidden_dim, num_layers=num_encoders)

        # Bottleneck
        self.bottleneck = TimeFreqGlobalBottleneck(
            hidden_dim=hidden_dim, num_heads=num_heads, num_blocks=num_blocks
        )

        # Decoder
        self.decoder = Decoder(hidden_dim, growth_rate, hidden_dim, num_layers=num_decoders)

        # Linear projection: hidden_dim → STFT bins
        self.output_proj = nn.Linear(hidden_dim, self.in_channels)

    def forward(self, waveform):
        B, T = waveform.shape

        # 1. STFT
        spec = self.stft(waveform)  # (B, F, frames)
        mag = spec.abs()
        phase = spec.angle()

        # 2. Prepare input for encoder: (B, frames, F)
        x = mag.transpose(1, 2)

        # 3. Encoder
        x = self.encoder(x)  # (B, frames, hidden_dim)

        # 4. Bottleneck
        B, T_frames, C = x.shape
        x = self.bottleneck(x, T_frames, 1)  # (B, frames, hidden_dim)

        # 5. Decoder
        x = self.decoder(x)  # (B, frames, hidden_dim)

        # 6. Project back to original freq bins
        x = self.output_proj(x)  # (B, frames, F)

        # 7. Reconstruct spectrogram
        x = torch.relu(x)
        x = x.transpose(1, 2)  # (B, F, frames)
        spec_out = torch.polar(x, phase)  # (B, F, frames)

        # 8. ISTFT
        out_wave = self.istft(spec_out, length=waveform.shape[-1])
        return out_wave


# --------------------------------------------
# Test
# --------------------------------------------
if __name__ == "__main__":
    SAMPLE_WAV = "p226_001.wav"

    # Load using soundfile
    waveform_np, sample_rate = sf.read(SAMPLE_WAV)  # (samples,) or (samples, channels)

    # Convert to tensor
    waveform = torch.from_numpy(waveform_np).float()

    # Ensure shape (B, T)
    if len(waveform.shape) > 1:  # multi-channel
        waveform = waveform[:, 0]  # take first channel
    waveform = waveform.unsqueeze(0)  # (B=1, T)

    model = TFDenseUNet()
    out_wave = model(waveform)

    print("Input shape :", waveform.shape)
    print("Output shape:", out_wave.shape)