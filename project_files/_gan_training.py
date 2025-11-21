import torch
import torch.nn.functional as F
from _tfdense_unet import TFDenseUNet
from torch import nn


# ------------- Reuse the discriminator definition -------------
def conv_block(in_ch, out_ch, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True)
    )


class SpecBranch(nn.Module):
    def __init__(self, in_ch=2, channels=[32, 64, 128, 256]):
        super().__init__()
        layers = []
        ch = in_ch
        for c in channels:
            layers.append(conv_block(ch, c, kernel=4, stride=2, padding=1))
            ch = c
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(ch, 1)
        )

    def forward(self, x):
        h = self.net(x)
        return self.head(h).squeeze(-1)


class MultiSpecDiscriminator(nn.Module):
    def __init__(self, stft_configs=[(512, 256), (1024, 256), (256, 128)], device='cpu'):
        super().__init__()
        self.stft_configs = stft_configs
        self.device = device
        self.branches = nn.ModuleList([SpecBranch(in_ch=2) for _ in stft_configs])

    def _spec(self, wav, n_fft, hop):
        return torch.stft(
            wav, n_fft=n_fft, hop_length=hop,
            win_length=n_fft, return_complex=True, center=True
        )

    def forward(self, clean, gen):
        scores = []
        for i, (n_fft, hop) in enumerate(self.stft_configs):
            S_clean = self._spec(clean, n_fft, hop)
            S_gen = self._spec(gen, n_fft, hop)
            in_tensor = torch.stack([S_clean.real, S_gen.real], dim=1)
            in_tensor = (in_tensor - in_tensor.mean(dim=[2, 3], keepdim=True)) / \
                        (in_tensor.std(dim=[2, 3], keepdim=True) + 1e-9)
            scores.append(self.branches[i](in_tensor))
        return torch.stack(scores, dim=1)


# ------------- Simple extra losses reused from your training -------------
def si_sdr_loss(est, ref, eps=1e-8):
    ref_energy = torch.sum(ref ** 2, dim=1, keepdim=True)
    scale = torch.sum(ref * est, dim=1, keepdim=True) / (ref_energy + eps)
    est_scaled = scale * ref
    noise = est - est_scaled
    si_sdr = 10 * torch.log10(torch.sum(est_scaled ** 2, dim=1) / (torch.sum(noise ** 2, dim=1) + eps))
    return torch.mean(torch.clamp(20 - si_sdr, min=0))


def multi_res_stft_loss(est, ref):
    fft_sizes = [256, 512, 1024]
    total_loss = 0.0
    for n_fft in fft_sizes:
        est_spec = torch.stft(est, n_fft=n_fft, hop_length=n_fft // 4, return_complex=True)
        ref_spec = torch.stft(ref, n_fft=n_fft, hop_length=n_fft // 4, return_complex=True)
        total_loss += torch.mean((torch.abs(est_spec) - torch.abs(ref_spec)) ** 2)
    return total_loss / len(fft_sizes)
