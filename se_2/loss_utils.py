import torch
import torch.nn as nn
import torch.nn.functional as Func
import torchaudio


######################################################################
# Utility Loss Functions
######################################################################

class L1MagLoss(nn.Module):
    def forward(self, clean_mag, pred_mag):
        return torch.mean(torch.abs(clean_mag - pred_mag))


class LogMagLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, clean_mag, pred_mag):
        clean_log = torch.log(clean_mag + self.eps)
        pred_log = torch.log(pred_mag + self.eps)
        return torch.mean(torch.abs(clean_log - pred_log))


class SpectralConvergenceLoss(nn.Module):
    def forward(self, clean_mag, pred_mag):
        num = torch.norm(clean_mag - pred_mag, p='fro')
        den = torch.norm(clean_mag, p='fro') + 1e-8
        return num / den


######################################################################
# Multi-Resolution STFT Loss
######################################################################

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.configs = [
            (1024, 256, 1024),
            (512, 128, 512),
            (256, 64, 256),
            (512, 256, 512),    # training resolution
        ]

        self.stfts = nn.ModuleList([
            torchaudio.transforms.Spectrogram(
                n_fft=fft,
                hop_length=hop,
                win_length=win,
                power=None
            ).to(device)
            for (fft, hop, win) in self.configs
        ]) 

        self.eps = 1e-7

    def spectral_convergence(self, S_clean, S_pred):
        return torch.norm(S_clean - S_pred, p='fro') / (torch.norm(S_clean, p='fro') + self.eps)

    def log_stft_mag(self, S_clean, S_pred):
        return torch.mean(torch.abs(torch.log(S_clean + self.eps) - torch.log(S_pred + self.eps)))

    def forward(self, clean_wave, rec_wave):
        total_sc = 0.0
        total_mag = 0.0
        for stft in self.stfts:
            S_clean = torch.abs(stft(clean_wave))
            S_pred  = torch.abs(stft(rec_wave))

            total_sc  += self.spectral_convergence(S_clean, S_pred)
            total_mag += self.log_stft_mag(S_clean, S_pred)

        return (total_sc + total_mag) / len(self.stfts)

######################################################################
# SI-SDR Loss
######################################################################

def si_sdr_loss(clean, enhanced, eps=1e-8):
    clean_energy = torch.sum(clean ** 2, dim=-1, keepdim=True)
    proj = torch.sum(clean * enhanced, dim=-1, keepdim=True) * clean / (clean_energy + eps)
    noise = enhanced - proj

    ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    si_sdr = 10 * torch.log10(ratio + eps)

    return -torch.mean(si_sdr)  # maximize SI-SDR


######################################################################
# Mel Spectrogram Loss
######################################################################

class MelSpectrogramLoss(nn.Module):
    def __init__(self, device, sample_rate=16000, n_mels=80):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels
        ).to(device)

    def forward(self, clean_wave, rec_wave):
        mel_clean = self.mel(clean_wave)
        mel_pred  = self.mel(rec_wave)
        return torch.mean(torch.abs(mel_clean - mel_pred))


######################################################################
# Composite Loss (FINAL TOTAL LOSS)
######################################################################

class CompositeSpeechEnhancementLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1_mag = L1MagLoss()
        self.log_mag = LogMagLoss()
        self.spec_conv = SpectralConvergenceLoss()
        self.mrstft = MultiResolutionSTFTLoss(device)
        self.mel = MelSpectrogramLoss(device)

    def forward(self,
                clean_mag, pred_mag,
                clean_wave, rec_wave):

        L1_mag_loss = self.l1_mag(clean_mag, pred_mag)

        LogMag_loss = self.log_mag(clean_mag, pred_mag)

        SpecConv_loss = self.spec_conv(clean_mag, pred_mag)

        MRSTFT_loss = self.mrstft(clean_wave, rec_wave)

        SISDR_loss = si_sdr_loss(clean_wave, rec_wave)

        Mel_loss = self.mel(clean_wave, rec_wave)

        total_loss = (
            1.0 * L1_mag_loss +
            1.0 * LogMag_loss +
            0.5 * SpecConv_loss +
            0.5 * MRSTFT_loss +
            0.1 * SISDR_loss +
            0.002 * Mel_loss
        )

        return {
            "total_loss": total_loss,
            "L1_mag_loss": L1_mag_loss,
            "LogMag_loss": LogMag_loss,
            "SpecConv_loss": SpecConv_loss,
            "MRSTFT_loss": MRSTFT_loss,
            "SISDR_loss": SISDR_loss,
            "Mel_loss": Mel_loss
        }

