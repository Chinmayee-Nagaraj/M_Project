import torch
import torch.nn as nn
import torchaudio


class SISDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio Loss.
    Output is negative SI-SDR (because we minimize loss).
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, target, pred):
        """
        pred:   [B, T] or [B, 1, T]
        target: [B, T] or [B, 1, T]
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        # Zero-mean normalization (important for SI-SDR)
        pred_zm   = pred   - pred.mean(dim=-1, keepdim=True)
        target_zm = target - target.mean(dim=-1, keepdim=True)

        # Project pred onto target
        proj = (
            torch.sum(pred_zm * target_zm, dim=-1, keepdim=True)
            * target_zm
            / (torch.sum(target_zm ** 2, dim=-1, keepdim=True) + self.eps)
        )

        # Error term
        noise = pred_zm - proj

        # SI-SDR
        ratio = (proj ** 2).sum(dim=-1) / (noise ** 2).sum(dim=-1).clamp(min=self.eps)
        sisdr = 10 * torch.log10(ratio + self.eps)

        # Loss = negative SI-SDR
        return -sisdr.mean()


class  MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss:
    - Fast MSE on linear magnitude 
    - Spectral convergence 
    - Log-magnitude loss (perceptual)
    """

    def __init__(self, device):
        super().__init__()
        self.configs = [
            (256, 64, 256),
            (512, 128, 512),
            (1024, 256, 1024)
        ]

        self.windows = nn.ParameterList([
            nn.Parameter(torch.hann_window(win), requires_grad=False)
            for (_, _, win) in self.configs
        ])

        self.eps = 1e-7

        # Loss weights (balanced)
        self.w_mse = 1.0         # Fast convergence
        self.w_sc  = 0.5         # Structural accuracy
        self.w_log = 0.5         # Perceptual sharpness

        self.to(device)

    def _stft_mag(self, x, n_fft, hop, win, window):
        window = window.to(device=x.device, dtype=x.dtype)
        return torch.abs(torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True
        ))    

    def spectral_convergence(self, S_clean, S_pred):
        return torch.norm(S_clean - S_pred, p='fro') / (
            torch.norm(S_clean, p='fro') + self.eps
        )

    def log_mag_loss(self, S_clean, S_pred):
        return torch.mean(torch.abs(
            torch.log(S_clean + self.eps) -
            torch.log(S_pred + self.eps)
        ))

    def forward(self, clean, pred):
        # clean, pred: [B,T] or [B,1,T]
        if clean.dim() == 3:
            clean = clean.squeeze(1)
        if pred.dim() == 3:
            pred = pred.squeeze(1)

        total_loss = 0.0
        for (n_fft, hop, win), window in zip(self.configs, self.windows):
            S_clean = self._stft_mag(clean, n_fft, hop, win, window)
            S_pred  = self._stft_mag(pred,  n_fft, hop, win, window)

            # --- Fast linear MSE ---
            mse_loss = torch.mean((S_clean - S_pred) ** 2)

            # --- Structural convergence ---
            sc_loss = self.spectral_convergence(S_clean, S_pred)

            # --- Perceptual log-magnitude loss ---
            log_loss = self.log_mag_loss(S_clean, S_pred)

            # --- Weighted sum ---
            total_loss += (
                self.w_mse * mse_loss +
                self.w_sc  * sc_loss +
                self.w_log * log_loss
            )

        return total_loss / len(self.configs)


