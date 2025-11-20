import torch
import torchaudio

class STFTProcessor:
    def __init__(self, device, n_fft=512, win_length=512, hop_length=256):
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.window = torch.hann_window(win_length).to(device)

    def stft_ri(self, clean_wave, noisy_wave):
        clean_spec = torch.stft(
            clean_wave.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True
        )
        noisy_spec = torch.stft(
            noisy_wave.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True
        )

        return clean_spec.real, clean_spec.imag, noisy_spec.real, noisy_spec.imag

    def istft_ri(self, pred_complex, length):
        rec = torch.istft(
            pred_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=length,
            center=True,
            return_complex=False
        )
        return rec.unsqueeze(1)

