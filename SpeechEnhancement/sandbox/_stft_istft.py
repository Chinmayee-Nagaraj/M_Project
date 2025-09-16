import torch
import torchaudio
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from _dataloading import VCTK_DEMAND_Dataset

def plot_spectrogram(stft_batch, sample_rate=16000, hop_length=256, title="Spectrogram"):
    """
    Visualize the magnitude spectrogram of the first audio in the batch.
    Args:
        stft_batch: (batch, freq, frames) complex tensor
    """
    # Take first example in batch
    stft = stft_batch[0]  # (freq, frames)

    # Convert to magnitude (absolute value)
    magnitude = torch.abs(stft).cpu().numpy()

    # Convert to dB for better visualization
    magnitude_db = 20 * torch.log10(torch.clamp(torch.abs(stft), min=1e-10)).cpu().numpy()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(magnitude_db, origin="lower", aspect="auto",
               cmap="magma", extent=[0, magnitude.shape[1]*hop_length/sample_rate,
                                     0, sample_rate/2])
    plt.colorbar(format="%+2.f dB")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()



trainset_dir = "../../dataset/VCTK_DEMAND/trainset"
#transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64) #TODO
train_ds = VCTK_DEMAND_Dataset(trainset_dir)#, transform=transform)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

for clean, noisy, length in train_loader:
    clean = torch.squeeze(clean, 1)
    noisy = torch.squeeze(noisy, 1)

    # Convert clean waveforms to STFT
    stft_output = torch.stft(
        clean,
        n_fft=512,
        hop_length=256,
        win_length=512,
        return_complex=True
    )

    # Convert noisy waveforms to STFT
    stft_output_noisy = torch.stft(
        noisy,
        n_fft=512,
        hop_length=256,
        win_length=512,
        return_complex=True
    )

    print("STFT shape:", stft_output.shape)  # (batch, freq, frames)
    # Plot spectrogram of first audio
    plot_spectrogram(stft_output, sample_rate=16000, hop_length=256)

    # Convert back to waveform
    reconstructed = torch.istft(
        stft_output,
        n_fft=512,
        hop_length=256,
        win_length=512,
        return_complex=True
    )

    print("Reconstructed shape:", reconstructed.shape)  # (batch, time)
    assert waveforms == reconstructed
    break


   
