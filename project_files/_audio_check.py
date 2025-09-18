import torch    # PyTorch- an open-source deep learning framework, primarily developed by Facebook's AI Research lab (FAIR), which is now known as Meta AI.
import torchaudio   # torchaudio- a library for audio and signal processing with PyTorch.
import matplotlib.pyplot as plt
from IPython.display import Audio   # IPython- Interactive Python, interactive command line shell

print(torch.__version__)
print(torchaudio.__version__)

SAMPLE_WAV = 'p232_001.wav'

metadata = torchaudio.info(SAMPLE_WAV)      # returns an AudioMetaData 
print(metadata)     # (PCM_S: Signed integer linear PCM)

waveform, sample_rate = torchaudio.load(SAMPLE_WAV)

def plot_waveform(waveform, sample_rate):
    print("The audio tensor has shape: ",waveform.shape)
    print("The audio tensor: ", waveform)

    waveform = waveform.numpy()  

    num_channels, num_frames = waveform.shape
    
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show()

plot_waveform(waveform, sample_rate)


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show()

plot_specgram(waveform, sample_rate)

Audio(waveform.numpy()[0], rate=sample_rate)

