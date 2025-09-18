# On the fly pre-processing --> make 2 seconds chunk of audio file and check fot a good entropy
# dataloader while loading take random 1sec part form the 2-sec audio


import os
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Define constants
TARGET_SR = 16000       # sampling rate = 16kHz
TARGET_LEN = TARGET_SR * 2   # 2 seconds
CROP_LEN = TARGET_SR         # 1 second crop for training

MAX_TRIES = 5

N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 256

def compute_entropy(wav, num_bins=64):
    hist = torch.histc(wav, bins=num_bins, min=-1.0, max=1.0)
    p = hist / torch.sum(hist)
    p = p[p > 0]
    return -torch.sum(p * torch.log2(p))

def is_informative(wav, threshold=3.0):
    return compute_entropy(wav) > threshold

class VCTK_DEMAND_Dataset(Dataset):
    def __init__(self, data_dir):

        self.clean_dir = os.path.join(data_dir, 'clean_testset_wav/clean_testset_wav')
        self.noisy_dir = os.path.join(data_dir, 'noisy_testset_wav/noisy_testset_wav')

        self.clean_wav_names = [f for f in os.listdir(self.clean_dir) if f.endswith(".wav")]
        self.clean_wav_names = sorted(self.clean_wav_names)
        #self.transform = transform #TODO

    def __len__(self):
        return len(self.clean_wav_names)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_names[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_names[idx])

        # Load audio
        clean_wave, clean_sr = torchaudio.load(clean_file) 
        noisy_wave, sr = torchaudio.load(noisy_file)
        #normalize (bool, optional) – When True, this function converts the native sample type to float32. Default: True.
        #channel_first - Default: True; it returns the tensor as [channel, time] 

        assert clean_sr == sr

        # If file is not in 16kHz (TARGET_SR), resample:
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            clean_wave = resampler(clean_wave)
            noisy_wave = resampler(noisy_wave)
         
        assert clean_wave.shape == noisy_wave.shape        
        length = clean_wave.shape[-1]

        # Retry loop within the same file
        for _ in range(MAX_TRIES):
            if (length < TARGET_LEN) :
                pad_size = TARGET_LEN - length    
                clean_w = F.pad(clean_wave, (0, pad_size))
                noisy_w = F.pad(noisy_wave, (0, pad_size))

            else:
                # randomly cut 2 seconds segment
                start2s = random.randint(0, length - TARGET_LEN)
                clean_w = clean_wave[:, start2s:start2s + TARGET_LEN]
                noisy_w = noisy_wave[:, start2s:start2s + TARGET_LEN]

            if is_informative(clean_w):
                break

        clean_wave = clean_w
        noisy_wave = noisy_w
 
        # Random 1s crop for training
        start1s = random.randint(0, TARGET_LEN - CROP_LEN)
        clean_wave = clean_wave[:, start1s:start1s+ CROP_LEN]
        noisy_wave = noisy_wave[:, start1s:start1s+ CROP_LEN]

        return clean_wave, noisy_wave, length



if __name__ == "__main__":
    trainset_dir = "../mp-dataset"

    train_ds = VCTK_DEMAND_Dataset(trainset_dir)#, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)


    # to check the working of the above code

    for clean, noisy, length in train_loader:
        print(clean.shape, noisy.shape, length)
        break


#To load the train and test set #TODO

'''
def load_data(ds_dir, batch_size, n_cpu, cut_len):
#     torchaudio.set_audio_backend("sox_io")         # in linux
    train_dir = os.path.join(ds_dir, 'train')
    test_dir = os.path.join(ds_dir, 'test')

    train_ds = DemandDataset(train_dir, cut_len)
    test_ds = DemandDataset(test_dir, cut_len)

    train_dataset = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True,
                                                drop_last=True, num_workers=n_cpu)
    test_dataset = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False,
                                               drop_last=False, num_workers=n_cpu)

    return train_dataset, test_dataset

'''

