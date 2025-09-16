import os
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class VCTK_DEMAND_Dataset(Dataset):
    def __init__(self, data_dir, cut_len=16000*2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, 'clean')
        self.noisy_dir = os.path.join(data_dir, 'noisy')
        self.clean_wav_name = [f for f in os.listdir(self.clean_dir) if f.endswith(".wav")]
        self.clean_wav_name = sorted(self.clean_wav_name)
        #self.transform = transform #TODO

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        # Load audio
        clean_ds, clean_sr = torchaudio.load(clean_file) 
        noisy_ds, sr = torchaudio.load(noisy_file)
        #normalize (bool, optional) – When True, this function converts the native sample type to float32. Default: True.
        #channel_first - Default: True; it returns the tensor as [channel, time] 

        #print(clean_ds.shape) #TODO
        assert clean_sr == sr

        # If file is not in 16kHz, resample:
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            clean_ds = resampler(clean_ds)
            noisy_ds = resampler(noisy_ds)
        
        length = clean_ds.shape[-1]
        assert clean_ds.shape == noisy_ds.shape

        if length < self.cut_len:
            pad_size = self.cut_len - length    
            clean_ds = F.pad(clean_ds, (0, pad_size))
            noisy_ds = F.pad(noisy_ds, (0, pad_size))
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            clean_ds = clean_ds[:, wav_start:wav_start + self.cut_len]
            noisy_ds = noisy_ds[:, wav_start:wav_start + self.cut_len]
            
        return clean_ds, noisy_ds, length


trainset_dir = "../../dataset/VCTK_DEMAND/trainset"

#transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64) #TODO

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

