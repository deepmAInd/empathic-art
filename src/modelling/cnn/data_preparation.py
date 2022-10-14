import pandas as pd

import torch
import torchaudio
import torch.nn as nn

import torchaudio.functional as F
import torchaudio.transforms as T


from torch.utils.data.dataloader import DataLoader, Dataset, T_co

import matplotlib.pyplot as plt

import librosa


# /home/michael/Documents/fontys/semester_7/depmAInd/empathic_art/empathic-art/data/reference_df.csv
ANNOTATIONS_DIR = "/home/michael/Documents/fontys/semester_7/depmAInd/empathic_art/empathic-art/data/reference_df.csv"
AUDIO_DIR = "/home/michael/Documents/fontys/semester_7/depmAInd/empathic_art/empathic-art/data/soundfiles"


class SpecgramDataset(Dataset):
    
    def __init__(self, annotations: str = ANNOTATIONS_DIR, audio_dir: str = AUDIO_DIR,  
                transform = None, sample_rate: int = 44100, num_frames: int = 220500) -> None:
        super().__init__()
        self.annotations = pd.read_csv(annotations)
        self.labels = self.annotations.emotion

        self.audio_dir = audio_dir
        self.tranform = transform

        self.sample_rate = sample_rate
        self.num_frames = num_frames
    
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> T_co:

        label = self.labels[index]
        waveform, sr = torchaudio.load(f"{self.audio_dir}/{self.annotations.iloc[index].filename}")
        
        waveform = self._resample_if_necessary(waveform, sr)
        waveform = self._cut_if_necessary(waveform)
        waveform = self._right_pad_if_necessary(waveform)

        specgram = self.tranform(waveform)
        return specgram, label

    def _cut_if_necessary(self, waveform: torch.Tensor) -> torch.Tensor:
        num_frames = waveform.shape[1]
        if num_frames > self.num_frames:
            waveform = waveform[:, :self.num_frames]
        return waveform

    def _right_pad_if_necessary(self, waveform: torch.Tensor) -> torch.Tensor:
        num_frames = waveform.shape[1]
        if num_frames < self.num_frames:
            missing_frames = self.num_frames - num_frames
            last_dim_padding = (0, missing_frames)
            waveform = nn.functional.pad(waveform, last_dim_padding)
        return waveform

    def _resample_if_necessary(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if self.sample_rate != sr:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        return waveform


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


n_fft = 1024
win_length = None
hop_length = 512

spec_transform = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
)


# waveform, sr = torchaudio.load("/home/michael/Documents/fontys/semester_7/depmAInd/empathic_art/empathic-art/data/soundfiles/03-01-01-01-01-01-01.wav")


spec_dataset = SpecgramDataset(
    transform=spec_transform
)

print(f"The number of entries is: {len(spec_dataset)}")
# print(f"The shape of the dataset is: {spec_dataset.shape}")

spec, label = spec_dataset[0]

plot_spectrogram(specgram=spec[0], title=label)

input("Press any key to finish.")