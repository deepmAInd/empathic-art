import pandas as pd

import torch
import torchaudio
import torch.nn as nn

import torchaudio.functional as F
import torchaudio.transforms as T

from torch.utils.data.dataloader import Dataset, T_co


ANNOTATIONS_DIR = "/home/michael/Documents/fontys/semester_7/depmAInd/empathic_art/empathic-art/data/reference_df.csv"
AUDIO_DIR = "/home/michael/Documents/fontys/semester_7/depmAInd/empathic_art/empathic-art/data/soundfiles"

SAMPLE_RATE = 44100
NUM_FRAMES = 220500


class SpecgramDataset(Dataset):
    
    def __init__(self, annotations: str = ANNOTATIONS_DIR, audio_dir: str = AUDIO_DIR,  
                to_specgram = None, sample_rate: int = SAMPLE_RATE, num_frames: int = NUM_FRAMES) -> None:
        super().__init__()
        self.annotations = pd.read_csv(annotations)
        self.labels = self.annotations.emotion

        self.audio_dir = audio_dir

        self.to_specgram = to_specgram

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

        specgram = self.to_specgram(waveform)

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


# spectogram transform
spec_transform = T.Spectrogram(
    n_fft=1024,
    win_length=None,
    hop_length=512,
    center=True,
    pad_mode="reflect",
    power=2.0,
)
# spectogram dataset
spec_dataset = SpecgramDataset(to_specgram=spec_transform)


# mel spectogram transform
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    win_length=None,
    hop_length=512,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    onesided=True,
    n_mels=128,
    mel_scale="htk",
)
# mel spectogram dataset
mel_dataset = SpecgramDataset(to_specgram=mel_transform)


# mfcc tranform
mfcc_transform = T.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=256,
    melkwargs={
        "n_fft": 2048,
        "n_mels": 256,
        "hop_length": 512,
        "mel_scale": "htk",
    },
)
# mfcc dataset
mfcc_dataset = SpecgramDataset(to_specgram=mfcc_transform)


# lfcc transform
lfcc_tranform = T.LFCC(
    sample_rate=SAMPLE_RATE,
    n_lfcc=256,
    speckwargs={
        "n_fft": 2048,
        "win_length": None,
        "hop_length": 512,
    },
)
# lfcc dataset
lfcc_dataset = SpecgramDataset(to_specgram=lfcc_tranform)

# print(len(spec_dataset))
# print(spec_dataset[0][0].shape)

# def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
#     fig, axs = plt.subplots(1, 1)
#     axs.set_title(title or "Spectrogram (db)")
#     axs.set_ylabel(ylabel)
#     axs.set_xlabel("frame")
#     im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
#     fig.colorbar(im, ax=axs)
#     plt.show(block=False)

# for index, (spec, label) in enumerate(lfcc_dataset):
#     plot_spectrogram(specgram=spec[0], title=label)
#     if index == 5:
#         break

# input("Press enter to exit.")