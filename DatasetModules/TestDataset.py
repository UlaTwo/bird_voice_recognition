from torch.utils.data import  Dataset
import torch
import torchaudio
import numpy as np
from torchvision import transforms

import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class TestDataset(Dataset):
    # Argument list
    # path to the BirdVox-20k csv file
    # path to the BirdVox-20k audio files
    
    def __init__(self, file_path, label):
        self.labels = []
        self.labels.append(label)
            
        self.file_path = file_path
        self.waveform, self.sample_rate = torchaudio.load(self.file_path)
        self.mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate=44100,n_fft=1261, n_mels=80, window_fn=torch.hamming_window, f_min=50, f_max = 12000)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.resample = torchaudio.transforms.Resample(self.sample_rate, 44100)
        self.cropp = transforms.CenterCrop((80,700))
    
    def __getitem__(self, index):
        
        waveform = self.resample(self.waveform)
        # utworzenie Mal Spektogramu
        specgram = self.mel_spectogram(waveform)
        
        specgram_length = specgram.size()[2]
        specgram_height = specgram.size()[1]
        
        if specgram_length<700:
            specgram = transforms.Pad( (0,0,700-specgram_length,0))(specgram)
        elif specgram_length>700:
            specgram  = self.cropp(specgram)

        if specgram_height>80:
            specgram  = self.cropp(specgram)


        # transformacja za skali amplitud do decybeli
        transformedAmpToDB = self.amplitude_to_db(specgram)
        
        # normalizacja
        tensor_minusmean = transformedAmpToDB - transformedAmpToDB.mean()
        soundFormatted = tensor_minusmean/tensor_minusmean.abs().max()

        return soundFormatted, np.float32( self.labels[index] ), self.file_path

    
    def __len__(self):
        return len(self.labels)