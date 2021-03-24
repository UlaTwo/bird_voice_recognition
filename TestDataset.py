from torch.utils.data import  Dataset
import torch
import torchaudio
import numpy as np
from torchvision import transforms

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
        self.resize = transforms.Resize((80,700))
        self.cropp = transforms.CenterCrop((80,700))
    
    def __getitem__(self, index):
        
        print(self.file_path)
        waveform = self.resample(self.waveform)
        # utworzenie Mal Spektogramu
        specgram = self.mel_spectogram(waveform)
        print("size: ", specgram.size())
        if specgram.size()[2]<700:
            specgram  = self.resize(specgram)
            print("resize")
        else:
            specgram  = self.cropp(specgram)
            print("cropp")
        
#         specgram  = self.cropp(specgram)
        print(specgram.size())
        # transformacja za skali amplitud do decybeli
        transformedAmpToDB = self.amplitude_to_db(specgram)
        
        # normalizacja
        tensor_minusmean = transformedAmpToDB - transformedAmpToDB.mean()
        soundFormatted = tensor_minusmean/tensor_minusmean.abs().max()

        #PYTANIE: czy tak jest bardzo nieelegancko?
        #jeśli nie dam tutaj tego rzutowania na float32, to w cross_entropy jest LongTensor zamiast Float/tensor
        return soundFormatted, np.float32( self.labels[index] )
    
    def __len__(self):
        return len(self.labels)