from torch.utils.data import  Dataset
import torch
import torchaudio
import numpy as np
from torchvision import transforms
import pytorch_lightning as pl
import pandas as pd

from torch.utils.data import DataLoader, random_split, Dataset, IterableDataset

from math import floor

class Dataset(Dataset):

    # Arguments:
    #   path to the BirdVox-20k csv file
    #   path to the BirdVox-20k audio files  
    def __init__(self, csv_paths,file_paths):
        
        csvData = pd.DataFrame
        csvDatas = []

        self.file_paths_list = []
        self.csvData_lengths = []
        index_file_paths = 0
        
        for csv_path in csv_paths:
            new_csvData = pd.read_csv(csv_path,dtype = {'itemid': 'string','hasbird':np.float32})
            csvDatas.append(new_csvData)

            self.csvData_lengths.append(len(new_csvData))
            self.file_paths_list.extend([file_paths[index_file_paths] for i in range(len(new_csvData))])
            index_file_paths +=1

        csvData = pd.concat(csvDatas)
            
        self.file_names = []
        self.labels = []
        
        for i in range( 0,len(csvData) ):
            self.file_names.append(csvData.iloc[i,0])
            self.labels.append(csvData.iloc[i,2])
              
        self.file_paths = []
        for file_path in file_paths:
            self.file_paths.append(file_path)
            
        self.mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate=44100,n_fft=1261, n_mels=80, 
                                                                   window_fn=torch.hamming_window,
                                                                   f_min=50, f_max = 12000)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.cropp = transforms.CenterCrop((80,700))
    
    def __len__(self):
        
        return len(self.file_names)
    
    def __getitem__(self, index):
        
        data_section = 0
        path = self.file_paths_list[index]+"/"+self.file_names[index]+".wav"
        
        #Load audio file into torch.Tensor object. 
        waveform, sample_rate = torchaudio.load(path)
        #to only 1 channel
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        # creating Mal Spectogramu
        specgram = self.mel_spectogram(waveform)
        
        specgram_length = specgram.size()[2]
        specgram_height = specgram.size()[1]

        if specgram_length<700:
            specgram = transforms.Pad( (0,0,700-specgram_length,0))(specgram)
        elif specgram_length>700:
            specgram  = self.cropp(specgram)
        
        if specgram_height>80:
            specgram  = self.cropp(specgram)
                    
        # transform from amplitud scale to decibels
        transformedAmpToDB = self.amplitude_to_db(specgram)

        # normalization
        tensor_minusmean = transformedAmpToDB - transformedAmpToDB.mean()
        soundFormatted = tensor_minusmean/tensor_minusmean.abs().max()

        return soundFormatted,self.labels[index], self.file_names[index]
    
class DataModule(pl.LightningDataModule):
    
    def __init__(self, csv_paths, file_paths, batch_size, num_workers,num_dataset):
        super().__init__()
        self.batch_size = batch_size
        self.csv_paths = csv_paths
        self.file_paths = file_paths
        self.num_workers = num_workers
        self.num_dataset = num_dataset
    
    def setup(self, stage=None):
        
        size_train_set =floor(0.8*self.num_dataset)
        size_val_set = floor(0.05*self.num_dataset)
        size_test_set = floor(0.15*self.num_dataset)
        
        #na wypadek, gdyby wartości zostały zaokrąglone w dół
        size_train_set += (self.num_dataset-size_train_set-size_test_set-size_val_set)
        birdvox_dataset = Dataset(self.csv_paths, self.file_paths)
        self.train_set, self.test_set, self.val_set = torch.utils.data.random_split(birdvox_dataset, [size_train_set,size_test_set,size_val_set], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size = self.batch_size, num_workers= self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size = self.batch_size, num_workers= self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size = self.batch_size, num_workers= self.num_workers) 