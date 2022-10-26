from math import floor

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

"""Implementacja klasy wczytującej dane
   i przetwarzającej nagrania na mel-spektrogram dla zbioru danych
"""
class Dataset(Dataset):

    # Argumenty:
    #   lista ścieżek do zbiorów danych
    #   lista ścieżek do plików csv z informacją o zbiorach danych
    def __init__(self, csv_paths, file_paths):

        dfs = []

        self.file_paths_list = []
        self.df_lengths = []
        index_file_paths = 0

        # wczytanie plików csv
        for csv_path in csv_paths:
            new_df = pd.read_csv(csv_path, dtype={'itemid': 'string', 'hasbird': np.float32})
            dfs.append(new_df)

            self.df_lengths.append(len(new_df))
            self.file_paths_list.extend([file_paths[index_file_paths] for i in range(len(new_df))])
            index_file_paths += 1

        df = pd.concat(dfs)

        self.file_names = []
        self.labels = []

        # utworzenie listy nazw plików oraz listy ich etykiet
        for i in range(0, len(df)):
            self.file_names.append(df.iloc[i, 0])
            self.labels.append(df.iloc[i, 2])

        # utworzenie listy ścieżek do plików
        self.file_paths = []
        for file_path in file_paths:
            self.file_paths.append(file_path)

        # utworzenie funkcji przekształacjących nagranie
        self.mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=1261, n_mels=80,
                                                                   window_fn=torch.hamming_window,
                                                                   f_min=50, f_max=12000)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.crop = transforms.CenterCrop((80, 700))

    def __len__(self):

        return len(self.file_names)

    """ Przekształcenie pojedynczego elementu ze zbioru danych,
        zwraca: mel-spektrogram, etykietę nagrania oraz jego nazwę
    """
    def __getitem__(self, index):

        data_section = 0
        path = self.file_paths_list[index]+"/"+self.file_names[index]+".wav"

        # pobranie nagranie i przekształcenie go do obiektu torch.Tensor
        waveform, sample_rate = torchaudio.load(path)
        # ustwienie tylko 1 kanału
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        # utworznie mel-spektogramu
        specgram = self.mel_spectogram(waveform)

        specgram_length = specgram.size()[2]
        specgram_height = specgram.size()[1]

        if specgram_length < 700:
            specgram = transforms.Pad((0, 0, 700-specgram_length, 0))(specgram)
        elif specgram_length > 700:
            specgram = self.crop(specgram)

        if specgram_height > 80:
            specgram = self.crop(specgram)

        # transform from amplitud scale to decibels
        # transformacja amplitudy do skali decybeli
        transformed_amp_to_db = self.amplitude_to_db(specgram)

        # normalizacja
        tensor_minusmean = transformed_amp_to_db - transformed_amp_to_db.mean()
        sound_formatted = tensor_minusmean / tensor_minusmean.abs().max()

        return sound_formatted, self.labels[index], self.file_names[index]


"""Klasa ustawiająca podział danych na zbiór treningowy, walidacyjny i testowy
   oraz implementująca funkcje wczytania danych 
   (ze skorzystaniem z funkcjonalności klasy Dataset)
"""
class DataModule(pl.LightningDataModule):

    def __init__(self, csv_paths, file_paths, batch_size, num_workers, num_dataset):
        super().__init__()
        self.batch_size = batch_size
        self.csv_paths = csv_paths
        self.file_paths = file_paths
        self.num_workers = num_workers
        self.num_dataset = num_dataset

    """Ustawienie podziału zbioru"""
    def setup(self, stage=None):

        size_train_set = floor(0.8*self.num_dataset)
        size_val_set = floor(0.05*self.num_dataset)
        size_test_set = floor(0.15*self.num_dataset)

        # na wypadek, gdyby wartości zostały zaokrąglone w dół
        size_train_set += (self.num_dataset-size_train_set-size_test_set-size_val_set)
        birdvox_dataset = Dataset(self.csv_paths, self.file_paths)
        self.train_set, self.test_set, self.val_set = torch.utils.data.random_split(
            birdvox_dataset, [size_train_set, size_test_set, size_val_set], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)
