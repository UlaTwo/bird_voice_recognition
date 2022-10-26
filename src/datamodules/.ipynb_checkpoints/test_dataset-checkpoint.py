import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms

"""Implementacja klasy wczytującej dane 
   i przetwarzającej nagrania na mel-spektrogram dla pojedynczego pliku
"""
class TestDataset(Dataset):

    """Argumenty:
        ścieżka do nagrani
        ścieżka do plików csv z informacją o etykiecie nagrania
    """
    def __init__(self, file_path, label):
        self.labels = []
        self.labels.append(label)

        self.file_path = file_path
        self.waveform, self.sample_rate = torchaudio.load(self.file_path)

        # utworznie przekształceń potrzebnych do transformacji nagrania do mel-spektrogramu
        self.mel_spectogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100, n_fft=1261, n_mels=80, window_fn=torch.hamming_window, f_min=50, f_max=12000)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.resample = torchaudio.transforms.Resample(self.sample_rate, 44100)
        self.crop = transforms.CenterCrop((80, 700))

    """ Przekształcenie pojedynczego elementu ze zbioru danych,
        zwraca: mel-spektrogram, etykietę nagrania oraz jego nazwę
    """
    def __getitem__(self, index):

        waveform = self.resample(self.waveform)

        # utworzenie mel-spektrogramu
        specgram = self.mel_spectogram(waveform)

        specgram_length = specgram.size()[2]
        specgram_height = specgram.size()[1]

        # ewentualne zmiany, gdy rozmiar jest nieodpowiedni
        if specgram_length < 700:
            specgram = transforms.Pad((0, 0, 700 - specgram_length, 0))(specgram)
        elif specgram_length > 700:
            specgram = self.crop(specgram)
        if specgram_height > 80:
            specgram = self.crop(specgram)

        # transformacja za skali amplitud do decybeli
        transformed_amp_to_db = self.amplitude_to_db(specgram)

        # normalizacja
        tensor_minusmean = transformed_amp_to_db - transformed_amp_to_db.mean()
        sound_formatted = tensor_minusmean / tensor_minusmean.abs().max()

        return sound_formatted, np.float32(self.labels[index]), self.file_path

    def __len__(self):
        return len(self.labels)
