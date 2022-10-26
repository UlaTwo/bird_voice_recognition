import os
from pathlib import Path

import pytorch_lightning as pl
from pydub import AudioSegment
from torch.utils.data import DataLoader

from datamodules.test_dataset import TestDataset
from models.cnn import CNN
from models.cnn_ag import CNNAG

"""Moduł służący do wydania predykcji dla zadanego pliku przez zadany zapisany model sieci,
możliwe są dwa sposoby uruchomienia klasyfikacji:

- z podziałem nagrania na ramki (gdy nagranie jest zbyt długie)
  i zmiksowaniem z nagraniem ze zbioru BirdVox (gdy nagranie jest zbyt krótkie)

- tylko z użyciem funkcji crop i padding, gdy nagranie ma niewłaściwy rozmiar

"""
class Classifier():

    def __init__(self, path, checkpoint_path, with_AG='False', run_model='normal'):

        self.path = path

        if with_AG == 'True':
            self.model = CNNAG.load_from_checkpoint(checkpoint_path)
        elif with_AG == 'False':
            self.model = CNN.load_from_checkpoint(checkpoint_path)

        self.trainer = pl.Trainer()

        if run_model == 'normal':
            self.model_prediction = self.run_model()
        elif run_model == 'with_frames':
            self.model_prediction = self.run_model_with_frames()


    """Uruchomienie klasyfikacji modelu dla nagrania,
    w którym pliki są po prostu wczytywane, a problem zbyt dużego lub zbyt małego rozmiaru
    jest rozwiązywany przez funkcje crop i padding
    """
    def run_model(self):
        result = self.trainer.test(model=self.model, test_dataloaders=DataLoader(
            TestDataset(self.path, 1), batch_size=64))
        return result[0]["test_epoch_end_accuracy"]

    
    """Uruchomienie klasyfikacji modelu dla nagrania:

     - problem zbyt dużego rozmiaru jest rozwiązywany poprzez podzielenie nagrania na fragmenty - ramki
       i dokonywania klasyfikacji dla każdego z nich. 
       Ramki nachodzą częściowo na siebie, 
       a ostateczny wynik jest równy 1, jeśli jakikolwiek fragment został zaklasyfikowany 
       jako posiadający odgłos ptaka

     - problem zbyt krótkiego nagrania jest rozwiązywany poprzez zmiksowanie go z nagraniem 
       pochodzącym ze zbioru BirdVox, w którym nie ma odgłosu ptaka
    """
    def run_model_with_frames(self):

        loaded_audio = AudioSegment.from_wav(self.path)

        time = loaded_audio.duration_seconds

        # zmiksowane z nagraniem ze zbioru BirdVoxa, 
        # jeśli nagranie jest zbyt krótkie
        if time < 10:
            root = Path(__file__).parents[0]
            birdvox_sound = AudioSegment.from_file(str(root / "data/000db435-a40f-4ad9-a74e-d1af284d2c44.wav"))
            # lekkie ściszenie nagrania z BirdVox
            birdvox_sound = birdvox_sound - 3
            loaded_audio = birdvox_sound.overlay(loaded_audio)
            time = loaded_audio.duration_seconds

        second_frame = 8    # ustalenie liczby sekund, o którą przesuwana jest ramka

        counter = int(loaded_audio.duration_seconds/second_frame) - 1
        model_predictions = []

        # klasyfikacja nagrania z uwzględnieniem wszystkich ramek, 
        # na które nagranie jest podzielone
        for i in range(counter):
            cut_audio = loaded_audio[i*second_frame*1000:i*second_frame*1000+10*1000]

            cut_audio.export('processedAudio.wav', format="wav")
            result = self.trainer.test(model=self.model, test_dataloaders=DataLoader(
                TestDataset('./processedAudio.wav', 1), batch_size=64))
            model_predictions.append(result[0]["test_epoch_end_accuracy"])

        # klasyfikacja ostatniego fragmentu nagrania
        if time % second_frame != 0:
            cut_audio = loaded_audio[time-10:]
            cut_audio.export('processedAudio.wav', format="wav")
            result = self.trainer.test(model=self.model, test_dataloaders=DataLoader(
                TestDataset('./processedAudio.wav', 1), batch_size=64))
            model_predictions.append(result[0]["test_epoch_end_accuracy"])
        os.remove('./processedAudio.wav')
        if 1 in model_predictions:
            return 1
        else:
            return 0

