import numpy as np

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pydub import AudioSegment

import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from CnnModels.CNN_Model_with_AG import CNN_Model_with_AG
from CnnModels.CNN_Model import CNN_Model

from DatasetModules.TestDataset import TestDataset

class Classifier():
    
    def __init__(self,path, checkpoint_path,with_AG='False', run_model='normal'):
        
        self.path = path

        if with_AG=='True':
            self.model = CNN_Model_with_AG.load_from_checkpoint(checkpoint_path)
        elif with_AG=='False':
            self.model = CNN_Model.load_from_checkpoint(checkpoint_path)
        
        self.trainer = pl.Trainer()
        
        if run_model=='normal':
            self.model_prediction = self.run_model()
        elif run_model == 'with_frames':
            self.model_prediction = self.run_model_with_frames()
        
    def run_model_with_frames(self):

        loadedAudio = AudioSegment.from_wav(self.path)

        time = loadedAudio.duration_seconds

        # gdy nagranie jest krótkie zostaje ono zmiksowane z nagraniem z BirdVoxa, na którym nie ma odgłosów ptaków
        if time<10:
            birdVoxSound = AudioSegment.from_file("./Classifier/000db435-a40f-4ad9-a74e-d1af284d2c44.wav")
            #make mixed sound little quiter
            birdVoxSound = birdVoxSound - 3
            loadedAudio = birdVoxSound.overlay(loadedAudio)
            time = loadedAudio.duration_seconds

        second_frame = 8
        
        counter = int(loadedAudio.duration_seconds/second_frame) -1
        model_predictions = []
        
        for i in range(counter):
            cutAudio = loadedAudio[i*second_frame*1000:i*second_frame*1000+10*1000]

            cutAudio.export('processedAudio.wav', format="wav")
            result = self.trainer.test(model=self.model,test_dataloaders = DataLoader(TestDataset( './processedAudio.wav', 1), batch_size = 64) )
            model_predictions.append( result[0]["test_epoch_end_accuracy"] )
                        
        #the end of the sound
        if time%second_frame !=0:
            cutAudio = loadedAudio[time-10:]
            cutAudio.export('processedAudio.wav', format="wav")
            result = self.trainer.test(model=self.model,test_dataloaders = DataLoader(TestDataset( './processedAudio.wav', 1), batch_size = 64) )
            model_predictions.append( result[0]["test_epoch_end_accuracy"] )
        os.remove('./processedAudio.wav')
        if 1 in model_predictions:
            return 1
        else:
            return 0
        
    #run_model, w którym pliki są po prostu wczytywane, a problem zbyt dużego rozmiaru jest rozwiązywany przez cropp i padding
    def run_model(self):
        result = self.trainer.test(model=self.model,test_dataloaders = DataLoader(TestDataset( self.path, 1), batch_size = 64) )
        return result[0]["test_epoch_end_accuracy"]

