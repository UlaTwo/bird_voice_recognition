import streamlit as st
import urllib.error 

import pandas as pd
import numpy as np

from CNN_Audio_Model import CNN_Audio_Model
from TestDataset import TestDataset

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import torchaudio
import os
import matplotlib.pyplot as plt
from pydub import AudioSegment


class Audio():
    def __init__(self,path, file_name):
        self.path = path
        self.file_name = file_name
        self.waveform, self.sample_rate = torchaudio.load(path)
        #tutaj by sie jeszcze resample na wszelki wypadek przydało / albo i nie, bo jest w TestDataset
        mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate=44100,n_fft=1261, n_mels=80, window_fn=torch.hamming_window, f_min=50, f_max = 12000)
        self.specgram = mel_spectogram(self.waveform)

    def display_waveform(self):
        st.write("Shape of original waveform: {}".format(self.waveform.size()) )
        st.write("Original sample rate of waveform: {}".format(self.sample_rate) )
        #wyswietlenie waveform
        'Waveform: '
        fig, ax = plt.subplots()
        ax = plt.plot(self.waveform[0].t().numpy())
        st.pyplot(fig)

    def display_melspecgram(self):
        #wyswietlenie spektogramu
        'MelSpectogram'
        fig, ax = plt.subplots()
        specgram_im = self.specgram.log2()[0,:,:].detach().numpy()
        ax = plt.imshow(specgram_im)
        st.pyplot(fig)

    def display_audio_information(self):
        st.write('Wybrane nagranie: ', self.file_name)
        st.audio(self.path)
        self.display_waveform()      
        self.display_melspecgram()


class Classifier():
    def __init__(self,path, file_name):
        self.path = path
        self.file = file_name
        self.model = CNN_Audio_Model.load_from_checkpoint(checkpoint_path="./model_25e_vallist.ckpt")
        self.trainer = pl.Trainer()
        self.left_column, self.right_column = st.beta_columns(2)
        #wczytanie wyniku z csv:
        self.csv_path = "./BirdVox/BirdVoxDCASE20k.csv"
        self.csv_data = pd.read_csv(self.csv_path)
        self.model_prediction = self.run_model()

    #TODO: to możnaby w sumie jakoś ładniej podzielić
    def run_model(self):
        #i tutaj jakoś sprawdzenie długości nagrania i ewentualne odpalenie kilka razy
        loadedAudio = AudioSegment.from_wav(self.path)

        time = loadedAudio.duration_seconds

        #dodanie ciszy, tak żeby długość nagrania była podzielna przez 10
        if time%10 !=0:
            silent_time = 10.00 - time%10
            silent_time = np.ceil(silent_time)
        else:
            silent_time = 0
        silent_segment = AudioSegment.silent(duration=silent_time*1000) 
        st.sidebar.write(silent_segment.duration_seconds)
        st.sidebar.write(loadedAudio.duration_seconds)
        loadedAudio = loadedAudio + silent_segment
        counter = int(loadedAudio.duration_seconds/10)
        loadedAudio = loadedAudio[0:10*counter*1000]

        st.sidebar.write(loadedAudio.duration_seconds)
        
        model_predictions = []
        for i in range(counter):
            st.sidebar.write(i)
            cutAudio = loadedAudio[i*10*1000:10*(i+1)*1000]
            cutAudio.export('processedAudio.wav', format="wav")
            result = self.trainer.test(model=self.model,test_dataloaders = DataLoader(TestDataset( './processedAudio.wav', 1), batch_size = 64) )
            model_predictions.append( result[0]["test_epoch_end_accuracy"] )

        st.sidebar.write(model_predictions)
        os.remove('./processedAudio.wav')
        if 1 in model_predictions:
            return 1
        else:
            return 0

    
    def display_model_prediction(self):
        self.left_column.write("Klasyfikacja nagrania według modelu:")
        if self.model_prediction==1:
            self.right_column.write("w nagraniu pojawia się odgłos ptaka")
        else:
            self.right_column.write("w nagraniu nie pojawia się odgłos ptaka")

    def display_data_prediction(self):
        exist_in_csv = self.file[:-4] in self.csv_data.itemid.values

        #to jest w sumie głównie na wypadek, gdybym w liście miała też te wgrane, spoza zbioru
        if exist_in_csv:
            csv_row = self.csv_data.loc[ self.csv_data['itemid']==self.file[:-4] ] 
            self.left_column.write("Klasyfikacja nagrania zawarta w danych: ")
            data_prediction = int(csv_row['hasbird'] )
            if data_prediction == 1:
                self.right_column.write("w nagraniu pojawia się odgłos ptaka")
            else:
                self.right_column.write("w nagraniu nie pojawia się odgłos ptaka")
            
            self.left_column.write("Czy predykcja modelu jest zgodna z predykcją zawartą w danych?")
            if self.model_prediction == data_prediction:
                self.right_column.write("Tak")
            else:
                self.right_column.write("Nie")

    def display_prediction_information(self):
        self.display_model_prediction()
        self.display_data_prediction()


class FileUploadedProcessing():
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
    def processing(self):
        read_audio = False
        newAudio = AudioSegment.silent(duration=1)
        if self.uploaded_file is not None:
            extension = self.uploaded_file.name[-4:]
            if extension ==".wav" or extension == ".mp3":
                if extension == ".wav":
                    newAudio = AudioSegment.from_wav(uploaded_file)
                else:
                    newAudio = AudioSegment.from_mp3(uploaded_file)

                newAudio = newAudio.set_channels(1)
                newAudio.export('newSong.wav', format="wav") #Exports to a wav file in the current path.
                read_audio = True
                # else:
                #     time = newAudio.duration_seconds
                #     st.sidebar.write('Dzielność: ',np.ceil(10/time))
                #     repeat = np.ceil(10/time)
                #     newAudio = newAudio*int(repeat)
                #     st.sidebar.write('Nagranie powinno trwać co najmniej 10 sekund! ')
            else:
                st.sidebar.write('Nagranie powinno być w formacie .wav lub .mp3! ')
        return read_audio


try:

    ##### SIDE BAR #####
#     st.sidebar.write("Nagrania pochądzą ze zbioru BirdVox ")

    #SELECT BOX
    listFiles=[]
    
#     for filename in os.listdir("./BirdVox/data/wav"):
#         listFiles.append(filename)

#     file_name = st.sidebar.selectbox( 'Wybierz nagranie: ', listFiles)
    
    #SELECT BOX z tymi błędnymi z validacyjnego
    #uwaga! tu już powoli ci się robi bajzel, bo one pochądzą z działania innego modelu niż ten, który był używany ;) 
    
    if st.sidebar.checkbox('Wyświetl tylko nagrania błędne na zbiorze walidacyjnym'):
        st.sidebar.write("Nagrania pochądzą ze zbioru BirdVox - błędne w walidacyjnym ")
        files_list = []
        with open('wrong_classified_validation_file_names_25e.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                #currentPlace = line[:-1]
                #bez '.wav'
                currentPlace = line[:-1]+'.wav'
                # add item to the list
                files_list.append(currentPlace)

        file_name = st.sidebar.selectbox( 'Wybierz nagranie: ', files_list)
    else:
        for filename in os.listdir("./BirdVox/data/wav"):
            listFiles.append(filename)

        file_name = st.sidebar.selectbox( 'Wybierz nagranie: ', listFiles)
        
    
    #FILE UPLOAD
    st.sidebar.header('Wgraj nagrania: ')
    uploaded_file = st.sidebar.file_uploader("Choose a file")

    fileUpPro = FileUploadedProcessing(uploaded_file)
    read_audio = fileUpPro.processing()

    ##### MAIN BAR #####
    if read_audio:
        path = "./newSong.wav"
        file_name = uploaded_file.name
    else:
        path = "./BirdVox/data/wav/"+file_name

    st.title("Klasyfikacja nagrań zawierających dźwięki ptaków")

    st.header('Nagranie')
    aud = Audio(path, file_name)
    aud.display_audio_information()

    st.header('Klasyfikacja')
    clas = Classifier(path,file_name)
    clas.display_prediction_information()



except urllib.error.URLError as e:
    st.error(
        """
        **Requires internet access.**

        Connection error: %s
    """
        % e.reason
    )