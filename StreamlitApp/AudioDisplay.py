import streamlit as st

import torch

import torchaudio
import matplotlib.pyplot as plt


import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class AudioDisplay():
    
    def __init__(self,path, file_name):
        self.path = path
        self.file_name = file_name
        self.waveform, self.sample_rate = torchaudio.load(path)
        self.mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate=44100,n_fft=1261, n_mels=80, window_fn=torch.hamming_window, f_min=50, f_max = 12000)
        amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.specgram_toDb = amplitude_to_db(self.mel_spectogram(self.waveform) )
        self.specgram = self.mel_spectogram(self.waveform)
        
        
        self.num_channels, self.num_frames = self.waveform.shape
        self.time_axis = torch.arange(0, self.num_frames) / self.sample_rate

    def display_waveform(self):
        st.write("Oryginalna częstotliwość próbkowania przetwarzanego nagrania: {}".format(self.sample_rate) )
        
        fig, ax = plt.subplots(self.num_channels, 1)
        if self.num_channels == 1:
            ax = [ax]
        for c in range(self.num_channels):
            ax[c].plot(self.time_axis, self.waveform[c], linewidth=1)
            ax[c].grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplituda [arbitrary unit]")
        plt.title('Przebieg fali akustycznej wybranego nagrania')
        st.pyplot(fig)


    def display_melspecgram(self):

        fig, axs = plt.subplots()
        axs.title.set_text('MelSpektrogram')
        #tutaj troszkę nie wiem, jak je ponazywać
#         axs.set_ylabel('ylabel')
#         axs.set_xlabel('xlabel')
        #jest dokładnie to samo
#         specgram_im = self.specgram.log2()[0,:,:].detach().numpy()
        specgram_im = self.specgram_toDb[0,:,:].detach().numpy()

        axs = plt.imshow(specgram_im)
        st.pyplot(fig)


    def display_audio_information(self):
        st.write('Wybrane nagranie: ', self.file_name)
        st.audio(self.path)
        self.display_waveform()      
        self.display_melspecgram()