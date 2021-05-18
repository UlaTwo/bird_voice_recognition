import streamlit as st
import urllib.error 

import torchaudio

import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pydub import AudioSegment


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
                    newAudio = AudioSegment.from_wav(self.uploaded_file)
                else:
                    newAudio = AudioSegment.from_mp3(self.uploaded_file)

                newAudio = newAudio.set_channels(1)
                newAudio.export('./StreamlitApp/newSong.wav', format="wav") #Exports to a wav file in the current path.
                read_audio = True

            else:
                st.sidebar.write('Uwaga! Nagranie powinno być w formacie .wav lub .mp3! ')
        return read_audio