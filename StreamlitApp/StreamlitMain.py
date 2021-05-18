import streamlit as st
import urllib.error 

import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from StreamlitApp.AudioDisplay import AudioDisplay
from StreamlitApp.ClassifierDisplay import ClassifierDisplay
from StreamlitApp.FileUploadedProcessing import FileUploadedProcessing

try:

    # # # # # # # # # # # # # #
    #####  SIDE  BAR  #####
    
    #SELECT BOX
    listFiles=[]
    
    st.sidebar.write("Nagrania pochądzą ze zbioru BirdVox - błędne w walidacyjnym ")
    for filename in os.listdir("./StreamlitApp/Recordings/data"):
        listFiles.append(filename)

    file_name = st.sidebar.selectbox( 'Wybierz nagranie: ', listFiles)
        
    #FILE UPLOAD
    st.sidebar.header('Wgraj nagrania: ')
    uploaded_file = st.sidebar.file_uploader("Wybierz nagranie")

    fileUpPro = FileUploadedProcessing(uploaded_file)
    read_audio = fileUpPro.processing()

    
    # # # # # # # # # # # # # #
    ##### MAIN BAR #####
    if read_audio:
        path = "./StreamlitApp/newSong.wav"
        file_name = uploaded_file.name
    else:
        path = "./StreamlitApp/Recordings/data/"+file_name

    st.title("Klasyfikacja nagrań zawierających dźwięki ptaków")

    st.header('Nagranie')
    aud = AudioDisplay(path, file_name)
    aud.display_audio_information()

    st.header('Klasyfikacja')

    clas = ClassifierDisplay(path,file_name)

    clas.display_prediction_information()


except urllib.error.URLError as e:
    st.error(
        """
        **Requires internet access.**

        Connection error: %s
    """
        % e.reason
    )