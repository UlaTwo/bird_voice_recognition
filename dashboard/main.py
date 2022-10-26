"""Uruchomienie interfejsu umożliwiającego użytkownikowi skorzystanie z modelu"""

import os
import urllib.error

import streamlit as st

from audio_info import AudioInfo
from classification_info import ClassificationInfo
from settings import *
from upload import ExtensionNotSupported, process_upload

try:
    st.set_page_config(page_title='Klasyfikacja nagrań zawierających dźwięki ptaków')

    # Ustawienie stylu czcionki
    st.markdown(f"""
    <style>
    {FONT_IMPORT}
    #root * {{
        font-family: "{FONT_FAMILY}" !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Wyświetlenie panelu bocznego
    st.sidebar.header('Wybór nagrania')

    # Pole wyboru nagrania
    st.sidebar.subheader('Wybrane nagrania ze zbiorów BirdVox, Freefield oraz Warblr')
    filenames = []
    for filename in os.listdir("./recordings/data"):
        if filename[-3:]=='wav':
            filenames.append(filename)

    filename = st.sidebar.selectbox('Wybierz nagranie: ', filenames)
    path = "recordings/data/" + filename

    # Pobieranie pliku od użytkownika
    st.sidebar.subheader('Nagranie zewnętrzne')
    uploaded_file = st.sidebar.file_uploader("Wgraj nagranie:", type=['wav', 'mp3'])

    print(uploaded_file)

    if uploaded_file is not None:
        try:
            process_upload(uploaded_file)
        except ExtensionNotSupported:
            st.sidebar.write('Uwaga! Nagranie powinno być w formacie .wav lub .mp3!')

        path = 'recordings/upload.wav'
        filename = uploaded_file.name


    # Panel główny
    st.title("Klasyfikacja nagrań zawierających dźwięki ptaków")

    with st.spinner('Przetwarzanie...'):
        st.header('Nagranie')
        audio_info = AudioInfo(path, filename)
        audio_info.display()

        st.header('Klasyfikacja')
        classification_info = ClassificationInfo(path, filename)
        classification_info.display()


except urllib.error.URLError as e:
    st.error(
        """
        **Requires internet access.**

        Connection error: %s
    """
        % e.reason
    )
