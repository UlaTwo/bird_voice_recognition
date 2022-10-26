import os

from pydub import AudioSegment
from streamlit.uploaded_file_manager import UploadedFile


class ExtensionNotSupported(Exception):
    pass

""" Pobranie pliku w formacie .wav lub .mp3 """
def process_upload(uploaded_file: UploadedFile):
    extension = os.path.splitext(uploaded_file.name)[1]
    if extension not in ['.wav', '.mp3']:
        raise ExtensionNotSupported

    if extension == ".wav":
        audio = AudioSegment.from_wav(uploaded_file)
    elif extension == '.mp3':
        audio = AudioSegment.from_mp3(uploaded_file)

    audio = audio.set_channels(1)

    # Wyeksportowanie pliku w formacie .wav
    audio.export('recordings/upload.wav', format="wav")
