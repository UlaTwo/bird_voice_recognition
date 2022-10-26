from functools import partial

import librosa
import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import streamlit as st
import torch
import torchaudio

import pandas as pd

from settings import *

""" Paramtery potrzebne do utworzenia spektrogramu """
SR = 44100                          # częstotliwość próbkowania
N_FFT = 1261                        # długość okna czasowego
N_MELS = 80                         # liczba pasm melowych
F_MIN = 50                          # minimalna częstotliwość próbkowania
F_MAX = 12000                       # maksymalna częstotliwość próbkowania 
WINDOW_FN = torch.hamming_window    # funkcja tworząca okno czasowe

frames_to_time = partial(librosa.core.convert.frames_to_time, sr=SR, hop_length=N_FFT // 2, n_fft=N_FFT)
time_to_frames = partial(librosa.core.convert.frames_to_time, sr=SR, hop_length=N_FFT // 2, n_fft=N_FFT)


""" Klasa implementująca metody potrzebne do wyświetlenia informacji o nagraniu:
    - nazwy nagrania oraz zbioru, z którego pochodzi
    - paska umożliwiającego odtworzenie wybranego dźwięku
    - przebiegu fali akustycznej zarejestrowanej w nagraniu
    - mel-spektrogramu wybranego nagrania
"""
class AudioInfo():

    def __init__(self, path, filename):

        self.path = path
        self.filename = filename
        self.waveform, self.sample_rate = torchaudio.load(path)  # type: ignore
        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            n_mels=N_MELS,
            window_fn=WINDOW_FN,
            f_min=F_MIN,
            f_max=F_MAX,
        )
        

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec = self.amplitude_to_db(self.melspec_transform(self.waveform)).detach().numpy()

        self.num_channels, self.num_frames = self.waveform.shape
        self.time_axis = torch.arange(0, self.num_frames) / self.sample_rate
        
        self.csv_path = "recordings/data_streamlit_app.csv"
        self.csv_data = pd.read_csv(self.csv_path)

    def display(self):
        st.write(f'Wybrane nagranie: `{self.filename}` ')
        exist_in_csv = self.filename[:-4] in self.csv_data.itemid.values
        if exist_in_csv:
            csv_row = self.csv_data.loc[self.csv_data['itemid'] == self.filename[:-4]]
            data = csv_row['datasetid'].values[0]
            st.write(f'Nagranie pochodzi ze zbioru o id: `{data}`')
        st.audio(self.path)

        st.write("Oryginalna częstotliwość próbkowania przetwarzanego nagrania: {}".format(self.sample_rate))

        fig = plotly.subplots.make_subplots(rows=2, cols=1,
                                            row_heights=[0.3, 0.7],
                                            shared_xaxes=True,
                                            vertical_spacing=0.1)

        # Wykres fali akustycznej
        # -------------
        waveform = go.Scatter(
            x=self.time_axis,
            y=self.waveform[0],
            mode='lines',
            line=dict(width=1, color=WAVEFORM_COLOR),
            hovertemplate="Czas: %{x:.2f} s<br>Wartość: %{y:.3f}",
        )

        # Przebieg czasowy sygnału wybranego nagrania

        fig.add_trace(waveform, row=1, col=1)
        fig.update_yaxes(  
            row=1,
            col=1,
            side='left',
            range=[-1, 1],
            fixedrange=True,
            gridcolor=GRID_COLOR,
            showgrid=False,
            linecolor=BORDER_COLOR,
            mirror=True,
            ticks='outside',
            tickvals=[-1, 1],
            tickfont=dict(size=10),
            tickcolor=BORDER_COLOR,
            title='<b>Amplituda</b>',
        )

        fig.update_xaxes( 
            row=1,
            col=1,
            tickfont=dict(size=10),
            gridcolor=GRID_COLOR,
            linecolor=BORDER_COLOR,
            mirror=True,
            ticks='outside',
            tickmode='linear',
            tick0=0,
            dtick=1,
            tickcolor=BORDER_COLOR,
            zeroline=False,
        )

        # Wykres mel-spektrogramu
        # -------------
        spectrogram = go.Heatmap(  # type: ignore
            z=self.spec[0],
            x=frames_to_time([x for x in range(self.spec.shape[2])]),
            coloraxis='coloraxis',
            hovertemplate="Ramka: %{customdata} (%{x:.2f} s)<br>Pasmo melowe: %{y}<br>Wartość (dB): %{z:.2f}<extra></extra>",
            customdata=[[x for x in range(self.spec.shape[2])] for _ in range(self.spec.shape[1])],
        )

        fig.add_trace(spectrogram, row=2, col=1)
        fig.update_yaxes(
            row=2,
            col=1,
            side='left',
            fixedrange=True,
            gridcolor=GRID_COLOR,
            showgrid=False,
            linecolor=BORDER_COLOR,
            mirror=True,
            ticks='outside',
            tickfont=dict(size=10),
            tickcolor=BORDER_COLOR,
            title='<b>Pasmo melowe</b>',
        )
        fig.update_xaxes(
            row=2,
            col=1,
            tickfont=dict(size=10),
            gridcolor=GRID_COLOR,
            linecolor=BORDER_COLOR,
            mirror=True,
            ticks='outside',
            tickmode='linear',
            tick0=0,
            dtick=1,
            tickcolor=BORDER_COLOR,
            zeroline=False,
            title='<b>Czas [s]</b>',
        )
        fig.update_layout(
            coloraxis=dict(
                colorscale='Viridis',
                showscale=False
            ),
        )

        # Ułożenie wykresów
        # -------------
        pio.templates["custom"] = go.layout.Template(
            layout=dict(
                paper_bgcolor=PAPER_BGCOLOR,
                plot_bgcolor=PLOT_BGCOLOR,
            ),
        )
        fig.update_layout(
            template='plotly_white+custom',
            showlegend=False,
            height=400,
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0,
                pad=0,
            ),
            font=dict(
                family=FONT_FAMILY,
            ),
            hoverlabel=dict(
                font=dict(
                    family=FONT_FAMILY,
                ),
            ),
        )

        st.plotly_chart(fig, config=dict(
            displayModeBar=False
        ))
