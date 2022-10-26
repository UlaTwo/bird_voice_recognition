import logging
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import streamlit as st

sys.path.append(str(Path(__file__).parents[1] / 'src'))
from classifier import Classifier

""" Klasa implementująca metody potrzebne do wyświetlenia informacji
o klasyfikacji nagrania przez zadany model splotowej sieci neuronowej
"""
class ClassificationInfo():

    def __init__(self, path, file_name):
        self.path = path
        self.file = file_name

        checkpoint_path = "../models/T+B+F_model_25e_vallist.ckpt"

        self.left_column, self.right_column = st.columns(2)

        # wczytanie wyniku z csv:
        self.csv_path = "recordings/data_streamlit_app.csv"
        self.csv_data = pd.read_csv(self.csv_path)

        # dokonanie predykcji przez model i zapisanie jej
        classifier = Classifier(self.path, checkpoint_path, run_model='with_frames')
        self.model_prediction = classifier.model_prediction

    # wyświetlenie predykcji modelu
    def display_model_prediction(self):
        self.left_column.write("Klasyfikacja nagrania według modelu:")
        if self.model_prediction == 1:
            self.right_column.write("<strong>w nagraniu pojawia się odgłos ptaka</strong>", unsafe_allow_html=True)
        else:
            self.right_column.write("w nagraniu nie pojawia się odgłos ptaka")

    # wyświetlenie predykcji zawartej w danych
    def display_data_prediction(self):
        exist_in_csv = self.file[:-4] in self.csv_data.itemid.values

        if exist_in_csv:
            csv_row = self.csv_data.loc[self.csv_data['itemid'] == self.file[:-4]]
            self.left_column.write("Klasyfikacja nagrania zawarta w danych: ")
            data_prediction = int(csv_row['hasbird'])
            if data_prediction == 1:
                self.right_column.write("<strong>w nagraniu pojawia się odgłos ptaka</strong>", unsafe_allow_html=True)
            else:
                self.right_column.write("w nagraniu nie pojawia się odgłos ptaka")

            self.left_column.write("Czy predykcja modelu jest zgodna z predykcją zawartą w danych?")
            if self.model_prediction == data_prediction:
                self.right_column.write("<p style=\"color: green\"><strong>Tak</strong>", unsafe_allow_html=True)
            else:
                self.right_column.write("<p style=\"color: red\"><strong>Nie</strong>", unsafe_allow_html=True)


    def display(self):
        self.display_model_prediction()
        self.display_data_prediction()
