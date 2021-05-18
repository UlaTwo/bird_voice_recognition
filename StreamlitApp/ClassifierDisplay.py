import streamlit as st

import pandas as pd
import pytorch_lightning as pl

import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Classifier.Classifier import Classifier

class ClassifierDisplay():
    def __init__(self,path, file_name):
        self.path = path
        self.file = file_name
        
        checkpoint_path="./models/T+B+F_model_25e_vallist_sparkling_thunder_1.ckpt"
        self.trainer = pl.Trainer()
        
        self.left_column, self.right_column = st.beta_columns(2)
        #wczytanie wyniku z csv:
        self.csv_path = "./StreamlitApp/Recordings/DataStreamlitApp.csv"
        self.csv_data = pd.read_csv(self.csv_path)
        classifier = Classifier(self.path, checkpoint_path, run_model='with_frames')
        self.model_prediction = classifier.model_prediction

    def display_model_prediction(self):
        self.left_column.write("Klasyfikacja nagrania według modelu:")
        if self.model_prediction==1:
            self.right_column.write("w nagraniu pojawia się odgłos ptaka")
        else:
            self.right_column.write("w nagraniu nie pojawia się odgłos ptaka")

    def display_data_prediction(self):
        exist_in_csv = self.file[:-4] in self.csv_data.itemid.values

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
