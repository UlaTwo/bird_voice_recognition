"""Sprawdzenie działania zadanego modelu dla zbioru danych Warblr"""

import os
import argparse
import sys

import pandas as pd

from classifier import Classifier

"""Dodanie argumantów do parsera """
def add_arguments_parser(ArgumentParser):

    ArgumentParser.add_argument("-m", "--model_name", required=True, help="Model name")
    ArgumentParser.add_argument("-ag", "--with_AG", required=True,
                                help="Model with, or without attention mechanism: True or False")

def main():

    # utworzenie zmiennej z argumentami zadanymi przez użytkownika
    ap = argparse.ArgumentParser()
    add_arguments_parser(ap)
    args = vars(ap.parse_args())

    file_list = []

    for filename in os.listdir("./datasets/Warblr/data"):
        file_list.append(filename)

    # wczytanie wyniku z csv:
    csv_path = "./datasets/warblrb10k_info.csv"
    csv_data = pd.read_csv(csv_path)

    checkpoint_path = "./models/"+args["model_name"]

    with_AG = args["with_AG"]

    good_predictions = []
    number_of_file = 0

    # pętla sprawdzająca predykcję dla każdego z nagrań
    for file in file_list:
        print(number_of_file)
        number_of_file += 1
        classif = Classifier("./datasets/Warblr/data/" + file, checkpoint_path, with_AG)
        print("file: ", file)
        csv_row = csv_data.loc[csv_data['itemid'] == file[:-4]]
        print("hasbird: ", int(csv_row['hasbird']))
        data_prediction = int(csv_row['hasbird'])
        if classif.model_prediction == data_prediction:
            good_predictions.append(1)
        else:
            good_predictions.append(0)

    # zapis do pliku nazw plików oraz informacji, czy predykcja była dla niego poprawna
    with open('predictions_for_warblr.txt', 'w') as filehandle:
        for ide in range(len(good_predictions)):
            filehandle.write('%s,' % file_list[ide])
            filehandle.write('%s\n' % good_predictions[ide])

    print("Liczba dobrych predykcji: ", sum(good_predictions))
    print("Liczba wszystkich predykcji: ", len(good_predictions))


if __name__ == "__main__":
    main()
