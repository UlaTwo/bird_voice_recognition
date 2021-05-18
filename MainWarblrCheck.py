import pandas as pd

import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from DatasetModules.Dataset_Module import Dataset
from Classifier.Classifier import Classifier


def main():
    
    listFiles=[]

    for filename in os.listdir("./Warblr/data"):
        listFiles.append(filename)

    #wczytanie wyniku z csv:
    csv_path = "./Warblr/warblrb10k_public_metadata.csv"
    csv_data = pd.read_csv(csv_path)
    
    checkpoint_path="./models/B_5_5_21_57_AG_E25.ckpt"
    with_AG='True'
    # warblrb10k_public_metadata.csv
    goodPredictions = []
    number_of_file = 0
    for file in listFiles:
        print(number_of_file)
        number_of_file+=1
        classif = Classifier("./Warblr/data/"+file, checkpoint_path, with_AG)
        print("file: ", file)
        csv_row = csv_data.loc[ csv_data['itemid']==file[:-4] ] 
        print("hasbird: ", int(csv_row['hasbird']) )
        data_prediction = int(csv_row['hasbird'] )
        if classif.model_prediction == data_prediction:
            goodPredictions.append(1)
        else:
            goodPredictions.append(0)

    #zapis do pliku nazw plików oraz informacji, czy predykcja była dla niego poprawna
    with open('predictions_for_warblr.txt', 'w') as filehandle:
        for ide in range(len(goodPredictions)):
            filehandle.write('%s,' % listFiles[ide])
            filehandle.write('%s\n' % goodPredictions[ide])

    print("Liczba dobrych predykcji: ",sum(goodPredictions))
    print("Liczba wszystkich predykcji: ", len(goodPredictions))
    

if __name__ == "__main__":
    main()