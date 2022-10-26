"""
Program uruchamiający trening splotowej sieci neuronowej
scharakteryzowanej przez zadana przez użytkownika parametry
"""
import argparse
import os
import sys

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from datamodules.dataset import DataModule
from models.cnn import CNN
from models.cnn_ag import CNNAG

"""Dodanie argumantów do parsera """
def add_arguments_parser(ArgumentParser):

    ArgumentParser.add_argument("-ag", "--with_AG", required=True,
                                help="Model with, or without attention mechanism: True or False")
    ArgumentParser.add_argument("-n", "--model_name", required=True, help="Model name")
    ArgumentParser.add_argument("-d", "--dataset", required=True,
                                help="Which datasets should be involved: 1 - birdVox , 2 - birdVox and freefield, 3 - birdVox, freefield and TutAcousticScenes, 4 - warblr, 5 - freefield")
    ArgumentParser.add_argument("-e", "--epochs", required=True, help="Number of trained epochs")
    ArgumentParser.add_argument("-w", "--wandb", required=True,
                                help="Model with or without Wandb Logger: True or False")

    ArgumentParser.add_argument("-v", "--with_wrongList", required=True,
                                help="With list of files incorrect classified on validation epoch.")
    ArgumentParser.add_argument("-vn", "--name_wrongList", required=False,
                                help="List name of files incorrect classified on validation epoch.")


"""Ustawienie zmiennych opisujących zadany zbiór 
w zależności od podanych przez użytkownika argumentów
"""
def set_dataset(arguments):
    num_dataset = 0
    csv_paths = []
    file_paths = []

    if arguments['dataset'] == "1":
        wandb_logger = WandbLogger(project="birdVox-NeuralNetwork")
        csv_paths.append('./datasets/BirdVoxDCASE20k_info.csv')
        file_paths.append('./datasets/BirdVox/data')
        num_dataset = 20000

    elif arguments['dataset'] == "2":
        wandb_logger = WandbLogger(project="birdVox+Freefield-NeuralNetwork")
        csv_paths.append('./datasets/BirdVoxDCASE20k_info.csv')
        csv_paths.append('./datasets/freefield1010_info.csv')
        file_paths.append('./datasets/BirdVox/data')
        file_paths.append('./datasets/freefield1010/data')
        num_dataset = 27690

    elif arguments['dataset'] == "3":
        wandb_logger = WandbLogger(project="TUT+birdVox+Freefield-NeuralNetwork")
        csv_paths.append('./datasets/BirdVoxDCASE20k_info.csv')
        csv_paths.append('./datasets/freefield1010_info.csv')
        csv_paths.append('./datasets/TUT-acoustic_scenes_2017_info.csv')
        file_paths.append('./datasets/BirdVox/data')
        file_paths.append('./datasets/freefield1010/data')
        file_paths.append('./datasets/TUT-acoustic-scenes-2017-development/data')
        num_dataset = 30810

    elif arguments['dataset'] == "4":
        wandb_logger = WandbLogger(project="Warblr-NeuralNetwork")
        csv_paths.append('./datasets/warblrb10k_info.csv')
        file_paths.append('./datasets/Warblr/data')
        num_dataset = 8000

    elif arguments['dataset'] == "5":
        wandb_logger = WandbLogger(project="Freefield-NeuralNetwork")
        csv_paths.append('./datasets/freefield1010_info.csv')
        file_paths.append('./datasets/freefield1010/data')
        num_dataset = 7690
    else:
        sys.exit('Error: wrong -d value - dataset value')

    return num_dataset, csv_paths, file_paths, wandb_logger


"""Ustawienie zmiennej określającej liczbę epok"""
def set_epochs(arguments):

    epochs = 0
    try:
        epochs = int(arguments["epochs"])
    except ValueError:
        sys.exit('Error: wrong -e value - epoch value')

    return epochs


""" Ustawienie zmiennych dotyczących procesu treningu sieci """
def set_trainer(arguments, epochs, wandb_logger):

    if arguments["wandb"] == "True":
        trainer = pl.Trainer(
            logger=wandb_logger,
            log_every_n_steps=50,
            max_epochs=epochs,
            gpus=0,
            progress_bar_refresh_rate=50
        )
    elif arguments["wandb"] == "False":
        trainer = pl.Trainer(
            log_every_n_steps=50,
            max_epochs=epochs,
            gpus=0,
            progress_bar_refresh_rate=50
        )
    else:
        sys.exit('Error: wrong -q value - wandb value')

    return trainer


"""Ustawienie modelu splotowej sieci neuronowej:
z lub bez mechanizmu skupiania uwagi, 
w zależności od zadanego przez użytkownika argumentu
"""
def set_model(arguments):

    if arguments["with_AG"] == "True":
        model = CNNAG()
    elif arguments["with_AG"] == "False":
        model = CNN()
    else:
        sys.exit('Error: wrong -ag value - with_AG value')
    return model


"""Zapis listy nagrań, 
które zostały żle zaklasyfikowane w ostatniej epoce procesu walidacji
"""
def save_validation_wrong_list(arguments, model):
    if arguments["with_wrongList"] == "True":
        if arguments["name_wrongList"] is not None:
            file_name = arguments["name_wrongList"]
        else:
            file_name = "wrong_classified_validation_file_list"
        with open('./validation_wrong_list/' + file_name + '.txt', 'w') as filehandle:
            for listitem in model.validation_wrong_classified[-1]:
                filehandle.write('%s\n' % listitem)


def main():

    # utworzenie zmiennej z argumentami zadanymi przez użytkownika
    ap = argparse.ArgumentParser()
    add_arguments_parser(ap)
    args = vars(ap.parse_args())

    num_dataset, csv_paths, file_paths, wandb_logger = set_dataset(args)
    epochs = set_epochs(args)

    batch_size = 32
    num_workers = 24

    trainer = set_trainer(args, epochs, wandb_logger)
    
    # utworzenie klasy implementującej zbiór danych
    birdvox_dm = DataModule(csv_paths, file_paths, batch_size, num_workers, num_dataset)
    model = set_model(args)

    # uruchomienie treningu
    trainer.fit(model, birdvox_dm)
    
    # zapisanie utworzonego modelu
    trainer.save_checkpoint("./models/" + args["model_name"] + ".ckpt")

    result = trainer.test(model)

    wandb.finish()

    save_validation_wrong_list(args, model)


if __name__ == "__main__":
    main()
