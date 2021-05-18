import torch

import pytorch_lightning as pl

import sys
import argparse 

import wandb
from pytorch_lightning.loggers import WandbLogger

import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from CnnModels.CNN_Model import CNN_Model
from CnnModels.CNN_Model_with_AG import CNN_Model_with_AG
from DatasetModules.Dataset_Module import DataModule

def add_arguments_parser(ArgumentParser):
    
    # Add the arguments to the parser
    ArgumentParser.add_argument("-ag", "--with_AG", required=True, help="Model with, or without attention mechanism: True or False")
    ArgumentParser.add_argument("-n", "--model_name", required=True, help="Model name")
    ArgumentParser.add_argument("-d", "--dataset", required=True, help="Which datasets should be involved: 1 - birdVox , 2 - birdVox and freefield, 3 - birdVox, freefield and TutAcousticScenes, 4 - warblr, 5 - freefield")
    ArgumentParser.add_argument("-e", "--epochs", required=True, help="Number of trained epochs")
    ArgumentParser.add_argument("-w", "--wandb", required=True, help= "Model with or without Wandb Logger: True or False")

    ArgumentParser.add_argument("-v", "--with_wrongList", required=True, help="With list of files incorrect classified on validation epoch.")
    ArgumentParser.add_argument("-vn", "--name_wrongList", required=False, help="List name of files incorrect classified on validation epoch.")

def set_dataset(arguments):
    num_dataset = 0
    csv_paths = []
    file_paths = []
    
    if arguments['dataset']=="1":
        wandb_logger = WandbLogger(project="birdVox-NeuralNetwork")
        csv_paths.append('./BirdVox/BirdVoxDCASE20k.csv')
        file_paths.append('./BirdVox/data')
        num_dataset = 20000
    elif arguments['dataset']=="2":
        wandb_logger = WandbLogger(project="birdVox+Freefield-NeuralNetwork")
        csv_paths.append( './BirdVox/BirdVoxDCASE20k.csv')
        csv_paths.append( './freefield1010/ff1010bird_metadata_2018.csv')
        file_paths.append('./BirdVox/data')
        file_paths.append('./freefield1010/data')
        num_dataset = 27690
    elif arguments['dataset']=="3":
        wandb_logger = WandbLogger(project="TUT+birdVox+Freefield-NeuralNetwork")
        csv_paths.append( './BirdVox/BirdVoxDCASE20k.csv')
        csv_paths.append( './freefield1010/ff1010bird_metadata_2018.csv')
        csv_paths.append('./TUT-acoustic-scenes-2017/TUT-acoustic-scenes-2017-development.csv')
        file_paths.append('./BirdVox/data')
        file_paths.append('./freefield1010/data')
        file_paths.append('./TUT-acoustic-scenes-2017/data')
        num_dataset = 30810
    elif arguments['dataset']=="4":
        wandb_logger = WandbLogger(project="Warblr-NeuralNetwork")
        csv_paths.append('./Warblr/warblrb10k_public_metadata.csv')
        file_paths.append('./Warblr/data')
        num_dataset = 8000
    elif arguments['dataset']=="5":
        wandb_logger = WandbLogger(project="Freefield-NeuralNetwork")
        csv_paths.append('./freefield1010/ff1010bird_metadata_2018.csv')
        file_paths.append('./freefield1010/data')
        num_dataset = 7690
    else:
        sys.exit('Error: wrong -d value - dataset value')
        
    return num_dataset, csv_paths, file_paths, wandb_logger

def set_epochs(arguments):
    
    epochs = 0
    
    try:
        epochs = int(arguments["epochs"])
    except ValueError:
        sys.exit('Error: wrong -e value - epoch value')
        
    return epochs

def set_trainer(arguments, epochs, batch_size, num_workers):

    if arguments["wandb"] == "True":
        trainer = pl.Trainer(
            logger = wandb_logger,  #W&B integration
            log_every_n_steps = 50, #set the logging frequency
            max_epochs=epochs,           #number of epochs  
            gpus =0,
            progress_bar_refresh_rate=50
        )
    elif arguments["wandb"] == "False":
        trainer = pl.Trainer(
            log_every_n_steps = 50, #set the logging frequency
            max_epochs=epochs,           #number of epochs  
            gpus =0,
            progress_bar_refresh_rate=50
        )
    else:
        sys.exit('Error: wrong -q value - wandb value')
        
    return trainer

def set_model(arguments):
    
    if arguments["with_AG"]=="True":
        model = CNN_Model_with_AG()
    elif arguments["with_AG"]=="False":
        model = CNN_Model()
    else:
        sys.exit('Error: wrong -ag value - with_AG value')
    return model

def save_validation_wrong_list(arguments, model):
        if arguments["with_wrongList"] == "True":
            if arguments["name_wrongList"] is not None:
                file_name = arguments["name_wrongList"]
            else:
                file_name = "wrong_classified_validation_file_list"
            with open('./validation_wrong_list/'+file_name+'.txt', 'w') as filehandle:
                for listitem in model.validation_wrong_classified[-1]:
                    filehandle.write('%s\n' % listitem)

def main():
    
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    add_arguments_parser(ap)
    args = vars(ap.parse_args())
       
    num_dataset, csv_paths, file_paths, wandb_logger = set_dataset(args)
    epochs = set_epochs(args)
    
    batch_size = 32
    num_workers = 24

    trainer = set_trainer(args, epochs, batch_size, num_workers)

    birdvox_dm = DataModule(csv_paths, file_paths, batch_size, num_workers, num_dataset)
    model = set_model(args)

    trainer.fit(model, birdvox_dm)
    trainer.save_checkpoint("./models/"+args["model_name"]+".ckpt")

    result = trainer.test(model)

    wandb.finish()
    
    save_validation_wrong_list(args, model)


if __name__ == "__main__":
    main()