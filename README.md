# Bird voice recognition
Klasyfikacja nagrań dźwiękowych ptaków za pomocą modeli ze skupianiem uwagi
Architektura bazowa: bulbul
Modyfikacja: wprowadzenie mechanizmu skupiania uwagi

Uruchamianie programu:

1. Trening modeli:

    usage:  train.py [-h] -ag WITH_AG -n MODEL_NAME -d DATASET -e EPOCHS -w WANDB
                 -v WITH_WRONGLIST [-vn NAME_WRONGLIST]

    optional arguments:
    
  -h, --help            show this help message and exit
  
  -ag WITH_AG, --with_AG WITH_AG
                        Model with, or without attention mechanism: True or
                        False
                        
  -n MODEL_NAME, --model_name MODEL_NAME
                        Model name
                        
  -d DATASET, --dataset DATASET
                        Which datasets should be involved: 1 - birdVox , 2 -
                        birdVox and freefield, 3 - birdVox, freefield and
                        TutAcousticScenes, 4 - warblr, 5 - freefield
                        
  -e EPOCHS, --epochs EPOCHS
                        Number of trained epochs
                        
  -w WANDB, --wandb WANDB
                        Model with or without Wandb Logger: True or False
                        
  -v WITH_WRONGLIST, --with_wrongList WITH_WRONGLIST
                        With list of files incorrect classified on validation
                        epoch.
                        
  -vn NAME_WRONGLIST, --name_wrongList NAME_WRONGLIST
                        File name, to which list of incorrect classified on validation
                        epoch files will be .

   przykład:
    
      python src/train.py -ag True -n nazwaModelu -d 1 -e 1 -w True -v False -vn nazwaPliku

2. Testowanie modelu na zbiorze danych Warblr:
    usage:  train.py [-h] -m model_name

    optional arguments:
    
  -m, --model_name            Model name (ex. T+B+F_model_25e_vallist.ckpt)

        python src/check_warblr.py -m T+B+F_model_25e_vallist.ckpt
  
3. Uruchomienie aplikacji Streamlit:
	     
        cd dashboard && streamlit run main.py [ --server.port 9900 --browser.serverAddress 127.0.0.1 ]

4. Pobranie zbiorów danych:

        cd datasets && sh BirdVox_download.sh && freefield_download.sh && tut_download.sh && warblr_download.sh
        
Zawartość poszczególnych plików:

  * src/
    * train.py - uruchomienie procesu uczenia się modelu
    * check_warblr.py - uruchomienie testowania modelu na zbiorze danych Warblr
    * classifier.py - moduł służący do wydania predykcji dla zadanego pliku przez zadany zapisany model sieci
    * models/ - implementacja sieci bazowej oraz ze skupianiem uwagi
    * datamodules/ - implementacja klas wczytujących dane i przetwarzających nagrania na spektrogram (dataset.py - dla zbioru danych; test_dataset.py - dla pojedynczego pliku)
  * dashboard/
    * main.py - uruchomienie aplikacji, obsługa flag
    * audio_info.py - wyświetlanie informacji o nagraniu
    * classification_info.py - wyświetlenie informacji o dokonanej klasyfikacji
    * settings.py - ustawienia wykresów
    * upload.py - obsługa wgranego przez użytkownika nagrania
    * recordings/ - nagrania wyświetlane przez aplikację
  * models/ zapisane modele
  * datasets/
    * pliki .csv z informacją o zbiorach danych
    * skrypty .sh, które po uruchomieniu pobierają zbiory danych
