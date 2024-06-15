

import os, shutil
import ultralytics
from ultralytics import YOLO
import prepareDataset as pD
from pathlib import Path


class trainingYolo:
    """
    Cette classe permet de faire l'entrainement d'un model YOLO et de sauvegarder le
    résultat de l'entrainement dans un dossier TRAIN_RESULT.
    """

    def __init__(self) -> None:
        
        self.dataset = None
        self.model = None
        self.model_name = None
        self.model_path = None


    def downloadModelYolo(self, model_name):
        """
        Télécharge le modèle YOLO spécifié et le sauvegarde dans le répertoire self.save_dir.
        """
        self.model_name = model_name
        # Télécharge et charge le modèle YOLO
        self.model = YOLO(self.model_name)
        chemin_fichier = os.path.abspath(self.model_name)
        
        fichier  = chemin_fichier+'.pt'
        dossier = "C:/Users/zakaria.gamane/Desktop/HELOU_KOMLAN_MAWULE/projet_personnel/TP_ETUDE_IAI_ING2/projet_reconnaissance_faciale/projet_de_reconnaissance_faciale/Model"

        
        if not os.path.exists(dossier):
            os.makedirs(dossier)

        # Chemin complet vers le fichier déplacé dans le dossier de destination
        destination = os.path.join(dossier, os.path.basename(fichier))
        self.model_path = dossier+self.model_name+'.pt'

        # Déplacement du fichier vers le dossier de destination
        shutil.move(fichier, destination)

        return self

        



    def useDatasetYolo(self,dataset:str, model_path:str, epochs : int, img_size: int, batch_size : int,
                   project_path: str, run_name : str):
        

        if self.model_path == None:
            self.model_path = model_path
                     
        self.data_path = dataset+'/dataset.yaml'   # Chemin vers le fichier .yaml des données
        self.epochs = epochs                          # Nombre d'époques pour l'entraînement
        self.img_size = img_size                       # Taille des images d'entrée
        self.batch_size = batch_size                   # Taille de la batch pour l'entraînement
        self.project_path = project_path               # Chemin vers le répertoire de sortie du projet
        self.run_name = run_name                       # Nom du dossier pour cette exécution

        return self
    

    def trainingOfModelYolo(self ):
        """
        Méthode pour entraîner le modèle YOLOv8.
        """
        # Charger le modèle YOLOv8
        model = YOLO(self.model_path)

        # Démarrer l'entraînement
        model.train(
            data=self.data_path,
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            project=self.project_path,
            name=self.run_name
        )

if __name__ == "__main__":
    model = trainingYolo()

    model.useDatasetYolo("dataset_games_of_trones",
                         'Model/yolov8s.pt',
                         10, 640, 50,'RESULTAT/', 'RUN/'
                         )
    model.trainingOfModelYolo()
    
