"""
CLASSE: preparationDataset

DESCRIPTION :   Cette classe permet de prendre un fichier zip de dataset  en séparant les données
                en données d'entraînement (80%), de validation (20%).
                STRUCTURE DU DATASET:
                Image :   - TRAIN
                          - VAL

                LAbel:    - TRAIN
                          - VAL


LES METHODES:

init() -> Constructeur

Méthode de classe:
    - _unzip_file(chemin du fichier, dossier d'extraction) -> dézipper
    - _unrar_file(chemin du fichier, dossier d'extraction) -> désarchiver

Méthodes:
    - createDatasetYolo(path_of_uncomprssion_file)
    - preparerDatasetYolo(path_of_dataset)
"""

import os
import zipfile
import rarfile
import random
import shutil

class preparationDatasetYolo:
    """
    Classe permettant de préparer le dataset en le décompressant et en séparant les données
    en ensembles d'entraînement et de validation pour YOLO.
    """

    def __init__(self) -> None:
        self.dataset_path = ""

    @staticmethod
    def _unzip_file(file_path, extract_to):
        """
        Décompresse un fichier .zip dans un répertoire temporaire puis déplace les fichiers directement dans le répertoire cible.

        :param file_path: Chemin vers le fichier .zip à décompresser.
        :param extract_to: Chemin vers le répertoire où les fichiers doivent être déplacés après décompression.
        """
        temp_extract_path = os.path.join(extract_to, 'temp')

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_path)
        
        # Déplacement des fichiers du répertoire temporaire vers le répertoire cible
        for root, dirs, files in os.walk(temp_extract_path):
            for file in files:
                shutil.move(os.path.join(root, file), extract_to)
        
        # Suppression du répertoire temporaire
        shutil.rmtree(temp_extract_path)

    @staticmethod
    def _unrar_file(file_path, extract_to):
        """
        Décompresse un fichier .rar dans un répertoire temporaire puis déplace les fichiers directement dans le répertoire cible.

        :param file_path: Chemin vers le fichier .rar à décompresser.
        :param extract_to: Chemin vers le répertoire où les fichiers doivent être déplacés après décompression.
        """
        temp_extract_path = os.path.join(extract_to, 'temp')

        with rarfile.RarFile(file_path, 'r') as rar_ref:
            rar_ref.extractall(temp_extract_path)
        
        # Déplacement des fichiers du répertoire temporaire vers le répertoire cible
        for root, dirs, files in os.walk(temp_extract_path):
            for file in files:
                shutil.move(os.path.join(root, file), extract_to)
        
        # Suppression du répertoire temporaire
        shutil.rmtree(temp_extract_path)

    def createDatasetYolo(self, dataset_path):
        """
        La méthode décompresse un fichier .zip ou .rar à l'emplacement spécifié.

        :param dataset_path: Chemin vers le fichier .zip ou .rar à décompresser.
        """
        
        # Vérifier si le fichier existe
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Le fichier {dataset_path} n'existe pas.")

        # Le nouveau chemin du dataset
        self.dataset_path = os.path.splitext(dataset_path)[0]

        # Créer le répertoire cible si ce n'est pas déjà fait
        os.makedirs(self.dataset_path, exist_ok=True)

        # Obtenir l'extension du fichier
        file_extension = os.path.splitext(dataset_path)[1].lower()

        # Vérifier l'extension et décompresser en conséquence
        if file_extension == '.zip':
            self._unzip_file(dataset_path, self.dataset_path)
        elif file_extension == '.rar':
            self._unrar_file(dataset_path, self.dataset_path)
        else:
            raise ValueError("Format de fichier non supporté. Veuillez fournir un fichier .zip ou .rar.")

        return self

    def separationOfDatasetYolo(self):
        """
        Méthode pour séparer les données d'un dataset en données d'entrainement et de validation
        en utilisant 80% des données pour l'entrainement et 20% pour la validation de manière aléatoire.
        """

        # Le chemin du dataset
        path = self.dataset_path

        # Dossiers pour les données d'entraînement et de validation
        train_image_path = os.path.join(path, "TRAIN", "Images")
        train_label_path = os.path.join(path, "TRAIN", "Labels")
        val_image_path = os.path.join(path, "VAL", "Images")
        val_label_path = os.path.join(path, "VAL", "Labels")

        # Création des dossiers s'ils n'existent pas
        os.makedirs(train_image_path, exist_ok=True)
        os.makedirs(train_label_path, exist_ok=True)
        os.makedirs(val_image_path, exist_ok=True)
        os.makedirs(val_label_path, exist_ok=True)

        # Liste des fichiers d'images dans le dossier source
        image_extensions = ('.jpg', '.jpeg', '.png')
        images = [f for f in os.listdir(path) if f.lower().endswith(image_extensions)]
        
        # Associer chaque image à son fichier de label correspondant
        labeled_images = [(image, image.rsplit('.', 1)[0] + '.txt') for image in images]

        # Vérification que les fichiers de labels existent pour chaque image
        labeled_images = [(img, lbl) for img, lbl in labeled_images if os.path.exists(os.path.join(path, lbl))]

        # Mélanger les paires d'images et de labels de manière aléatoire
        random.shuffle(labeled_images)

        # Calculer la taille des ensembles d'entraînement (80%) et de validation (20%)
        split_index = int(len(labeled_images) * 0.8)

        # Diviser les fichiers en ensembles d'entraînement et de validation
        train_files = labeled_images[:split_index]
        val_files = labeled_images[split_index:]

        # Déplacement des fichiers dans les répertoires d'entraînement
        for image, label in train_files:
           
            shutil.move(os.path.join(path, image), train_image_path)
            shutil.move(os.path.join(path, label), train_label_path)

        # Déplacement des fichiers dans les répertoires de validation
        for image, label in val_files:

            shutil.move(os.path.join(path, image), val_image_path)
            shutil.move(os.path.join(path, label), val_label_path)

        return self

if __name__ == "__main__":
    # Remplacez ce chemin par le chemin réel de votre fichier .zip ou .rar
    dataset_zip_path = "C:/Users/zakaria.gamane/Desktop/HELOU/dataset_games_of_trones.zip"

    # Instanciation et utilisation de la classe
    preparer = preparationDatasetYolo()
    preparer.createDatasetYolo(dataset_zip_path)
    preparer.separationOfDatasetYolo()
