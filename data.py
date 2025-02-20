import pandas as pd
from tqdm.notebook import tqdm
import cv2


class Dataset:
    def __init__(self, folder : str, save_in_ram : bool =True):
        """
        folder: str
            Path to the folder containing the data
        save_in_ram: bool
            If True, the images will be loaded in memory
        """
        self.folder = folder
        self.train_annotation = pd.read_csv(f"{folder}/train/annotation.csv")
        self.test_annotation = pd.read_csv(f"{folder}/test/annotation.csv")
        self.save_in_ram = save_in_ram
        self.train_patients = self.train_annotation["patient"].unique()
        self.test_patients = self.test_annotation["patient"].unique()

        self.train_x = {
            patient: [] for patient in self.train_patients
        }
        self.train_y = {
            patient: [] for patient in self.train_patients
        }
        print("Loading train data")
        for i, row in tqdm(self.train_annotation.iterrows(), total=len(self.train_patients)):
            if save_in_ram:
                img = cv2.imread(f"{folder}/{row['path']}")
            else:
                img = f"{folder}/{row['path']}"
            self.train_x[row["patient"]].append(img)
            self.train_y[row["patient"]].append(row["target"])

        self.test_x = {
            patient: [] for patient in self.test_patients
        }
        self.test_y = {
            patient: [] for patient in self.test_patients
        }
        print("Loading test data")
        for i, row in tqdm(self.test_annotation.iterrows(), total=len(self.test_patients)):
            if save_in_ram:
                img = cv2.imread(f"{folder}/{row['path']}")
            else:
                img = f"{folder}/{row['path']}"
            self.test_x[row["patient"]].append(img)
            self.test_y[row["patient"]].append(row["target"])
        print("Data loaded")
        print(f"Train patients: {len(self.train_patients)}")
        print(f"Test patients: {len(self.test_patients)}")
        