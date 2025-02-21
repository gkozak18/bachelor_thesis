import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm.notebook import tqdm
import cv2
import os


class Dataset(Dataset):
    def __init__(self, folder : str, size : tuple = (320, 320), save_in_ram : bool =True, device : str = "cuda"):
        """
        folder: str
            Path to the folder containing the data
        size: tuple
            Size of the images
        save_in_ram: bool
            If True, the images will be loaded in memory
        device: str
            Device to use
        """
        self.folder = folder
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size)
        ])
        self.annotation = pd.read_csv(f"{folder}/annotation.csv")
        self.save_in_ram = save_in_ram
        self.patients = self.annotation["patient"].unique()

        self.x = []
        self.y = []
        print(f"Loading data from {folder}")
        for i, row in tqdm(self.annotation.iterrows(), total=len(self.annotation)):
            if save_in_ram:
                img = cv2.imread(f"{folder}/{os.path.split(row['path'])[-1]}")
            else:
                img = f"{folder}/{row['path']}"
            self.x.append(img)
            self.y.append(row["target"])
        print("Data loaded")
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        x = self.transform(self.x[idx]).to(self.device)
        y = torch.tensor([self.y[idx]]).to(self.device).float()
        return x, y
