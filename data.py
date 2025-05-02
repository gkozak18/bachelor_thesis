import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
import cv2
import os


class Dataset(Dataset):
    def __init__(self, folder : str, size : tuple = (320, 320), save_in_ram : bool =True, device : str = "cuda", float_target : bool = True):
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
        self.float_target = float_target
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
                img = f"{folder}/{os.path.split(row['path'])[-1]}"
            self.x.append(img)
            self.y.append(row["target"])
        print("Data loaded")
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        if not self.save_in_ram:
            img = cv2.imread(self.x[idx])
        else:
            img = self.x[idx]
        x = self.transform(img).to(self.device)
        if self.float_target:
            y = torch.tensor([self.y[idx]]).to(self.device).float()
        else:
            y = int(self.y[idx])
            # y = F.one_hot(torch.tensor(y, dtype=torch.long), num_classes=4).to(self.device, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long, device=self.device)
        return x, y


class LOOCV_Dataset(Dataset):
    def __init__(self, data: list[tuple[str, float]], size : tuple = (320, 320), device : str = "cuda"):
        """
        data: list[tuple[str, float]]
            data[i] = (path, target)
        """
        self.data = data
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size)
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx][0])
        x = self.transform(img).to(self.device)
        y = torch.tensor([self.data[idx][1]]).to(self.device).float()
        return x, y


class LOOCV_datasets:
    def __init__(self, data_folders: list, size : tuple = (320, 320)):
        self.size = size
        self.data = self.load_data(data_folders)
        print(f"Loaded {len(self.data)} patients")
        self.LOOCV_datasets = self.get_LOOCV_datasets(self.data)
    
    def load_data(self, data_folders: list):
        data : dict[str, list[list, float]] = {}
        for folder in data_folders:
            annotation = pd.read_csv(f"{folder}/annotation.csv")
            for i, row in tqdm(annotation.iterrows(), total=len(annotation)):
                patient = folder + str(row["patient"])
                if patient not in list(data.keys()):
                    data[patient] = [[], float(row["target"])]
                row = f"{folder}/{os.path.split(row["path"])[1]}"
                if row not in data[patient][0]:
                    data[patient][0].append(row)
        return data
    
    def get_LOOCV_datasets(self, data):
        """
        data: dict[str, list[list, float]]
            data[patient] = [paths, target]
        """
        datasets = []
        for patient in data.keys():
            train_data = []
            test_data = []
            for p in data.keys():
                if p != patient:
                    for img_path in data[p][0]:
                        train_data.append((img_path, data[p][1]))
                else:
                    for img_path in data[p][0]:
                        test_data.append((img_path, data[p][1]))
            datasets.append((train_data, test_data))
        return datasets
    
    def __len__(self):
        return len(self.LOOCV_datasets)
    
    def __getitem__(self, idx):
        train_data, test_data = self.LOOCV_datasets[idx]
        train_dataset = LOOCV_Dataset(train_data, size=self.size)
        test_dataset = LOOCV_Dataset(test_data, size=self.size)
        return train_dataset, test_dataset
