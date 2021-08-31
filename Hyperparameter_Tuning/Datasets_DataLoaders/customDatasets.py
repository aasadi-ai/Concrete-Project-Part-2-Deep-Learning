import sys
sys.path.append("C:\\Users\\afa30\\Desktop\\concreteNet")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from Utilities.dataformater import DataFormater

class ImageFromTabular(Dataset):
    def __init__(self,X,y):
        imgs = []
        utilities = DataFormater()
        for i in range(len(X)):
            row = np.array(X.iloc[i])
            img = np.reshape(row,(28,28))
            imgScaled = utilities.standardize(img,False)*-1
            imgs.append(torch.tensor(imgScaled))

        self.X = torch.stack(imgs)
        self.y = torch.tensor(np.array(y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]


class TabularDataset(Dataset):
    def __init__(self,X,y):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]
