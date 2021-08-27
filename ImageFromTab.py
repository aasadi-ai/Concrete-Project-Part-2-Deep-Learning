import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from featureEngineering import featureEngineering
from Utilities import Utils

class ImageFromTabular(Dataset):
    def __init__(self,X,y):
        imgs = []
        for i in range(len(X)):
            row = np.array(df.iloc[i])
            img = np.reshape(row,(28,28))
            imgScaled = utilities.standardize(img,False)*-1
            imgs.append(imgs)


        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]

utilities = Utils()
_,_,df = utilities.loadData()
X,y = featureEngineering(df)
testClass = ImageFromTabular(X,y)


