import torch
import numpy as np
from architectures import *
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class BinaryClassifier(torch.nn.Module):
    def __init__(self,architecture):
        super().__init__()
        architectures = {"tab":tabularClassifier,"img":imageClassifier}
        self.layers = architectures[architecture]
    
    def forward(self,X):
        return self.layers(X)

def train(model,trainDataLoader,valDataLoader,epochs=100,lr=0.001):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr)
    trainLoss = []
    valLoss = []

    for epoch in tqdm(range(epochs)):
        trainLossEpoch = []
        valLossEpoch = []
        for X,y in trainDataLoader:
            optimizer.zero_grad()
            yHat = model(X)
            loss = criterion(yHat.squeeze(),y.squeeze())
            loss.backward()
            optimizer.step()
            #print(f"Accuracy:{accuracy(model,X,y)}")
            #print(f"{epoch}:{loss.item()}")
            trainLossEpoch.append(loss.item())
            losses = []
            for X1,y1 in valDataLoader:
                losses.append(criterion(model(X1).squeeze(),y1.squeeze()).item())
            valLossEpoch.append(np.mean(losses))
        trainLoss.append(np.mean(trainLossEpoch))
        if len(valLoss)>10:
            if valLoss[-1]>=valLoss[-10] and valLoss[-3]>=valLoss[-10]:
                break
        valLoss.append(np.mean(valLossEpoch))
    return trainLoss,valLoss

def accuracy(model,dataloader):
    accuracies = []
    for X,y in dataloader:
        yHat = model(X)>0.5
        accuracies.append(accuracy_score(yHat.numpy(),y.numpy()))
    return np.mean(accuracies)



