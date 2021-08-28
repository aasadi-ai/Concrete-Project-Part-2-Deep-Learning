import torch
import numpy as np
from architectures import *
from Utilities import Utils,TabularDataset,loadDataNN
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score

class BinaryClassifier(torch.nn.Module):
    def __init__(self,architecture):
        super().__init__()
        architectures = {"tab":tabularClassifier,"img":imageClassifier}
        self.layers = architectures[architecture]
    def forward(self,X):
        return self.layers(X)

def train(model,trainDataLoader,testDataLoader,epochs=500,lr=0.01):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr)
    losses = []

    for epoch in range(epochs):
        for X,y in trainDataLoader:
            optimizer.zero_grad()
            yHat = model(X)
            loss = criterion(yHat.squeeze(),y.squeeze())
            loss.backward()
            optimizer.step()
            #print(f"Accuracy:{accuracy(model,X,y)}")
            #print(f"{epoch}:{loss.item()}")
            losses.append(loss.item())
    return losses

def accuracy(model,dataloader):
    accuracies = []
    for X,y in dataloader:
        yHat = model(X)>0.5
        accuracies.append(accuracy_score(yHat.numpy(),y.numpy()))
    return np.mean(accuracies)

utilities = Utils()
test = BinaryClassifier("tab")
trainDataLoader,validationDataLoader,testDataLoader = loadDataNN()
losses = train(test,trainDataLoader,validationDataLoader)
print(accuracy(test,validationDataLoader))

