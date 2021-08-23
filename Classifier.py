import torch
from architectures import SimpleClassifierArchitecture
from Utilities import Utils,TabularDataset
from torch.utils.data import Dataset,DataLoader

class SimpleClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = SimpleClassifierArchitecture

    def forward(self,X):
        return self.layers(X)

def train(model,dataLoader,epochs=100,lr=0.01):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr)

    for epoch in range(epochs):
        for X,y in dataLoader:
            optimizer.zero_grad()
            yHat = model(X)
            loss = criterion(yHat.squeeze(),y.squeeze())
            loss.backward()
            optimizer.step()
            print(f"{epoch}:{loss.item()}")

def loadDataNN():
    X_train,X_validation,X_test,y_train,y_validation,y_test = Utils().splitData()
    trainData = TabularDataset(X_train,y_train)
    validationData = TabularDataset(X_validation,y_validation)
    testData = TabularDataset(X_test,y_test)
    trainDataLoader = DataLoader(trainData,batch_size=64,shuffle=True)
    validationDataLoader = DataLoader(validationData,batch_size=64,shuffle=True)
    testDataLoader = DataLoader(testData,batch_size=64,shuffle=False)
    return trainDataLoader,validationDataLoader,testDataLoader

testModel = SimpleClassifier()
trainDataLoader,validationDataLoader,testDataLoader = loadDataNN()
train(testModel,trainDataLoader)
