from customDatasets import TabularDataset,ImageFromTabular
from Utilities.dataformater import DataFormater
from torch.utils.data import Dataset,DataLoader

def dataLoaderTabular():
    X_train,X_validation,X_test,y_train,y_validation,y_test = DataFormater().splitData()
    trainData = TabularDataset(X_train,y_train)
    validationData = TabularDataset(X_validation,y_validation)
    testData = TabularDataset(X_test,y_test)
    trainDataLoader = DataLoader(trainData,batch_size=64,shuffle=True)
    validationDataLoader = DataLoader(validationData,batch_size=500,shuffle=True)
    testDataLoader = DataLoader(testData,batch_size=64,shuffle=False)
    return trainDataLoader,validationDataLoader,testDataLoader

def dataLoaderTabularToImg():
    pass