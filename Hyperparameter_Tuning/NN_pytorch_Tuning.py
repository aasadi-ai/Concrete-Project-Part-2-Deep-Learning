#%%
from Datasets_DataLoaders.customDatasets import TabularDataset
import sys
sys.path.append("C:\\Users\\afa30\\Desktop\\concreteNet")
import torch
from Models.Classifier import *
from Utilities.dataformater import DataFormater
from Datasets_DataLoaders.customDataLoaders import dataLoaderTabular
#%%
X_train,X_validation,X_test,y_train,y_validation,y_test = DataFormater().preProcessing(toNumpy=True)
trainData,valData,testData = dataLoaderTabular(X_train,X_validation,X_test,y_train,y_validation,y_test)
basicNN = BinaryClassifier("tab")
basicNN,trainLoss,valLoss = train(basicNN,trainData,valData,epochs=200)
accuracy(basicNN,valData)

