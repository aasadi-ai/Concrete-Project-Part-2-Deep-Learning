import sys
sys.path.append("C:\\Users\\afa30\\Desktop\\concreteNet")
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.stats import mstats
import torch
from PIL import Image

class DataFormater():
    def __init__(self):
        self.currentMean = None
        self.currentStd = None
        self.winsorLimits = {}

    def loadData(self):
        #Load Data and Rename Columns
        df = pd.read_csv("C:\\Users\\afa30\\Desktop\\concreteNet\\data\\concreteData.csv")
        df.columns = ["Cement","Slag","Ash","Water","Plasticizer","CoarseAgg","FineAgg","Age","CompressiveStrength"]
        print(df["CompressiveStrength"].median())
        df["CompressiveStrength"] = df["CompressiveStrength"]>= df["CompressiveStrength"].median()
        return df.iloc[:,:-1],df["CompressiveStrength"]

    def splitData(self,X,y,trainSplit=0.8,testSplit=0.5,toNumpy=False,seed=43):
        #Split data into train,validation and test set
        X["Target"] = y
        X = shuffle(X,random_state=seed)
        X.reset_index(inplace=True,drop=True)
        y = X["Target"]
        X = X.iloc[:,:-1]
        def split(X,y,splitPercentage):
            splitIndex = int(len(X)*splitPercentage)
            return X.iloc[:splitIndex].copy(),X.iloc[splitIndex:].copy(),y[:splitIndex].copy(),y[splitIndex:].copy()
        X_train,X_temp,y_train,y_temp = split(X,y,trainSplit)
        X_test,X_validation,y_test,y_validation = split(X_temp,y_temp,testSplit)
        if toNumpy:
            return np.array(X_train),np.array(X_validation),np.array(X_test),np.array(y_train),np.array(y_validation),np.array(y_test)
        return X_train,X_validation,X_test,y_train,y_validation,y_test
    
    def preProcessing(self,winsorize=False,standardize=True,toNumpy=False,seed=43):
        X,y = self.loadData()
        X_train,X_validation,X_test,y_train,y_validation,y_test = self.splitData(X,y,seed=seed)
        if winsorize:
            X_train = self.winsorizeOutlier(X_train,useParams=False)
            X_validation = self.winsorizeOutlier(X_validation)
            X_test = self.winsorizeOutlier(X_test)
        if standardize:
            X_train = self.standardize(X_train,useParams=False)
            X_validation = self.standardize(X_validation)
            X_test = self.standardize(X_test)
        if toNumpy:
            return np.array(X_train),np.array(X_validation),np.array(X_test),np.array(y_train),np.array(y_validation),np.array(y_test)
        return X_train,X_validation,X_test,y_train,y_validation,y_test
       
    def standardize(self,X,useParams=True):
        if not useParams:
            self.currentMean = X.mean(0)
            self.currentStd = X.std(0)
        return (X-self.currentMean)/(self.currentStd+0.00001)

    def standardizeAll(self,*X_data,useParams=False):
        if useParams:
            return [self.standardize(X,True) for X in X_data]
        temp = self.standardize(X_data[0],False)
        return tuple([temp]+[self.standardize(X,True) for X in X_data[1:]])

    def winsorizeOutlier(self,X,limits=(0.03,0.03),useParams=True):
        #X = X.copy()
        def winsorize(col):
            sortedCol = col.sort_values(ascending=True)
            bottomLimit = sortedCol[int(len(sortedCol)*limits[0])]
            upperLimit = sortedCol[int(len(sortedCol)*(1-limits[1]))]
            return (bottomLimit,upperLimit)
        if not useParams:
            for colName in X.columns:
                self.winsorLimits[colName]=winsorize(X[colName])
        for colName in X.columns:
            bottomLimit,upperLimit = self.winsorLimits[colName]
            lowerMask = X[colName]<bottomLimit
            upperMask = X[colName]>upperLimit
            X.loc[lowerMask,colName] = bottomLimit
            X.loc[upperMask,colName] = upperLimit
        return X

    def visualizeImg(self,img):
    #reshape row and scale image
        max = np.amax(img)
        min = np.amin(img)
        img = (img-min)/(max-min)
        img*=255
        return Image.fromarray(img)