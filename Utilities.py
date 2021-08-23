import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Utils():
    def __init__(self):
        self.currentMean = None
        self.currentStd = None

    def loadData(self):
        #Load Data and Rename Columns
        df = pd.read_csv("data/concreteData.csv")
        df.columns = ["Cement","Slag","Ash","Water","Plasticizer","CoarseAgg","FineAgg","Age","CompressiveStrength"]
        df["CompressiveStrength"] = df["CompressiveStrength"]>= df["CompressiveStrength"].median()
        return np.array(df.iloc[:,:-1]),np.array(df["CompressiveStrength"]),df

    def splitData(self,randomSeed=47):
        #Split data into train,validation and test set
        X,y,df = self.loadData()
        X_train,X_temp,y_train,y_temp = train_test_split(X,y,train_size=0.8,random_state=randomSeed)
        X_validation,X_test,y_validation,y_test = train_test_split(X_temp,y_temp,train_size=0.5,random_state=randomSeed)
        X_train = self.standardize(X_train,useParams=False)
        X_validation = self.standardize(X_validation)
        X_test = self.standardize(X_test)
        return X_train,X_validation,X_test,y_train,y_validation,y_test

    def standardize(self,X,useParams=True):
        if not useParams:
            self.currentMean = np.mean(X,0)
            self.currentStd = np.std(X,0)
        return (X-self.currentMean)/(self.currentStd+0.00001)
