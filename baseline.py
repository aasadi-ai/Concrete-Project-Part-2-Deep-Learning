import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats

def loadData():
    #Load Data and Rename Columns
    df = pd.read_csv("data/concreteData.csv")
    df.columns = ["Cement","Slag","Ash","Water","Plasticizer","CoarseAgg","FineAgg","Age","CompressiveStrength"]
    df["CompressiveStrength"] = df["CompressiveStrength"]>= df["CompressiveStrength"].median()
    return np.array(df.iloc[:,:-1]),np.array(df["CompressiveStrength"]),df

def splitData():
    #Split data into train,validation and test set
    X,y,df = loadData()
    X_train,X_temp,y_train,y_temp = train_test_split(X,y,train_size=0.8,random_state=43)
    X_validation,X_test,y_validation,y_test = train_test_split(X_temp,y_temp,train_size=0.5,random_state=43)
    return X_train,X_validation,X_test,y_train,y_validation,y_test

def mode(X_train,y_train,X_validation):
    '''Mode'''
    modeTrain = stats.mode(y_train)[0][0]
    return np.full(len(X_validation),modeTrain)

def prototype(X_train,y_train,X_validation):
    '''Closest Prototype'''
    postiveMask = y_train==1
    negativeMask = ~postiveMask
    
    postivePrototype = np.mean(X_train[postiveMask],0)
    negativePrototype = np.mean(X_train[negativeMask],0)
    
    def closestClass(x):
        distancePositive = np.sum(np.abs(x-postivePrototype))
        distanceNegative = np.sum(np.abs(x-negativePrototype))
        if distancePositive<distanceNegative:
            return 1
        return 0

    yHat = np.apply_along_axis(closestClass,1,X_validation)
    return yHat

def basline():
    X_train,X_validation,X_test,y_train,y_validation,y_test = splitData()
    for base in [mode,prototype]:
        print(base.__doc__)
        yHat = base(X_train,y_train,X_validation)
        print(accuracy_score(yHat,y_validation))
        print("---------------")

basline()
