from sys import path
import os
path.append(os.path.dirname(os.curdir))
from Utilities.dataformater import DataFormater
import numpy as np
import pandas as pd

def featureCreation(X):
    epsilon = 0.0001
    oneVarFunctions = {"cbrt":np.cbrt,"sin":np.sin,"sqr":np.square}
    twoVarFunctions = {"sub":np.subtract,"prod":np.multiply,"div":np.divide}
    threeVarFunction = lambda x,y,z: np.square(x)+np.multiply(3,y)+z

    for funcName in oneVarFunctions.keys():
        for column in X.columns[:8]:
            for i in range(4):
                temp = X[column]+epsilon
                X[f"{i}_{funcName}-{column}"] = temp.apply(oneVarFunctions[funcName])

    for funcName in twoVarFunctions.keys():
        for column1 in X.columns[:8]:
            for column2 in X.columns[:8]:
                if column1!=column2:
                    X[f"{funcName}-{column1}&{column2}"] = twoVarFunctions[funcName](X[column1],X[column2]+epsilon)
    
    for column1 in X.columns[:8]:
       for column2 in X.columns[:8]:
           for column3 in X.columns[:8]:
               X[f"3_{column1}-{column2}-{column3}"] = threeVarFunction(X[column1],X[column2],X[column3])
    return X

def featureCreation_All(*X_data):
    return tuple([featureCreation(X) for X in X_data])

