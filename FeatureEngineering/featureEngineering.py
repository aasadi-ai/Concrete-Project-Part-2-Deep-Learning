from Utilities.dataformater import DataFormater
import numpy as np
import pandas as pd
from PIL import Image

def featureCreation(df):
    utilities = DataFormater()
    _,_,df = utilities.loadData()
    X = df.iloc[:,:-1]
    y = df["CompressiveStrength"]
    epsilon = 0.00001
    oneVarFunctions = {"log":np.log,"sin":np.sin,"sqr":np.square}
    twoVarFunctions = {"sub":np.subtract,"prod":np.multiply,"div":np.divide}
    threeVarFunction = lambda x,y,z: np.square(x)+np.multiply(3,y)+z

    for funcName in oneVarFunctions.keys():
        for column in X.columns[:8]:
            for i in range(4):
                X[f"{i}_{funcName}-{column}"] = (X[column]+epsilon).apply(oneVarFunctions[funcName])

    for funcName in twoVarFunctions.keys():
        for column1 in X.columns[:8]:
            for column2 in X.columns[:8]:
                if column1!=column2:
                    X[f"{funcName}-{column1}&{column2}"] = twoVarFunctions[funcName](X[column1],X[column2]+epsilon)
    
    for column1 in X.columns[:8]:
       for column2 in X.columns[:8]:
           for column3 in X.columns[:8]:
               X[f"3_{column1}-{column2}-{column3}"] = threeVarFunction(X[column1],X[column2],X[column3])

    return X,y

def visualizeImg(img):
    #reshape row and scale image
    max = np.amax(img)
    min = np.amin(img)
    img = (img-min)/(max-min)
    img*=255
    return Image.fromarray(img)

def winsorizeOutlier(df):
    pass