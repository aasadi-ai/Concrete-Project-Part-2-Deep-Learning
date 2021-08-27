#%%
from Utilities import Utils
import numpy as np
from PIL import Image

def featureEngineering(df):
    X = df.iloc[:,:-1]
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

    return X

def getImg(row):
    max = np.amax(row)
    min = np.amin(row)
    rowScaled = (row-min)/(max-min)
    row = row *255
    img = np.reshape(row,(28,28))
    return Image.fromarray(img)


utilities = Utils()
X,y,df = utilities.loadData()
df = featureEngineering(df)
testRow = np.array(df.iloc[0])
img = getImg(testRow)
img.thumbnail((1000,1000))
img.show()

# %%
