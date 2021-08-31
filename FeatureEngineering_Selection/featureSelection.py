#%%
import sys
sys.path.append("C:\\Users\\afa30\\Desktop\\concreteNet")
from Models.Baselines import Baseline
from sklearn import feature_selection
from Utilities.dataformater import DataFormater
from FeatureEngineering_Selection.featureEngineering import featureCreation,featureCreation_All
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

class Selector():
    def __init__(self,numFeatures=5):
        self.featureSelector = feature_selection.RFE(estimator=DecisionTreeRegressor(),n_features_to_select=numFeatures)
        self.pca = PCA(n_components=numFeatures)

    def recursiveFeatureElimination(self,X,y,useParams=True):
        if not useParams:
            self.featureSelector.fit(X,y)
        bestFeatures = X.columns[self.featureSelector.support_].tolist()
        return X[bestFeatures]

    def rfe_All(self,X_data,y_data,useParams=False):
        if useParams:
            return [self.recursiveFeatureElimination(X_data[i],y_data[i],useParams=True) for i in range(len(X_data))]
        temp = self.recursiveFeatureElimination(X_data[0],y_data[0],useParams=False)
        return tuple([temp]+[self.recursiveFeatureElimination(X_data[i],y_data[i],useParams=True) for i in range(1,len(X_data))])

    def principleComponentAnalysis(self,X,useParams=True):
        if not useParams:
            self.pca.fit(X)
        transformedDf = self.pca.transform(X)
        return pd.DataFrame(data=transformedDf)

    def pca_All(self,*X_data,useParams=False):
        if useParams:
            return [self.principleComponentAnalysis(X,True) for X in X_data]
        temp = self.principleComponentAnalysis(X_data[0],False)
        return tuple([temp]+[self.principleComponentAnalysis(X,True) for X in X_data[1:]])

    def bestN(self,n,toNumpyFlag=False,seed=43):
        selector = Selector(n)
        formater = DataFormater()
        baslineModels = Baseline()
        #get data, standardize and remove outliers
        X_train,X_validation,X_test,y_train,y_validation,y_test = formater.preProcessing(winsorize=False,standardize=True,toNumpy=toNumpyFlag,seed=43)
        #generate novel features
        X_train,X_validation,X_test = featureCreation_All(X_train,X_validation,X_test)
        #standardize before PCA
        X_train,X_validation,X_test = formater.standardizeAll(X_train,X_validation,X_test,useParams=False)
        #apply PCA
        X_train,X_validation,X_test = selector.pca_All(X_train,X_validation,X_test,useParams=False)
        #standardize again for fast convergence and no exploding gradients
        X_train,X_validation,X_test = formater.standardizeAll(X_train,X_validation,X_test,useParams=False)
        return X_train,X_validation,X_test,y_train,y_validation,y_test


        
def tuneNumFeatures():
    formater = DataFormater()
    baslineModels = Baseline()
    #get data, standardize and remove outliers
    X_train,X_validation,X_test,y_train,y_validation,y_test = formater.preProcessing(winsorize=False,standardize=True,seed=43)
    #generate novel features
    X_train,X_validation,X_test = featureCreation_All(X_train,X_validation,X_test)
    #standardize before PCA
    X_train,X_validation,X_test = formater.standardizeAll(X_train,X_validation,X_test,useParams=False)
    for i in range(1,101):
        selector = Selector(i)
        #apply RFE
        #X_train,X_validation,X_test = selector.rfe_All([X_train,X_validation,X_test],[y_train,y_validation,y_test],useParams=False)
        #apply PCA
        X_train1,X_validation1,X_test1 = selector.pca_All(X_train,X_validation,X_test,useParams=False)
        #standardize again for fast convergence and no exploding gradients
        X_train1,X_validation1,X_test1 = formater.standardizeAll(X_train1,X_validation1,X_test1,useParams=False)
        print(f"Num Features:{i} Accuracy: {baslineModels.basline(X_train1,X_validation1,y_train,y_validation)[0]}")

