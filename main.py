#%%
from sklearn import feature_selection
from Utilities.dataformater import DataFormater
from FeatureEngineering_Selection.featureEngineering import featureCreation
from FeatureEngineering_Selection.featureSelection import Selector
import numpy as np
#%%

X_train,X_validation,X_test,y_train,y_validation,y_test = DataFormater().preProcessing(standardize=False)
X_train = featureCreation(X_train)

selector = Selector()
formater = DataFormater()
#standardize before PCA
X_train = formater.standardize(X_train,useParams=False)
X_validation = formater.standardize(X_validation,useParams=True)
X_test = formater.standardize(X_test,useParams=True)
#apply PCA
X_train = selector.principleComponentAnalysis(X_train)
X_validation = selector.principleComponentAnalysis(X_validation)
X_test = selector.principleComponentAnalysis(X_test)
#standardize again for fast convergence and no exploding gradients
X_train = formater.standardize(X_train,useParams=False)
X_validation = formater.standardize(X_validation,useParams=True)
X_test = formater.standardize(X_test,useParams=True)


# %%
