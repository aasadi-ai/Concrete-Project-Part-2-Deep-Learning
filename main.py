#%%
from Models.Baselines import Baseline
from sklearn import feature_selection
from Utilities.dataformater import DataFormater
from FeatureEngineering_Selection.featureEngineering import featureCreation,featureCreation_All
from FeatureEngineering_Selection.featureSelection import Selector
import numpy as np
#%%
selector = Selector(4)
formater = DataFormater()

#get data, standardize and remove outliers
X_train,X_validation,X_test,y_train,y_validation,y_test = formater.preProcessing(winsorize=False,standardize=True)
#generate novel features
X_train,X_validation,X_test = featureCreation_All(X_train,X_validation,X_test)
#standardize before PCA
X_train,X_validation,X_test = formater.standardizeAll(X_train,X_validation,X_test,useParams=False)
#apply RFE
X_train,X_validation,X_test = selector.rfe_All([X_train,X_validation,X_test],[y_train,y_validation,y_test],useParams=False)
#apply PCA
X_train,X_validation,X_test = selector.pca_All(X_train,X_validation,X_test,useParams=False)
#standardize again for fast convergence and no exploding gradients
X_train,X_validation,X_test = formater.standardizeAll(X_train,X_validation,X_test,useParams=False)


test = Baseline()
test.basline(X_train,X_validation,y_train,y_validation)


# %%
