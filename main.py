#%%
from Models.Baselines import Baseline
from sklearn import feature_selection
from Utilities.dataformater import DataFormater
from FeatureEngineering_Selection.featureEngineering import featureCreation
from FeatureEngineering_Selection.featureSelection import Selector
import numpy as np
#%%
selector = Selector()
formater = DataFormater()
X_train,X_validation,X_test,y_train,y_validation,y_test = formater.preProcessing(winsorize=False,standardize=True)
X_train = featureCreation(X_train)
X_validation = featureCreation(X_validation)
X_test = featureCreation(X_test)
#standardize before PCA
X_train = formater.standardize(X_train,useParams=False)
X_validation = formater.standardize(X_validation,useParams=True)
X_test = formater.standardize(X_test,useParams=True)
#apply PCA
X_train = selector.principleComponentAnalysis(X_train,useParams=False)
X_validation = selector.principleComponentAnalysis(X_validation)
X_test = selector.principleComponentAnalysis(X_test)
#standardize again for fast convergence and no exploding gradients
X_train = formater.standardize(X_train,useParams=False)
X_validation = formater.standardize(X_validation,useParams=True)
X_test = formater.standardize(X_test,useParams=True)

test = Baseline()
test.basline(X_train,X_validation,y_train,y_validation)


# %%
