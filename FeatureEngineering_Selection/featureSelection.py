#Then retest baselines with RFE,PCA,None and with Windsorize,Standardize,None 9 tests
#Settle on features for logistic regression NN
#Hyperparameter tuning of simple logistic regression NN (colab)
#Test CNN Classifier
#Try adding conv layer, try removing conv layer
#Hyperparameter tuning of CNN (colab)
#Notebook walkthrough with !pip install requirements.txt and path of all modules added
#Tuesday Morning: Presentation and GitHub readme.txt
from sklearn import feature_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
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