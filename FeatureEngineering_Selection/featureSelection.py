#Implement RFE
#Implement PCA
#Implement more sophisticated classifiers in Sklearn i.e. gradientboosting,KNN
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
    def __init__(self):
        self.featureSelector = feature_selection.RFE(estimator=DecisionTreeRegressor(),n_features_to_select=6)
        self.pca = PCA(n_components=6)

    def recursiveFeatureElimination(self,X,y,useParams=True,runs=10):
        #run 10 times and take best n
        if not useParams:
            self.featureSelector.fit(X,y)
        bestFeatures = X.columns[self.featureSelector.support_].tolist()
        return X[bestFeatures]

    def principleComponentAnalysis(self,X,useParams=True):
        if not useParams:
            self.pca.fit(X)
        transformedDf = self.pca.transform(X)
        return pd.DataFrame(data=transformedDf)
