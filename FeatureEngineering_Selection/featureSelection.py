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

def recursiveFeatureElimination(X,y,runs=10):
    #run 10 times and take best n
    featureSelector = feature_selection.RFE(estimator=DecisionTreeRegressor(),n_features_to_select=6)
    featureSelector.fit(X,y)
    bestFeatures = X.columns[featureSelector.support_].tolist()
    return X[bestFeatures]