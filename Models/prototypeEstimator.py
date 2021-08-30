import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
 
class Prototype(BaseEstimator,ClassifierMixin):
    '''Prototype Estimator Class'''
    def __init__(self):
        self.positivePrototype = None
        self.negativePrototype = None
        self.X_ = None
        self.y_ = None
    
    def fit(self,X,y):
        self.X_ = X
        self.y_ = y

        positiveMask = y==1
        negativeMask = ~positiveMask

        self.positivePrototype = np.mean(X[positiveMask],0)
        self.negativePrototype = np.mean(X[negativeMask],0)
        return self

    def closestClass(self,x):
        distancePositive = np.sum(np.abs(x-self.positivePrototype))
        distanceNegative = np.sum(np.abs(x-self.negativePrototype))
        if distancePositive<distanceNegative:
            return 1
        return 0

    def predict(self,X):
        yHat = np.apply_along_axis(self.closestClass,1,X)
        return yHat