import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy import stats
from Utilities.dataformater import DataFormater

class Baseline():
    def __init__(self):
        pass

    def mode(self,X_train,y_train,X_validation):
        '''Mode'''
        modeTrain = stats.mode(y_train)[0][0]
        return np.full(len(X_validation),modeTrain)

    def prototype(self,X_train,y_train,X_validation):
        '''Closest Prototype'''
        postiveMask = y_train==1
        negativeMask = ~postiveMask
        
        postivePrototype = np.mean(X_train[postiveMask],0)
        negativePrototype = np.mean(X_train[negativeMask],0)
        
        def closestClass(x):
            distancePositive = np.sum(np.abs(x-postivePrototype))
            distanceNegative = np.sum(np.abs(x-negativePrototype))
            if distancePositive<distanceNegative:
                return 1
            return 0

        yHat = np.apply_along_axis(closestClass,1,X_validation)
        return yHat
    
    def logisticRegression(self,X_train,y_train,X_validation):
        '''Logistic Regression'''
        model = LogisticRegression()
        model.fit(X_train,y_train)
        return model.predict(X_validation)

    def basline(self):
        utilities = DataFormater()
        X_train,X_validation,X_test,y_train,y_validation,y_test = utilities.splitData()
        for base in [self.mode,self.prototype,self.logisticRegression]:
            print(base.__doc__)
            yHat = base(X_train,y_train,X_validation)
            print(accuracy_score(yHat,y_validation))
            print("---------------")
