import sys
sys.path.append("C:\\Users\\afa30\\Desktop\\concreteNet")
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier
from prototypeEstimator import Prototype
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

    def knn(self,X_train,y_train,X_validation):
        '''KNN'''
        model = KNeighborsClassifier(n_neighbors=12)
        model.fit(X_train,y_train)
        return model.predict(X_validation)

    def gradientBoosting(self,X_train,y_train,X_validation):
        '''GradientBoosting'''
        model = GradientBoostingClassifier(n_estimators=250,max_depth=3)
        model.fit(X_train,y_train)
        return model.predict(X_validation)

    def ensemble(self,X_train,y_train,X_validation):
        '''Ensemble'''
        model = VotingClassifier([
            ("knn",KNeighborsClassifier(n_neighbors=12)),
            ("LogisticRegression",LogisticRegression()),
            ("proto",Prototype())
            ],voting='hard')
        model.fit(X_train,y_train)
        return model.predict(X_validation)

    def basline(self,X_train,X_validation,y_train,y_validation):
        for base in [self.mode,self.prototype,self.logisticRegression,self.knn,self.gradientBoosting,self.ensemble]:
            print(base.__doc__)
            yHat = base(X_train,y_train,X_validation)
            print(accuracy_score(yHat,y_validation))
            print("---------------")

test = Baseline()
utilities = DataFormater()
X_train,X_validation,X_test,y_train,y_validation,y_test = utilities.preProcessing(toNumpy=True)
test.basline(X_train,X_validation,y_train,y_validation)