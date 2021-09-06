from sys import path
import os
path.append(os.path.dirname(os.curdir))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier
from Models.prototypeEstimator import Prototype
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
        model = KNeighborsClassifier()
        model.fit(X_train,y_train)
        return model.predict(X_validation)

    def gradientBoosting(self,X_train,y_train,X_validation):
        '''GradientBoosting'''
        model = GradientBoostingClassifier()
        model.fit(X_train,y_train)
        return model.predict(X_validation)

    def ensemble(self,X_train,y_train,X_validation):
        '''Ensemble'''
        model = VotingClassifier([
            ("knn",KNeighborsClassifier()),
            ("LogisticRegression",LogisticRegression()),
            ("proto",Prototype()),
            ("grad",GradientBoostingClassifier())
            ],voting='hard')
        model.fit(X_train,y_train)
        return model.predict(X_validation)

    def basline(self,X_train,X_validation,y_train,y_validation,display=False):
        accuracies = []
        for base in [self.mode,self.prototype,self.logisticRegression,self.knn,self.gradientBoosting,self.ensemble]:
            yHat = base(X_train,y_train,X_validation)
            accuracy = accuracy_score(yHat,y_validation)
            accuracies.append(accuracy)
            if display:
                print(base.__doc__)
                print(accuracy)
                print("---------------")
        return max(accuracies),accuracies[-1],accuracies

X_train,X_validation,_,y_train,y_validation,_ = DataFormater().preProcessing(toNumpy=True)
Baseline().basline(X_train,X_validation,y_train,y_validation,display=True)