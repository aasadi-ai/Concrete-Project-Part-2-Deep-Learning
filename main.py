from Utilities import Utils
from customDataLoaders import dataLoaderTabular
from Classifier import BinaryClassifier,train,accuracy
from display import displayLosses

def main():
    utilities = Utils()
    test = BinaryClassifier("tab")
    trainDataLoader,validationDataLoader,testDataLoader = dataLoaderTabular()
    trainLoss,valLoss = train(test,trainDataLoader,validationDataLoader,epochs=1000)
    print(accuracy(test,validationDataLoader))
    displayLosses(trainLoss,valLoss)

main()