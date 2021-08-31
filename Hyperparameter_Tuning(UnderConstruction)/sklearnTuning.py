#%%
import sys
sys.path.append("C:\\Users\\afa30\\Desktop\\concreteNet")
from Models.Baselines import Baseline
from FeatureEngineering_Selection.featureSelection import Selector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#%%

def tuneKNN():
    X_train,X_validation,X_test,y_train,y_validation,y_test =Selector().bestN(17,seed=43)
    for neighbors in range(1,51):
        model = KNeighborsClassifier(n_neighbors=neighbors)
        model.fit(X_train,y_train)
        yHat = model.predict(X_validation)
        print(f"Neighbors({neighbors}):{accuracy_score(yHat,y_validation)}")

#%%
testBase = Baseline()
X_train,X_validation,X_test,y_train,y_validation,y_test =Selector().bestN(17,seed=41)
testBase.basline(X_train,X_validation,y_train,y_validation,True)

# %%
