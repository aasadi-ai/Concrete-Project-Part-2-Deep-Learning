def train_custom(config,checkpoint_dir=None,data_dir=None):
    model = TabularClassifier(config["l1"],config["l2"],config["l3"])
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=config["lr"],momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir,"checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    X_train,X_validation,X_test,y_train,y_validation,y_test = DataFormater().preProcessing()
    trainLoader,valLoader,testLoader = dataLoaderTabular(X_train,X_validation,X_test,y_train,y_validation,y_test)

    for epoch in range(100):
        #training
        for X,y in trainLoader:
            optimizer.zero_grad()
            yHat = model(X)
            loss = criterion(yHat,y)
            loss.backward()
            optimizer.step()
        #validation
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valLoader, 0):
            with torch.no_grad():
                X,y = data
                yHat = model(X)
                _, predicted = torch.max(yHat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                loss = criterion(yHat, y)
                val_loss += loss.numpy()
                val_steps += 1
        
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(
                        (model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")

def main():
    configCustom = {
        "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
        "l3": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
    }
    scheduler = ASHAScheduler(
        max_t=100,
        grace_period=1,
        reduction_factor=2
        )

    result = tune.run(
    tune.with_parameters(train_custom),
    resources_per_trial={"cpu": 1, "gpu": 0},
    config=configCustom,
    metric="loss",
    mode="min",
    num_samples=10,
    scheduler=scheduler
    )

main()


#%%
from sys import path
path.append("..")
from itertools import product
from Models.TabularClassifier import TabularClassifier
from Models.Classifier import train,accuracy
from Datasets_DataLoaders.customDataLoaders import dataLoaderTabular
from Utilities.dataformater import DataFormater

config = {
    "l1":[2**i for i in range(3,9)],
    "l2":[2**i for i in range(3,9)],
    "l3":[2**i for i in range(3,9)]
}
#%%
def tune(config,trainLoader,valLoader,testLoader,testEpochs=100):
    modelConfigsToTest = list(product(*[config[param] for param in config]))
    bestModel = TabularClassifier()
    bestParams = None
    _,_,bestModelLoss = train(bestModel,trainLoader,valLoader,epochs=testEpochs)
    for modelConfig in modelConfigsToTest:
        model = TabularClassifier(*modelConfig)
        model,trainLoss,valLoss = train(model,trainLoader,valLoader,epochs=testEpochs)
        if valLoss<=bestModelLoss:
            bestModel = model
            bestModelLoss = valLoss
            bestParams = modelConfig
    return bestModel,accuracy(bestModel,valLoader),bestParams

#%%
X_train,X_validation,X_test,y_train,y_validation,y_test = DataFormater().preProcessing(toNumpy=True)
trainLoader,valLoader,testLoader = dataLoaderTabular(X_train,X_validation,X_test,y_train,y_validation,y_test)
# model,performance,bestParams = tune(config,trainLoader,valLoader,testLoader)
# print(f"{performance}:{bestParams}")
model = TabularClassifier(32,32,128)
model,_,_ = train(model,trainLoader,valLoader,epochs=10000)
accuracy(model,valLoader)
# %%