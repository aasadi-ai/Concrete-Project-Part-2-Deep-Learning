from sys import path
path.append("..")
import os
import torch
from sklearn.metrics import accuracy_score

class TabularClassifier(torch.nn.Module):
    def __init__(self,l1=16,l2=32,l3=64):
        super().__init__()
        self.layers = torch.nn.Sequential(
            #1
            torch.nn.Linear(8,l1),
            torch.nn.BatchNorm1d(l1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            #2
            torch.nn.Linear(l1,l2),
            torch.nn.BatchNorm1d(l2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            #3
            torch.nn.Linear(l2,l3),
            torch.nn.BatchNorm1d(l3),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            #4
            torch.nn.Linear(l3,1),
            torch.nn.Sigmoid()
        )

def train_custom(config,checkpoint_dir=None,data_dir=None):
    model = TabularClassifier(config["l1"],config["l2"],config["l3"])
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=config["lr"],momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir,"checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    