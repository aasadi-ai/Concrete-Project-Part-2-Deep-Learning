import torch

class TabularClassifier(torch.nn.Module):
    def __init__(self,l1=16,l2=32,l3=64):
        super(TabularClassifier,self).__init__()
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
    def forward(self,X):
        return self.layers(X)

