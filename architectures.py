import torch

tabularClassifier = torch.nn.Sequential(
            torch.nn.Linear(8,16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            torch.nn.Linear(16,32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            torch.nn.Linear(32,64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            torch.nn.Linear(64,32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            torch.nn.Linear(32,16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.01),
            torch.nn.Linear(16,1),
            torch.nn.Sigmoid()
        )

imageClassifier = torch.nn.Sequential(
    #1
    torch.nn.Conv2d(1,6,5),
    torch.nn.BatchNorm2d(6),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2,stride=2),
    #2
    torch.nn.Conv2d(6,16,5),
    torch.nn.BatchNorm2d(16),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2,stride=2),
    #3
    torch.nn.Conv2d(16,120,5),
    torch.nn.BatchNorm2d(120),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2,stride=2),
    #Flatten
    torch.nn.Flatten(),
    #Linear Layers
    torch.nn.Linear(32,64),
    torch.nn.BatchNorm1d(64),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.01),
    torch.nn.Linear(64,32),
    torch.nn.BatchNorm1d(32),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.01),
    torch.nn.Linear(32,16),
    torch.nn.BatchNorm1d(16),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.01),
    torch.nn.Linear(16,1),
    torch.nn.Sigmoid()
)