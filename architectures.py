import torch

SimpleClassifierArchitecture = torch.nn.Sequential(
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