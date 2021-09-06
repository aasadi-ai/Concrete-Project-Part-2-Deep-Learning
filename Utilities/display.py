import matplotlib.pyplot as plt

def displayLosses(trainLoss,valLoss):
    plt.plot(range(len(trainLoss)), trainLoss, 'r', label='Training Loss')
    plt.plot(range(len(trainLoss)), valLoss, 'b', label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()