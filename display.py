import matplotlib.pyplot as plt

def displayLosses(trainLoss,valLoss,epochs):
    plt.plot(epochs, trainLoss, 'r', label='Training Loss')
    plt.plot(epochs, valLoss, 'b', label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()