import matplotlib.pyplot as plt

def plot_learning_curve(train_loss, valid_loss):
    plt.figure(figsize=(10,5))
    plt.plot(train_loss, label='Train')
    plt.plot(valid_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Learning curve")
    plt.legend()
    plt.show()

def plot_prediction(prediction, target):
    plt.figure(figsize=(10,5))
    plt.plot(prediction, label='Prediction')
    plt.plot(target, label='Ground Truth')
    plt.title("Prediction vs Ground Truth")
    plt.legend()
    plt.show()

def plot_residual(prediction, target):
    residual = target - prediction

    plt.figure(figsize=(10,5))
    plt.plot(residual, label='Residual')
    plt.axhline(y=0, linestyle='--', label='y = 0')
    plt.legend()
    plt.title("Residual Learning")
    plt.show()

