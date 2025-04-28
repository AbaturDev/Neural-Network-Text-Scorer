import matplotlib.pyplot as plt

def plot_visualizer(history, title="Model Training History"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Trening Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Trening Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{title} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_mse(mse_values, datasets):
    plt.figure(figsize=(8, 6))
    plt.bar(datasets, mse_values, color='skyblue')
    plt.xlabel('Dataset')
    plt.ylabel('Mean Squared Error')
    plt.title('Mean Squared Error on Test Sets')
    for i, v in enumerate(mse_values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()