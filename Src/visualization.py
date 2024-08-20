import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_results(train_losses, train_accuracies, num_epochs):
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(df):
    corr = df.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Feature Correlation Matrix')
    plt.show()
