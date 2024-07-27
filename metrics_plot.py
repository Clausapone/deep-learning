import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# function to visualize the confusion matrix
def show_confusion_matrix(conf_mat):
    conf_mat = np.array(conf_mat)

    plt.figure(figsize=(8, 6))
    plt.title('CONFUSION MATRIX')
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# function to visualize training loss over epochs
def show_loss_history(loss_history):
    plt.figure(figsize=(8, 6))
    plt.title('LOSS HISTORY')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss_history)
    plt.show()


# function to visualize metrics values
def show_metrics(accuracy, precision, recall, f1_score):
    plt.figure(figsize=(8, 4))
    _, ax = plt.subplots()
    text = (f"Accuracy: {100 * accuracy:.2f} %"
            f"\n\nPrecision: {100 * precision:.2f}%"
            f"\n\nRecall: {100 * recall:.2f}%"
            f"\n\nF1_Score: {100 * f1_score:.2f}%")
    ax.text(0, 0.5, text, fontsize=20, ha='left', va='center')
    ax.axis('off')

    plt.show()
