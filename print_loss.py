import numpy as np
from matplotlib import pyplot as plt

ACC_DATA_PATH = '/Users/wasserman/Developer/mnists/loss_data/'

"""
Data is stored in the following order:
Column No.:
0. Epochs
1. Iterations per current epoch
2. Iterations from beginung of training
3. Training loss
4. Training Accuracy during training.
5. Accuracy of model over test set
6. optional - same as (6) for a different test set
"""


def fix_loss_data(filename, PATH=''):
    PATH = ACC_DATA_PATH + PATH
    data = np.loadtxt(PATH + filename, delimiter=',')
    delta = data[1, 1]
    for row in range(1, data.shape[0]):
        data[row, 2] = data[row - 1, 2] + delta

    np.savetxt(PATH + filename, data, delimiter=',')

def fix_loss_array(data):
    delta = data[0, 1, 1]
    for row in range(1, data.shape[1]):
        data[:, row, 2] = data[:, row - 1, 2] + delta
    return data


def print_train_loss_graph(filename, PATH=''):
    PATH = ACC_DATA_PATH + PATH
    acc_data = np.loadtxt(PATH + filename, delimiter=',')
    plt.scatter(acc_data[:, 2], acc_data[:, 4], label='Train Accuracy')
    plt.plot(acc_data[:, 2], acc_data[:, 3], label='Train Loss')
    plt.plot(acc_data[:, 2], acc_data[:, 5], label='Test Accuracy')
    if acc_data.shape[1] == 6:
        plt.plot(acc_data[:, 2], acc_data[:, 6], label='Test Acc - No. 2')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy and Loss')
    plt.title('Accuracy and Loss Vs. Iterations')
    plt.show()

def print_loss_graph(filename, PATH=''):
    PATH = ACC_DATA_PATH + PATH
    data = np.load(PATH + filename)
    mean_data, max_data, min_data = data.mean(axis=0), data.max(axis=0), data.min(axis=0)

    fig = plt.figure(figsize=(6, 7))
    plt.subplot(2, 1, 1)
    plt.plot(mean_data[:, 2], mean_data[:, 3], label='Train Loss')
    plt.xlabel("Iterations")
    plt.ylabel('Mean Loss')
    plt.title('Mean Loss Vs. Iterations (20 Training loops)')
    plt.subplot(2, 1, 2)
    plt.plot(mean_data[:, 2], mean_data[:, 4] * 100, label='Train Accuracy')
    plt.plot(mean_data[:, 2], mean_data[:, 5] * 100, label='Test Accuracy')
    plt.errorbar(mean_data[:, 2],
                 mean_data[:, 5] * 100,
                 [(mean_data[:, 5] - min_data[:, 5]) * 100, (max_data[:, 5] - mean_data[:, 5]) * 100],
                 fmt='.k', ecolor='gray', lw=1, capsize=2)
    if data.shape[2] == 7:
        plt.plot(mean_data[:, 2], mean_data[:, 6] * 100, label='Test Accuracy 2')
        plt.errorbar(mean_data[:, 2],
                     mean_data[:, 6] * 100,
                     [(mean_data[:, 6] - min_data[:, 6]) * 100, (max_data[:, 6] - mean_data[:, 6]) * 100],
                     fmt='.k', ecolor='gray', lw=1, capsize=2)

    plt.xlabel("Iterations")
    plt.ylabel('Mean Accuracy [%]')
    plt.title('Mean Accuracy Vs. Iterations (20 Training loops)')
    plt.tight_layout()
    plt.legend()
    plt.show()
