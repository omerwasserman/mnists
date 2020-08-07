from mnist_dataset_class import MnistDataset
import data_utils

import torch

import numpy as np
import my_models
import torch.optim as optim

#from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T

model_path = '/Users/wasserman/Developer/mnists/models/'
loss_path = '/Users/wasserman/Developer/mnists/loss_data/'

"""
This file trains a net over a regular 2 class mnist dataset.
"""


# Load data:
all_images_train, all_images_val, all_images_test,\
           all_labels_train, all_labels_val, all_labels_test = data_utils.load_processed_data_part_1()

# compute mean and std of train set:
train_mean = np.mean(all_images_train, axis=(0, 1))

# Data is already combined so we can skip the next section
"""
val_data, val_labels = data_utils.combine_dataset(val_one_group_images, val_one_group_labels)
test_data, test_labels = data_utils.combine_dataset(test_one_group_images, test_one_group_labels)
"""

train_loader = data_utils.create_loader(all_images_train - train_mean, all_labels_train)
val_loader = data_utils.create_loader(all_images_val - train_mean, all_labels_val)
test_loader = data_utils.create_loader(all_images_test - train_mean, all_labels_test)


# Check for GPU availability:
device = my_models.device_gpu_cpu()
print('using device:', device)

dtype = torch.float32  # we will be using float

# Constant to control how frequently we print train loss
print_every = 100



train = False
test = True
model = None

if train:
    # Create models:
    model = my_models.model_2()
    my_models.test_model_size(model, dtype)  # test model size output:

    optimizer = optim.Adadelta(model.parameters())

    # Train model:
    model, loss_data = my_models.train_model(model, optimizer, train_loader, val_loader, device, dtype, epoches=1)

    # Save model to file:
    torch.save(model.state_dict(), model_path + 'model_part_1.pt')

    # Save loss data to file:
    np.savetxt(loss_path + 'data_part_1_train.csv', loss_data, delimiter=',')
if test:
    print('Checking model Accuracy over Test Set...')
    if model is None:
        # load model:
        model = my_models.model_2()
        model.load_state_dict(torch.load(model_path + 'model_part_1.pt'))
    model.eval()

    # Check accuracy on test set:
    test_acc = my_models.check_accuracy(test_loader, model, device, dtype)

    # Saving acc to file
    acc_train_data = np.loadtxt(loss_path + 'data_part_1_train.csv', delimiter=',')
    acc_test_data = np.zeros((acc_train_data.shape[0], 7))
    acc_test_data[:, :6] = acc_train_data[:, :6]
    acc_test_data[:, 6] = test_acc
    np.savetxt(loss_path + 'data_part_1_train.csv', acc_test_data, delimiter=',')

print('Done!')