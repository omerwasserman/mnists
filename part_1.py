from mnist_dataset_class import MnistDataset
import data_utils
import torch
import numpy as np
import my_models
import torch.optim as optim
from print_loss import print_train_loss_graph, fix_loss_data, fix_loss_array, print_loss_graph

SECTION_PATH = 'Section 3/'
MODEL_PATH = '/Users/wasserman/Developer/mnists/models/' + SECTION_PATH
LOSS_PATH = '/Users/wasserman/Developer/mnists/loss_data/' + SECTION_PATH
MODEL_NAME = 'model_part_1.pt'
"""
This file trains a net over a regular 2 class mnist dataset.
"""


# Load data:
data = data_utils.prepare_data_from_dict(1)


all_images_train = data['train_images']
all_images_test = data['test_images']
all_labels_train = data['train_labels']
all_labels_test = data['test_labels']

# compute mean and std of train set:
train_mean = np.mean(all_images_train, axis=(0, 1))


train_loader = data_utils.create_loader(all_images_train - train_mean, all_labels_train, batch_size=64)
test_loader = data_utils.create_loader(all_images_test - train_mean, all_labels_test, batch_size=64)


# Check for GPU availability:
device = my_models.device_gpu_cpu()
print('using device:', device)

dtype = torch.float32  # we will be using float

train = False
train_loops = 20

loss_data = []

if train:
    for tl in range(train_loops):
        # Create models:
        model = my_models.model_2()
        #my_models.test_model_size(model, dtype)  # test model size output:
        optimizer = optim.Adadelta(model.parameters())

        # Train model:
        model, current_loss_data = my_models.train_model(model, optimizer, train_loader, test_loader, device, dtype, epoches=2, print_every=5)
        loss_data.append(current_loss_data)
        # Save model to file:
        #torch.save(model.state_dict(), MODEL_PATH + MODEL_NAME)

        print('Training Loop {}/{} is Finished !'.format(tl + 1, train_loops))

    # Add test accuracy and save data to file:
    loss_data = np.asarray(loss_data)
    loss_data = fix_loss_array(loss_data)
    np.save(LOSS_PATH + 'data_part_1_acc.npy', loss_data)

    # Saving test acc to file
    """
    test_acc = np.asarray(test_acc)
    all_loss_data = np.zeros((loss_data.shape[0], loss_data.shape[1], loss_data.shape[2] + 1))
    all_loss_data[:, :, :-1] = loss_data
    all_loss_data[:, :, -1] = test_acc.reshape(test_acc.shape[0], 1)
    """


print_loss_graph('data_part_1_acc.npy', SECTION_PATH)

print('Done!')