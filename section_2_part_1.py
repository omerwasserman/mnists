from mnist_dataset_class import MnistDataset
import data_utils
import torch
import numpy as np
import my_models
import torch.optim as optim
from print_loss import print_train_loss_graph, fix_loss_data

SECTION_PATH = 'Section 3/'
MODEL_PATH = '/Users/wasserman/Developer/mnists/models/' + SECTION_PATH
LOSS_PATH = '/Users/wasserman/Developer/mnists/loss_data/' + SECTION_PATH
MODEL_NAME = 'model_part_1.pt'
"""
This file trains a net over a regular 2 class mnist dataset.
"""


# Load data:
all_images_train, all_images_val, all_images_test,\
           all_labels_train, all_labels_val, all_labels_test = data_utils.load_processed_data_part_1(SECTION_PATH)

# compute mean and std of train set:
train_mean = np.mean(all_images_train, axis=(0, 1))

# Data is already combined so we can skip the next section
"""
val_data, val_labels = data_utils.combine_dataset(val_one_group_images, val_one_group_labels)
test_data, test_labels = data_utils.combine_dataset(test_one_group_images, test_one_group_labels)
"""

train_loader = data_utils.create_loader(all_images_train - train_mean, all_labels_train, batch_size=64)
val_loader = data_utils.create_loader(all_images_val - train_mean, all_labels_val, batch_size=64)
test_loader = data_utils.create_loader(all_images_test - train_mean, all_labels_test, batch_size=64)


# Check for GPU availability:
device = my_models.device_gpu_cpu()
print('using device:', device)

dtype = torch.float32  # we will be using float

train = True
test = True
model = None

if train:
    # Create models:
    model = my_models.model_2()
    my_models.test_model_size(model, dtype)  # test model size output:

    optimizer = optim.Adadelta(model.parameters())

    # Train model:
    model, loss_data = my_models.train_model(model, optimizer, train_loader, val_loader, device, dtype, epoches=2, print_every=5)

    # Save model to file:
    torch.save(model.state_dict(), MODEL_PATH + MODEL_NAME)

    # Save loss data to file:
    np.savetxt(LOSS_PATH + 'data_part_1_train.csv', loss_data, delimiter=',')
    fix_loss_data('data_part_1_train.csv', SECTION_PATH)
    print('Training is Finished !')
if test:
    print('Checking model Accuracy over Test Set...')
    if model is None:
        # load model:
        model = my_models.model_2()
        model.load_state_dict(torch.load(MODEL_PATH + MODEL_NAME))
    model.eval()

    # Check accuracy on test set:
    test_acc = my_models.check_accuracy(test_loader, model, device, dtype)

    # Saving acc to file
    acc_train_data = np.loadtxt(LOSS_PATH + 'data_part_1_train.csv', delimiter=',')
    acc_test_data = np.zeros((acc_train_data.shape[0], 7))
    acc_test_data[:, :6] = acc_train_data[:, :6]
    acc_test_data[:, 6] = test_acc
    np.savetxt(LOSS_PATH + 'data_part_1_train.csv', acc_test_data, delimiter=',')

    print_train_loss_graph('data_part_1_train.csv', SECTION_PATH)

print('Done!')