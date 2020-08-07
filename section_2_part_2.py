import data_utils
import torch
import numpy as np
import my_models
import torch.optim as optim
from print_loss import print_train_loss_graph, fix_loss_data

SECTION_PATH = 'Section 3/'
MODEL_PATH = '/Users/wasserman/Developer/mnists/models/' + SECTION_PATH
LOSS_PATH = '/Users/wasserman/Developer/mnists/loss_data/' + SECTION_PATH
MODEL_NAME = 'model_part_2.pt'

"""
This file trains a net over a 2 class mnist dataset with 2 types of images:
1. half of the images will be of type Regular and half of the images will be of type Inverted.
Testing will be done separately on the following 2 test-sets:
1. Only Regular images
2. Only Inverted images

Validation will be done on both types of images  
"""


# Load data:
images_train_set, labels_train_set, images_val_set, labels_val_set,\
images_REG_test, labels_REG_test, images_INV_test, labels_INV_test = data_utils.load_processed_data_part_2(SECTION_PATH)


# Show some images from train set
#data_utils.show_images(images_train_set, 6)


# compute mean and std of train set:
train_mean = np.mean(images_train_set, axis=(0, 1))

train_loader = data_utils.create_loader(images_train_set - train_mean, labels_train_set, batch_size=64)
val_loader = data_utils.create_loader(images_val_set - train_mean, labels_val_set, batch_size=64)
test_loader_REG = data_utils.create_loader(images_REG_test - train_mean, labels_REG_test, batch_size=64)
test_loader_INV = data_utils.create_loader(images_INV_test - train_mean, labels_INV_test, batch_size=64)


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
    np.savetxt(LOSS_PATH + 'data_part_2_train.csv', loss_data, delimiter=',')
    fix_loss_data('data_part_2_train.csv', SECTION_PATH)
    print('Training is Finished !')
if test:
    print('Checking model Accuracy over Test Set...')
    if model is None:
        # load model:
        model = my_models.model_2()
        model.load_state_dict(torch.load(MODEL_PATH + MODEL_NAME))
    model.eval()

    # Check accuracy on test set:
    test_REG_acc = my_models.check_accuracy(test_loader_REG, model, device, dtype)
    test_INV_acc = my_models.check_accuracy(test_loader_INV, model, device, dtype)

    # Saving acc to file
    acc_train_data = np.loadtxt(LOSS_PATH + 'data_part_2_train.csv', delimiter=',')
    acc_test_data = np.zeros((acc_train_data.shape[0], 8))
    acc_test_data[:, :6] = acc_train_data[:, :6]
    acc_test_data[:, 6] = test_REG_acc
    acc_test_data[:, 7] = test_INV_acc
    np.savetxt(LOSS_PATH + 'data_part_2_train.csv', acc_test_data, delimiter=',')


print_train_loss_graph('data_part_2_train.csv', SECTION_PATH)

print('Done!')