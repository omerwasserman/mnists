import data_utils
import torch
import numpy as np
import my_models
import torch.optim as optim
from print_loss import print_train_loss_graph, fix_loss_data

MODEL_PATH = '/Users/wasserman/Developer/mnists/models/'
LOSS_PATH = '/Users/wasserman/Developer/mnists/loss_data/'
MODEL_NAME = 'model_part_3.pt'

"""
This file trains a net over a 2 class mnist dataset with 1 type of images - it's the same as part (1) but we're working
on half of the data.  
"""


# Load data:
images_REG_train, labels_REG_train,\
images_REG_val, labels_REG_val,\
images_REG_test, labels_REG_test = data_utils.load_processed_data_part_3()


# Show some images from train set
#data_utils.show_images(images_REG_train, 6)


# compute mean and std of train set:
train_mean = np.mean(images_REG_train, axis=(0, 1))

train_loader = data_utils.create_loader(images_REG_train - train_mean, labels_REG_train)
val_loader = data_utils.create_loader(images_REG_val - train_mean, labels_REG_val)
test_loader = data_utils.create_loader(images_REG_test - train_mean, labels_REG_test)


# Check for GPU availability:
device = my_models.device_gpu_cpu()
print('using device:', device)

dtype = torch.float32  # we will be using float

# Constant to control how frequently we print train loss
print_every = 100



train = True
test = True
model = None

if train:
    # Create models:
    model = my_models.model_2()
    my_models.test_model_size(model, dtype)  # test model size output:

    optimizer = optim.Adadelta(model.parameters())

    # Train model:
    model, loss_data = my_models.train_model(model, optimizer, train_loader, val_loader, device, dtype, epoches=2)

    # Save model to file:
    torch.save(model.state_dict(), MODEL_PATH + MODEL_NAME)

    # Save loss data to file:
    np.savetxt(LOSS_PATH + 'data_part_3_train.csv', loss_data, delimiter=',')
    fix_loss_data('data_part_3_train.csv')
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
    acc_train_data = np.loadtxt(LOSS_PATH + 'data_part_3_train.csv', delimiter=',')
    acc_test_data = np.zeros((acc_train_data.shape[0], 7))
    acc_test_data[:, :6] = acc_train_data[:, :6]
    acc_test_data[:, 6] = test_acc
    np.savetxt(LOSS_PATH + 'data_part_3_train.csv', acc_test_data, delimiter=',')


print_train_loss_graph('data_part_3_train.csv')

print('Done!')