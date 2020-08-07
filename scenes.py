import data_utils
import my_models
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
from torch.utils.data import DataLoader, sampler, TensorDataset
import numpy as np

import matplotlib.pyplot as plt





def check_accuracy(loader, model):
    """
       if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    """
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train_model(model, optimizer, loader_train, loader_val, epoches=1):
    """
    This function train a model
    :param model: model to be trained
    :param optimizer: optimizer to be used
    :param epoches: number of epoches to train
    :return:
    """

    model = model.to(device=device)
    for e in range(epoches):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device (GPU or CPU)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(loader_val, model)
                print()
    return model

def scene_1():


    NUM_TRAIN = train_one_group_images.shape[0]
    NUM_VAL = val_one_group_images.shape[0]

    train_mean = np.mean(train_one_group_images, axis=(0, 1))
    train_std = np.std(train_one_group_images, axis=(0, 1))
    all_train_images = train_one_group_images - train_mean
    all_val_images = val_one_group_images - train_mean

    # Turn train dataset into Tensor
    train_images_tensor = torch.Tensor(all_train_images)
    train_labels_tensor = torch.Tensor(train_one_group_labels)
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)

    # Turn val dataset into Tensor
    val_images_tensor = torch.Tensor(all_val_images)
    val_labels_tensor = torch.Tensor(val_one_group_labels)
    val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)

    loader_train = DataLoader(train_dataset, batch_size=128, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    loader_val = DataLoader(val_dataset, batch_size=128, sampler=sampler.SubsetRandomSampler(range(NUM_VAL)))
    """
    # Calculate mean of train images - to be subtracted from all data in order to center it.
    train_mean = np.mean(train_one_group_images, axis=(0, 1))
    # Subtract mean from data:
    train_one_group_images -= train_mean
    val_one_group_images   -= train_mean
    test_one_group_images  -= train_mean

    loader_train = data_utils.create_loader( (train_one_group_images), (train_one_group_labels) )
    loader_val = data_utils.create_loader( (val_one_group_images), (val_one_group_labels) )
    loader_test = data_utils.create_loader( (test_one_group_images), (test_one_group_labels) )
    """

    optimizer = optim.Adadelta(model.parameters())



    loader_train_2 = data_utils.create_loader(all_train_images, train_one_group_labels, (train_mean, train_std))

    best_model = train_model(model, optimizer, loader_train_2, loader_val, epoches=1)

    # Check accuracy over test set:
    # Turn test set into Tensor
    NUM_TEST = test_one_group_images.shape[0]
    all_test_images = test_one_group_images - train_mean
    test_images_tensor = torch.Tensor(all_test_images)
    test_labels_tensor = torch.Tensor(test_one_group_labels)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
    loader_test = DataLoader(test_dataset, batch_size=128, sampler=sampler.SubsetRandomSampler(range(NUM_TEST)))

    check_accuracy(loader_test, best_model)




def scene_2():
    # Combine REGULAR and INVERTED data into one dataset:
    all_train_images, all_train_labels = data_utils.combine_dataset( {0: regular_train_images, 1: inverted_train_images},
                                                                     {0: regular_train_labels, 1: inverted_train_labels} )
    all_val_images, all_val_labels = data_utils.combine_dataset( {0: regular_val_images, 1: inverted_val_images},
                                                                 {0: regular_val_labels, 1: regular_val_labels} )

    NUM_TRAIN = all_train_labels.shape[0]
    NUM_VAL = all_val_labels.shape[0]


    all_train_images = data_utils.subtract_mean(all_train_images)
    all_val_images = data_utils.subtract_mean(all_val_images)

    # Turn train dataset into Tensor
    train_images_tensor = torch.Tensor(all_train_images)
    train_labels_tensor = torch.Tensor(all_train_labels)
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)

    # Turn val dataset into Tensor
    val_images_tensor = torch.Tensor(all_val_images)
    val_labels_tensor = torch.Tensor(all_val_labels)
    val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)



    loader_train = DataLoader(train_dataset, batch_size=128, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    loader_val = DataLoader(val_dataset, batch_size=128, sampler=sampler.SubsetRandomSampler(range(NUM_VAL)))

    optimizer = optim.Adadelta(model.parameters())




    train_model(model, optimizer,loader_train, loader_val, epoches=2)


# Upload data to memory
#data_utils.proccess_data()
regular_train_images, regular_test_images, regular_val_images,\
regular_train_labels, regular_test_labels, regular_val_labels,\
inverted_train_images, inverted_test_images, inverted_val_images,\
inverted_train_labels, inverted_test_labels, inverted_val_labels,\
train_one_group_images, test_one_group_images, val_one_group_images,\
train_one_group_labels, test_one_group_labels, val_one_group_labels = data_utils.load_processed_data()


# Check for GPU availability:
device = my_models.device_gpu_cpu()
print('using device:', device)

dtype = torch.float32  # we will be using float

# Constant to control how frequently we print train loss
print_every = 100



# Create models:
model = my_models.model_2()
my_models.test_model_size(model, dtype)  # test model size output:


scene_1()

"""
# Ploting 2 demo images
plt.subplot(1,2,1)
plt.imshow(regular_train_images[0].reshape((28, 28)), cmap='gray')
plt.subplot(1,2,2)
plt.imshow(inverted_train_images[0].reshape((28, 28)), cmap='gray')
plt.show()
"""

print("c")