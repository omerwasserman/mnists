import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
import torch.optim as optim


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """

    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


def model_1():
    num_classes = 2

    model = nn.Sequential( nn.Conv2d(1, 32, kernel_size=3),     # This layer makes the images 26X26
                           nn.ReLU(),
                           nn.Conv2d(32, 64, kernel_size=3),    # This layer makes the images 24X24
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=2),         # This layer makes the images 12X12
                           nn.Dropout(0.25),
                           Flatten(),                           # flattening from 12X12X64
                           nn.Linear(12 * 12 * 64, 128),
                           nn.ReLU(),
                           nn.Linear(128, num_classes) )

    return model

def model_2():
    num_classes = 2

    model = nn.Sequential( nn.Conv2d(1, 32, kernel_size=3),     # This layer makes the images 26X26
                           nn.ReLU(),
                           nn.Conv2d(32, 64, kernel_size=3),    # This layer makes the images 24X24
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=2),         # This layer makes the images 12X12
                           nn.BatchNorm2d(num_features=64),
                           nn.Dropout(0.25),
                           nn.Conv2d(64, 128, kernel_size=3),   # This layer makes the images 10X10
                           nn.ReLU(),
                           nn.Conv2d(128, 128, kernel_size=3),  # This layer makes the images 8X8
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=2),         # This layer makes the images 4X4
                           nn.BatchNorm2d(num_features=128),
                           nn.Dropout(0.25),
                           Flatten(),  # flattening from 4X4X128
                           nn.Linear(4 * 4 * 128, 128),
                           nn.ReLU(),
                           nn.Linear(128, num_classes) )

    return model
def device_gpu_cpu():
    USE_GPU = True


    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device

def test_model_size(model, dtype):
    """
    This function tests the sizes of arrays in the model
    :param model: The model tested
    :return: Nothing
    """
    input = torch.zeros((64, 1, 28, 28), dtype=dtype)
    scores = model(input)
    assert scores.size() == torch.Size([64, 2]), 'Model size is NOT good'


def check_accuracy(loader, model, device, dtype):
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
    return acc


def train_model(model, optimizer, loader_train, loader_test, device, dtype, epoches=1, print_every=100):
    """
    This function train a model
    :param model: model to be trained
    :param optimizer: optimizer to be used
    :param epoches: number of epoches to train
    :return: last_model, [epoch, iteration, train_loss, net_accuracy]
    """

    PRINT_EVERY = print_every

    loss_data = []
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

            if t % PRINT_EVERY == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                train_acc = check_accuracy(loader_train, model, device, dtype)
                if isinstance(loader_test, tuple):
                    test_acc_1 = check_accuracy(loader_test[0], model, device, dtype)
                    test_acc_2 = check_accuracy(loader_test[1], model, device, dtype)
                    loss_data.append([e, t, (e + 1) * t, loss.item(), train_acc, test_acc_1, test_acc_2])
                else:
                    test_acc = check_accuracy(loader_test, model, device, dtype)
                    loss_data.append([e, t, (e + 1) * t, loss.item(), train_acc, test_acc])
                print()

    return model, loss_data

def train_model_5(model, optimizer, loader_train, loader_test, device, dtype, epoches=1, print_every=100):
    """
    This function train a model
    :param model: model to be trained
    :param optimizer: optimizer to be used
    :param epoches: number of epoches to train
    :return: last_model, [epoch, iteration, train_loss, net_accuracy]
    """

    assert isinstance(model, tuple) and len(model) == 2, 'model should be a tuple of length 2!'
    model1_dict = model[0].state_dict()
    model2_dict = model[1].state_dict()
    PRINT_EVERY = print_every

    loss_data = []
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

            if t % PRINT_EVERY == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                train_acc = check_accuracy(loader_train, model, device, dtype)
                if isinstance(loader_test, tuple):
                    test_acc_1 = check_accuracy(loader_test[0], model, device, dtype)
                    test_acc_2 = check_accuracy(loader_test[1], model, device, dtype)
                    loss_data.append([e, t, (e + 1) * t, loss.item(), train_acc, test_acc_1, test_acc_2])
                else:
                    test_acc = check_accuracy(loader_test, model, device, dtype)
                    loss_data.append([e, t, (e + 1) * t, loss.item(), train_acc, test_acc])
                print()

    return model, loss_data

