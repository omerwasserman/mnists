import struct as st
import numpy as np
from random import sample
import collections
from torch.utils.data import DataLoader, sampler, TensorDataset
from torchvision import transforms as T
from mnist_dataset_class import MnistDataset
import matplotlib.pyplot as plt
import shelve

DATA_PATH_MNIST = '/Users/wasserman/Developer/DataSets/MNIST/'

def convert_files_to_npy():

    image_files = {'train_images': 'train-images.idx3-ubyte',
                   'test_images': 't10k-images.idx3-ubyte'}
    label_files = {'train_labels': 'train-labels.idx1-ubyte',
                   'test_labels': 't10k-labels.idx1-ubyte'}

    # Convert image files
    for name in image_files:
        imagesfile = open(DATA_PATH_MNIST + image_files[name], 'rb')

        # Read the magic number:
        imagesfile.seek(0)
        magic = st.unpack('>4B', imagesfile.read(4))

        # Read the dimensions of the file
        nImg = st.unpack('>I', imagesfile.read(4))[0] #num of images
        nR = st.unpack('>I', imagesfile.read(4))[0] #num of rows
        nC = st.unpack('>I', imagesfile.read(4))[0] #num of column

        # Reading the image data:
        nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
        images_array = np.zeros((nImg, nR, nC))
        images_array = 255 - np.asarray(st.unpack('>' +'B' * nBytesTotal, imagesfile.read(nBytesTotal))).reshape((nImg, 1, nR, nC))

        np.save(DATA_PATH_MNIST + name + '.npy', images_array)
        imagesfile.close()


    # Convert label files
    for name in label_files:
        labelsfile = open(DATA_PATH_MNIST + label_files[name], 'rb')

        # Read the magic number:
        labelsfile.seek(0)
        magic = st.unpack('>4B', labelsfile.read(4))

        # Read the dimensions of the file
        nImg = st.unpack('>I', labelsfile.read(4))[0] #num of labeled images

        # Reading the label data:
        nBytesTotal = nImg * 1  # since each pixel data is 1 byte
        labels_array = np.zeros((nImg))
        labels_array = np.asarray(st.unpack('>' + 'B' * nBytesTotal, labelsfile.read(nBytesTotal))).reshape(nImg)

        np.save(DATA_PATH_MNIST + name + '.npy', labels_array)
        labelsfile.close()

def load_data(PATH=''):

    """
    files = {'train_images': 'train_images.npy',
             'test_images': 'test_images.npy',
             'train_labels': 'train_labels.npy',
             'test_labels': 'test_labels.npy'}

    train_images = np.load(DATA_PATH_MNIST + files['train_images'])
    test_images = np.load(DATA_PATH_MNIST + files['test_images'])
    train_labels = np.load(DATA_PATH_MNIST + files['train_labels'])
    test_labels = np.load(DATA_PATH_MNIST + files['test_labels'])

    return train_images, train_labels, test_images, test_labels
    """

    files = {'train_images': 'train_images.npy',
             'train_labels': 'train_labels.npy'}

    train_images = np.load(DATA_PATH_MNIST + PATH + files['train_images'])
    train_labels = np.load(DATA_PATH_MNIST + PATH + files['train_labels'])

    return train_images, train_labels

def divide_dataset(images, labels, groups=2, seed=False):
    """
    This function divides a dataset to groups
    Inputs:
    images: Contains the image data set. It's of size N, C, H, W
    labels: Contains the labels for the dataset. It's of size N
    groups: If groups > 1 than groups is the number of groups to divide to. If groups is a list than the list contains
            the numbers smaller than 1 whos sum equals 1 which represent the portion of samples in each group
    seed: if seed is not False than a seed would be generated to remove randomness is sampling
    Outputs:
    images_dict: A dictionary containing the images in their groups
    labels_dict: A dictionary containing the labels in their groups
    """

    if seed:
        np.random.seed(1)

    N, C, H, W = images.shape
    images_dict = {}
    labels_dict = {}
    all_indices = sample(range(N), N)

    if isinstance(groups, int) and groups > 1:
        assert N / groups != 0, "Number of Samples do not divide by number of groups"


        idx_per_group = N // groups

        for group in range(groups):
            images_dict[group] = images[all_indices[idx_per_group * group:idx_per_group * (group + 1)], :, :, :]
            labels_dict[group] = labels[all_indices[idx_per_group * group:idx_per_group * (group + 1)]]
    elif isinstance(groups, list):
        assert sum(groups) > 0.98 or sum(groups) < 1, "Sum of group percentages is not 100%"
        last_idx = 0
        for idx, percent in enumerate(groups[:-1]):
            element_num = int(round(N * percent))
            next_idx = last_idx + element_num
            images_dict[idx] = images[all_indices[last_idx:next_idx], :, :, :]
            labels_dict[idx] = labels[all_indices[last_idx:next_idx]]
            last_idx = next_idx
        images_dict[idx + 1] = images[all_indices[last_idx:], :, :, :]
        labels_dict[idx + 1] = labels[all_indices[last_idx:]]

    return images_dict, labels_dict

def image_invert(image):
    """
    This function invert the colors of an images
    INPUT:
    image: a tuple of size (N, 1, H ,W) containing N images
    OUTPUT:
    image_back: a tuple of the same size with the inverted images
    """

    return 255 - image

def convert_classes(labels, in_classes, out_classes, random=False, seed=False):
    """
    This function changes the number of classes of the labels from in_classes to out_classes
    INPUT:
    labels: Original labels
    in_classes: Number of classes in original data
    out_classes: Number of classes in returned labels
    random: If true than the out classes will be samples randomly
    seed: If True than seed will not be set
    :return:
    labels_out: returned labels
    """
    labels_out = None

    assert in_classes % out_classes == 0, "in_classes does not divide by out_classes"
    elements_in_group = in_classes // out_classes
    if seed:
        np.random.seed(1)

    # Decide which original classes will go to new classes:
    classes = []
    if random:
        all_classes = set(range(in_classes))
        for i in range(out_classes):
            classes.append(sample(all_classes, elements_in_group))
            all_classes -= set(classes[i])
    else:
        all_classes = list(range(in_classes))
        for i in range(out_classes):
            classes.append(all_classes[i * elements_in_group : elements_in_group * (i + 1)])

    # Change labels of original classes to new classes:

    for new_class in range(out_classes):
        for old_class in classes[new_class]:
            labels[labels == old_class] = new_class

    return labels

def group_stats(labels, verbose=False):
    """
    This function checks how many labels from each class is represented in that group
    INPUTS:
    labels: A tuple of size (G,N) containing G groups with k possible lables
    verbose: if True than print percentage on terminal
    OUTPUTS:
    stats: an array containing the relative part (percent) of each class in each group

    """

    groups = len(labels)
    all_classes = set(np.concatenate(labels, axis=0))

    stats = np.zeros([groups, len(all_classes)])  # Initialize the output

    # Count how many from each class is present in each group
    for group in range(groups):
        counts = collections.Counter(labels[group])
        for key in counts:
            stats[group, key] = counts[key] / labels[group].shape[0]
            if verbose:
                print('Group {}: Percentage of Class {} is {:.2f}'.format(group, key, stats[group, key] * 100))

    return stats

def combine_dataset(images, labels):
    """
    This function gets 2 tuples. One that is filled with numpy arrarys of images and another with
    their respective labels.
    The function combines all datasets into one.
    The function returns 2 numpy arrays
    :param images: Tuple with data
    :param labels: Tuple with labels
    :return:
    images_one: A numpy array with all images.
    labels_one: A numpy array with all labels.
    """

    # At first we'll count the total number of samples in all dataset in order to be able to instantiate
    # images_one and labels_one

    if type(images) is tuple:
        images_one = images[0]
        labels_one = labels[0]
        for idx in range(1, len(images)):
            images_one = np.vstack((images_one, images[idx]))
            labels_one = np.hstack((labels_one, labels[idx]))
    else:
        return images, labels

    return images_one, labels_one

def create_loader(image_tuple, label_tuple, batch_size=128):
    """
    This function creates a data loader
    :param image_tuple:
    :param label_tuple:
    :param stat_tuple:
    :return:
    """

    # Combine all sets:
    all_data, all_labels = combine_dataset(image_tuple, label_tuple)

    # Create DataSet

    #transform = T.Compose([T.ToTensor()])
    transform = T.ToTensor()
    #transform = None
    dataset = MnistDataset(all_data, all_labels, transform=transform)

    data_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=sampler.SubsetRandomSampler(range(dataset.shape()[0])))

    return data_loader


def process_data(num_classes=10, PATH=''):
    """
    This function does the following procedures on the data:
    1. Opens the training set and labels from .npy files
    2. changes their labels such that we'll have only 2 classes
    3. divides the data set into 2 groups
    4. Inverts the colors of one of the groups
    5. divide each group into train set (70%) and test set (30%)
    6. saves back the data into .npy files ( total of 4 data files and 4 label files)
    :return: None
    """
    # Load train data from .npy files
    train_images, train_labels = load_data(PATH)

    # Change labels such that we'll have only 2 classes to be classified (we'll do this over all the training set):
    new_labels = convert_classes(train_labels, num_classes, 2, random=True)

    # Divide training dataset into 2 groups
    images, labels = divide_dataset(train_images, new_labels)

    # Invert colors of first group (images[0]) to make them belong to the "Inverse" group:
    image_inv = image_invert(images[0])

    # Let's check the statistics of the regular and inverted groups to check how many of each class is present in the group:
    stats = group_stats([labels[0], labels[1]], verbose=True)

    # From each group take 15% for test and 15% for validation set:
    regular_images, regular_labels = divide_dataset(images[1], labels[1], [0.7, 0.15, 0.15])
    inverted_images, inverted_labels = divide_dataset(image_inv, labels[0], [0.7, 0.15, 0.15])


    np.save(DATA_PATH_MNIST + PATH + 'train images-regular' + '.npy', regular_images[0])  # Train images regular (first group)
    np.save(DATA_PATH_MNIST + PATH + 'val images-regular' + '.npy', regular_images[1])    # Val images regular (second group)
    np.save(DATA_PATH_MNIST + PATH + 'test images-regular' + '.npy', regular_images[2])   # Test images regular (third group)

    np.save(DATA_PATH_MNIST + PATH + 'train images-inverted' + '.npy', inverted_images[0])  # Train images inverted (first group)
    np.save(DATA_PATH_MNIST + PATH + 'val images-inverted' + '.npy', inverted_images[1])    # Val images inverted (second group)
    np.save(DATA_PATH_MNIST + PATH + 'test images-inverted' + '.npy', inverted_images[2])   # Test images inverted (third group)


    np.save(DATA_PATH_MNIST + PATH + 'train labels-regular' + '.npy', regular_labels[0])  # Train labels regular (first group)
    np.save(DATA_PATH_MNIST + PATH + 'val labels-regular' + '.npy', regular_labels[1])    # Val labels regular (second group)
    np.save(DATA_PATH_MNIST + PATH + 'test labels-regular' + '.npy', regular_labels[2])   # Test labels regular (third group)


    np.save(DATA_PATH_MNIST + PATH + 'train labels-inverted' + '.npy', inverted_labels[0])  # Train labels inverted (first group)
    np.save(DATA_PATH_MNIST + PATH + 'val labels-inverted' + '.npy', inverted_labels[1])    # Val labels inverted (second group)
    np.save(DATA_PATH_MNIST + PATH + 'test labels-inverted' + '.npy', inverted_labels[2])   # Test labels inverted (third group)


    # Create Regular dataset from all original training images
    train_images_all, train_labels_all = divide_dataset(train_images, new_labels, [0.7, 0.15, 0.15])

    np.save(DATA_PATH_MNIST + PATH + 'all_same_train_images' + '.npy', train_images_all[0])  # Train images all regular (first group)
    np.save(DATA_PATH_MNIST + PATH + 'all_same_val_images' + '.npy', train_images_all[1])    # Val images all regular (second group)
    np.save(DATA_PATH_MNIST + PATH + 'all_same_test_images' + '.npy', train_images_all[2])   # Test images all regular (third group)

    np.save(DATA_PATH_MNIST + PATH + 'all_same_train_labels' + '.npy', train_labels_all[0])  # Train labels all regular (first group)
    np.save(DATA_PATH_MNIST + PATH + 'all_same_val_labels' + '.npy', train_labels_all[1])    # Val labels all regular (second group)
    np.save(DATA_PATH_MNIST + PATH + 'all_same_test_labels' + '.npy', train_labels_all[2])   # Test labels all regular (third group)



def load_processed_data():
    files = {'train images-regular': 'train images-regular.npy',
             'test images-regular': 'test images-regular.npy',
             'val images-regular': 'val images-regular.npy',
             'train images-inverted': 'train images-inverted.npy',
             'test images-inverted': 'test images-inverted.npy',
             'val images-inverted': 'test images-inverted.npy',
             'train labels-regular': 'train labels-regular.npy',
             'test labels-regular': 'test labels-regular.npy',
             'val labels-regular': 'val labels-regular.npy',
             'train labels-inverted': 'train labels-inverted.npy',
             'test labels-inverted': 'test labels-inverted.npy',
             'val labels-inverted': 'val labels-inverted.npy',
             'all_same_train_images': 'all_same_train_images.npy',
             'all_same_test_images': 'all_same_test_images.npy',
             'all_same_val_images': 'all_same_val_images.npy',
             'all_same_train_labels': 'all_same_train_labels.npy',
             'all_same_test_labels': 'all_same_test_labels.npy',
             'all_same_val_labels': 'all_same_val_labels.npy'}


    regular_train_images = np.load(DATA_PATH_MNIST + files['train images-regular'])
    regular_test_images = np.load(DATA_PATH_MNIST + files['test images-regular'])
    regular_val_images = np.load(DATA_PATH_MNIST + files['val images-regular'])
    regular_train_labels = np.load(DATA_PATH_MNIST + files['train labels-regular'])
    regular_test_labels = np.load(DATA_PATH_MNIST + files['test labels-regular'])
    regular_val_labels = np.load(DATA_PATH_MNIST + files['val labels-regular'])

    inverted_train_images = np.load(DATA_PATH_MNIST + files['train images-inverted'])
    inverted_test_images = np.load(DATA_PATH_MNIST + files['test images-inverted'])
    inverted_val_images = np.load(DATA_PATH_MNIST + files['val images-inverted'])
    inverted_train_labels = np.load(DATA_PATH_MNIST + files['train labels-inverted'])
    inverted_test_labels = np.load(DATA_PATH_MNIST + files['test labels-inverted'])
    inverted_val_labels = np.load(DATA_PATH_MNIST + files['val labels-inverted'])

    train_one_group_images = np.load(DATA_PATH_MNIST + files['all_same_train_images'])
    test_one_group_images = np.load(DATA_PATH_MNIST + files['all_same_test_images'])
    val_one_group_images = np.load(DATA_PATH_MNIST + files['all_same_val_images'])
    train_one_group_labels = np.load(DATA_PATH_MNIST + files['all_same_train_labels'])
    test_one_group_labels = np.load(DATA_PATH_MNIST + files['all_same_test_labels'])
    val_one_group_labels = np.load(DATA_PATH_MNIST + files['all_same_val_labels'])

    return regular_train_images, regular_test_images, regular_val_images,\
           regular_train_labels, regular_test_labels, regular_val_labels,\
           inverted_train_images, inverted_test_images, inverted_val_images,\
           inverted_train_labels, inverted_test_labels, inverted_val_labels,\
           train_one_group_images, test_one_group_images, val_one_group_images,\
           train_one_group_labels, test_one_group_labels, val_one_group_labels

def load_processed_data_part_1(SECTION_PATH=''):

    all_images_train = np.load(DATA_PATH_MNIST + SECTION_PATH + 'all_same_train_images.npy')
    all_labels_train = np.load(DATA_PATH_MNIST + SECTION_PATH + 'all_same_train_labels.npy')

    all_images_val = np.load(DATA_PATH_MNIST + SECTION_PATH + 'all_same_val_images.npy')
    all_labels_val = np.load(DATA_PATH_MNIST + SECTION_PATH + 'all_same_val_labels.npy')

    all_images_test = np.load(DATA_PATH_MNIST + SECTION_PATH + 'all_same_test_images.npy')
    all_labels_test = np.load(DATA_PATH_MNIST + SECTION_PATH + 'all_same_test_labels.npy')

    return all_images_train, all_images_val, all_images_test,\
           all_labels_train, all_labels_val, all_labels_test

def load_processed_data_part_2(SECTION_PATH=''):


    images_REG_train = np.load(DATA_PATH_MNIST + SECTION_PATH + 'train images-regular.npy')
    images_REG_val = np.load(DATA_PATH_MNIST + SECTION_PATH + 'val images-regular.npy')
    images_REG_test = np.load(DATA_PATH_MNIST + SECTION_PATH + 'test images-regular.npy')

    labels_REG_train = np.load(DATA_PATH_MNIST + SECTION_PATH + 'train labels-regular.npy')
    labels_REG_val = np.load(DATA_PATH_MNIST + SECTION_PATH + 'val labels-regular.npy')
    labels_REG_test = np.load(DATA_PATH_MNIST + SECTION_PATH + 'test labels-regular.npy')

    images_INV_train = np.load(DATA_PATH_MNIST + SECTION_PATH + 'train images-inverted.npy')
    images_INV_val = np.load(DATA_PATH_MNIST + SECTION_PATH + 'val images-inverted.npy')
    images_INV_test = np.load(DATA_PATH_MNIST + SECTION_PATH + 'test images-inverted.npy')

    labels_INV_train = np.load(DATA_PATH_MNIST + SECTION_PATH + 'train labels-inverted.npy')
    labels_INV_val = np.load(DATA_PATH_MNIST + SECTION_PATH + 'val labels-inverted.npy')
    labels_INV_test = np.load(DATA_PATH_MNIST + SECTION_PATH + 'test labels-inverted.npy')

    # Section 2 requires training and validation over train set of REG + INV
    # testing will be done separately on test_REG and test_INV

    # combining training sets:
    images_train_set, labels_train_set = combine_dataset( (images_REG_train, images_INV_train), (labels_REG_train, labels_INV_train) )

    # combining validation sets:
    images_val_set, labels_val_set = combine_dataset( (images_REG_val, images_INV_val), (labels_REG_val, labels_INV_val) )

    return images_train_set, labels_train_set, images_val_set, labels_val_set,\
           images_REG_test, labels_REG_test, images_INV_test, labels_INV_test

def load_processed_data_part_3(SECTION_PATH=''):


    images_REG_train = np.load(DATA_PATH_MNIST + SECTION_PATH + 'train images-regular.npy')
    images_REG_val = np.load(DATA_PATH_MNIST + SECTION_PATH + 'val images-regular.npy')
    images_REG_test = np.load(DATA_PATH_MNIST + SECTION_PATH + 'test images-regular.npy')

    labels_REG_train = np.load(DATA_PATH_MNIST + SECTION_PATH + 'train labels-regular.npy')
    labels_REG_val = np.load(DATA_PATH_MNIST + SECTION_PATH + 'val labels-regular.npy')
    labels_REG_test = np.load(DATA_PATH_MNIST + SECTION_PATH + 'test labels-regular.npy')

    return images_REG_train, labels_REG_train, images_REG_val, labels_REG_val, images_REG_test, labels_REG_test

def load_processed_data_part_4(SECTION_PATH):


    images_REG_train = np.load(DATA_PATH_MNIST + SECTION_PATH + 'train images-regular.npy')
    images_REG_val = np.load(DATA_PATH_MNIST + SECTION_PATH + 'val images-regular.npy')
    images_REG_test = np.load(DATA_PATH_MNIST + SECTION_PATH + 'test images-regular.npy')

    labels_REG_train = np.load(DATA_PATH_MNIST + SECTION_PATH + 'train labels-regular.npy')
    labels_REG_val = np.load(DATA_PATH_MNIST + SECTION_PATH + 'val labels-regular.npy')
    labels_REG_test = np.load(DATA_PATH_MNIST + SECTION_PATH + 'test labels-regular.npy')

    images_INV_train = np.load(DATA_PATH_MNIST + SECTION_PATH + 'train images-inverted.npy')
    images_INV_val = np.load(DATA_PATH_MNIST + SECTION_PATH + 'val images-inverted.npy')
    images_INV_test = np.load(DATA_PATH_MNIST + SECTION_PATH + 'test images-inverted.npy')

    labels_INV_train = np.load(DATA_PATH_MNIST + SECTION_PATH + 'train labels-inverted.npy')
    labels_INV_val = np.load(DATA_PATH_MNIST + SECTION_PATH + 'val labels-inverted.npy')
    labels_INV_test = np.load(DATA_PATH_MNIST + SECTION_PATH + 'test labels-inverted.npy')

    # Section 4 requires training and validation over train set of INV
    # testing will be done on test_REG.

    return images_INV_train, labels_INV_train, images_INV_val, labels_INV_val, images_REG_test, labels_REG_test



def show_images(images, num):
    """
    This function show randomly selected images from dataset
    :param images: image data set
    :param num: number of images to show
    :return:
    """

    N = images.shape[0]
    for _ in range(num):
        idx = np.random.randint(0, N, 1)
        plt.imshow(images[idx].reshape(28, 28), cmap='gray')
        plt.show()

def data_for_section_2(num_classes=2, train_num=1000, val_num=200, test_num=200, SECTION_PATH=None):
    """
    Section 2 requires smaller training data:
    train = train_num (2000) samples
    val = val_num (300) samples
    test = test_num (300) samples

    :return:
    """

    if SECTION_PATH is None:
        SECTION_2_PATH = 'Section 2/'
    else:
        SECTION_2_PATH = SECTION_PATH

    train_images, train_labels = load_data()

    # Count how many original labels from each class:
    counts = collections.Counter(train_labels)

    # Original class 3 will become new class 0
    # Original class 8 will become new class 1

    idx_class_0 = train_labels == 3
    idx_class_1 = train_labels == 8

    all_images_0 = train_images[idx_class_0]
    all_images_1 = train_images[idx_class_1]

    # Number of images from each class
    N_0 = all_images_0.shape[0]
    N_1 = all_images_1.shape[0]

    # From each class we need a total of (train_num + val_num + test_num)
    images_0 = all_images_0[sample(range(N_0), train_num + val_num + test_num)]
    images_1 = all_images_1[sample(range(N_1), train_num + val_num + test_num)]

    # Now we'll create labels for them:
    labels_0 = np.zeros(train_num + val_num + test_num, dtype=np.uint8)
    labels_1 = np.ones(train_num + val_num + test_num, dtype=np.uint8)

    # Combining dataset:
    train_images, train_labels = combine_dataset( (images_0, images_1), (labels_0, labels_1))

    # Saving the data
    np.save(DATA_PATH_MNIST + SECTION_2_PATH + 'train_images.npy', train_images)
    np.save(DATA_PATH_MNIST + SECTION_2_PATH + 'train_labels.npy', train_labels)

    # Continue from here with function process_data:
    process_data(num_classes, SECTION_2_PATH)

def data_for_section_3():
    data_for_section_2(num_classes=2, train_num=500, val_num=100, test_num=100, SECTION_PATH='Section 3/')

def shelve_data():
    filename = 'data_dictionary.npy'

    # Uploading original train data:
    train_images, train_labels = load_data()

    data_dict = {}

    # Going over the train data and dividing it by labels:
    for label in range(10):
        idx = train_labels == label
        images = train_images[idx]
        data_dict[label] = images

    np.save(DATA_PATH_MNIST + filename, data_dict)
    print('Saved Dictionary data file at location: {}'. format(DATA_PATH_MNIST + filename))

def prepare_data_from_dict(part, train_samples=500, filname='data_dictionary.npy', random=False):
    """
    This function prepares data for training and testing.

    :param part: part from mission (options 1-6)
    :param random: if True than each call to the function will choose different numbers
    :return:

    """

    data_dict = np.load(DATA_PATH_MNIST + filname, allow_pickle=True).item()

    N_label = train_samples // 2  # Number of samples to take from each class/label

    if random:  # We need to randomly pick 2 labels (classes)
        picked_labels = sample(range(10), 2)
    else:  # Else I'll choose in advance 3 and 8 as the controlling classes
        picked_labels = (3, 8)

    all_images_0 = data_dict[picked_labels[0]]
    all_images_1 = data_dict[picked_labels[1]]

    # Pick train_samples/2 from each class:
    idx_0 = sample(range(all_images_0.shape[0]), all_images_0.shape[0])
    idx_1 = sample(range(all_images_1.shape[0]), all_images_1.shape[0])

    train_images_0 = all_images_0[idx_0[:N_label]]
    train_images_1 = all_images_1[idx_1[:N_label]]

    test_images_0 = all_images_0[idx_0[N_label:]]
    test_images_1 = all_images_1[idx_1[N_label:]]

    if part == 1:
        images_train = np.vstack((train_images_0, train_images_1))
        labels_train = np.hstack((np.zeros(N_label, dtype=np.uint8), np.ones(N_label, dtype=np.uint8)))

        images_test = np.vstack((test_images_0, test_images_1))
        labels_test_0 = np.zeros(test_images_0.shape[0], dtype=np.uint8)
        labels_test_1 = np.ones(test_images_1.shape[0], dtype=np.uint8)
        labels_test = np.hstack((labels_test_0, labels_test_1))

    elif part == 2 or part == 5 or part == 6:
        # Half of the images from train/test should be inverted
        N_inv = N_label // 2
        idx = sample(range(N_label), N_label)

        train_images_0_REG = train_images_0[idx[:N_inv]]
        train_images_0_INV = train_images_0[idx[N_inv:]]
        train_images_0_INV = image_invert(train_images_0_INV)

        train_images_1_REG = train_images_1[idx[:N_inv]]
        train_images_1_INV = train_images_1[idx[N_inv:]]
        train_images_1_INV = image_invert(train_images_1_INV)

        # Now, we'll produce the test images:
        test_images_0 = all_images_0[idx_0[N_label:]]
        test_images_1 = all_images_1[idx_1[N_label:]]

        # Invert half of the images from each group:
        N_test_0 = test_images_0.shape[0]
        N_test_1 = test_images_1.shape[0]

        idx_0 = sample(range(N_test_0), N_test_0)
        test_images_0_REG = test_images_0[idx_0[:N_test_0 // 2]]
        test_images_0_INV = test_images_0[idx_0[N_test_0 // 2:]]
        test_images_0_INV = image_invert(test_images_0_INV)
        labels_test_0_REG = np.zeros(test_images_0_REG.shape[0], dtype=np.uint8)
        labels_test_0_INV = np.zeros(test_images_0_INV.shape[0], dtype=np.uint8)

        idx_1 = sample(range(N_test_1), N_test_1)
        test_images_1_REG = test_images_1[idx_1[:N_test_1 // 2]]
        test_images_1_INV = test_images_1[idx_1[N_test_1 // 2:]]
        test_images_1_INV = image_invert(test_images_1_INV)
        labels_test_1_REG = np.ones(test_images_1_REG.shape[0], dtype=np.uint8)
        labels_test_1_INV = np.ones(test_images_1_INV.shape[0], dtype=np.uint8)

        images_test = (np.vstack((test_images_0_REG, test_images_1_REG)), np.vstack((test_images_0_INV, test_images_1_INV)))
        labels_test = (np.hstack((labels_test_0_REG, labels_test_1_REG))), np.hstack((labels_test_0_INV, labels_test_1_INV))

        if part == 2:
            images_train = np.vstack((train_images_0_REG, train_images_0_INV, train_images_1_REG, train_images_1_INV))
            labels_train = np.hstack((np.zeros(N_label, dtype=np.uint8), np.ones(N_label, dtype=np.uint8)))


        elif part == 5 or part == 6:
            images_train_REG = np.vstack((train_images_0_REG, train_images_1_REG))
            labels_train_REG = np.hstack((np.zeros(N_inv, dtype=np.uint8), np.ones(N_inv, dtype=np.uint8)))
            images_train_INV = np.vstack((train_images_0_INV, train_images_1_INV))
            labels_train_INV = np.hstack((np.zeros(N_inv, dtype=np.uint8), np.ones(N_inv, dtype=np.uint8)))
            images_train = (images_train_REG, images_train_INV)
            labels_train = (labels_train_REG, labels_train_INV)

    elif part == 3:
        pass
    elif part == 4:
        pass


    # Let's mix up the train samples:
    if isinstance(images_train, tuple):
        images, labels = [], []
        for t, _ in enumerate(images_train):
            idx = sample(range(images_train[t].shape[0]), images_train[t].shape[0])
            images.append(images_train[t][idx])
            labels.append(labels_train[t][idx])
        images_train, labels_train = tuple(images), tuple(labels)
    else:
        idx = sample(range(images_train.shape[0]), images_train.shape[0])
        images_train = images_train[idx]
        labels_train = labels_train[idx]

    # Putting all train and test data in Dictionary
    tt_data = {'train_images': images_train,
               'train_labels': labels_train,
               'test_images': images_test,
               'test_labels': labels_test}

    return tt_data