import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
from scipy import io
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms



labels_list = [{"id": -1, "name": "void",       "rgb_values": [0,   0,    0]},
               {"id": 0,  "name": "building",   "rgb_values": [128, 0,    0]},
               {"id": 1,  "name": "grass",      "rgb_values": [0,   128,  0]},
               {"id": 2,  "name": "tree",       "rgb_values": [128, 128,  0]},
               {"id": 3,  "name": "cow",        "rgb_values": [0,   0,    128]},
               {"id": 4,  "name": "horse",      "rgb_values": [128, 0,    128]},
               {"id": 5,  "name": "sheep",      "rgb_values": [0,   128,  128]},
               {"id": 6,  "name": "sky",        "rgb_values": [128, 128,  128]},
               {"id": 7,  "name": "mountain",   "rgb_values": [64,  0,    0]},
               {"id": 8,  "name": "airplane",   "rgb_values": [192, 0,    0]},
               {"id": 9,  "name": "water",      "rgb_values": [64,  128,  0]},
               {"id": 10,  "name": "face",       "rgb_values": [192, 128,  0]},
               {"id": 11,  "name": "car",        "rgb_values": [64,  0,    128]},
               {"id": 12, "name": "bicycle",    "rgb_values": [192, 0,    128]},
               {"id": 13, "name": "flower",     "rgb_values": [64,  128,  128]},
               {"id": 14, "name": "sign",       "rgb_values": [192, 128,  128]},
               {"id": 15, "name": "bird",       "rgb_values": [0,   64,   0]},
               {"id": 16, "name": "book",       "rgb_values": [128, 64,   0]},
               {"id": 17, "name": "chair",      "rgb_values": [0,   192,  0]},
               {"id": 18, "name": "road",       "rgb_values": [128, 64,   128]},
               {"id": 19, "name": "cat",        "rgb_values": [0,   192,  128]},
               {"id": 20, "name": "dog",        "rgb_values": [128, 192,  128]},
               {"id": 21, "name": "body",       "rgb_values": [64,  64,   0]},
               {"id": 22, "name": "boat",       "rgb_values": [192, 64,   0]}]


class SegmentationData(data.Dataset):

    def __init__(self, root, image_list):
        self.root = root

        with open(os.path.join(self.root, image_list)) as f:
            self.image_names = f.read().splitlines()

    def __getitem__(self, index):
        to_tensor = transforms.ToTensor()
        img_id = self.image_names[index].replace('.bmp', '')

        img = Image.open(os.path.join(self.root, 'images', img_id + '.bmp')).convert('RGB')
        center_crop = transforms.CenterCrop(240)
        img = center_crop(img)
        img = to_tensor(img)

        target = Image.open(os.path.join(self.root, 'targets', img_id + '_GT.bmp'))
        target = center_crop(target)
        target = np.array(target, dtype=np.int64)

        target_labels = target[..., 0]
        for label in labels_list:
            mask = np.all(target == label['rgb_values'], axis=2)
            target_labels[mask] = label['id']

        target_labels = torch.from_numpy(target_labels.copy())

        return img, target_labels

    def __len__(self):
        return len(self.image_names)


class OverfitSampler(object):
    """
    Sample dataset to overfit.
    """

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples


class CIFAR10Data(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]

        img = torch.from_numpy(img)
        return img, label

    def __len__(self):
        return len(self.y)


def get_CIFAR10_data(num_training=48000, num_validation=1000,
                     num_test=1000, dtype=np.float32):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers.
    """
    # Load the raw CIFAR-10 data
    path = 'datasets/cifar10_train.p'
    with open(path, 'rb') as f:
        datadict = pickle.load(f)
        X = np.array(datadict['data'])
        y = np.array(datadict['labels'])
        X = X.reshape(-1, 3, 32, 32).astype(dtype)

    X /= 255.0
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X, axis=0)
    X -= mean_image

    # Subsample the data
    mask = range(num_training)
    X_train = X[mask]
    y_train = y[mask]
    mask = range(num_training, num_training + num_validation)
    X_val = X[mask]
    y_val = y[mask]
    mask = range(num_training + num_validation,
                 num_training + num_validation + num_test)
    X_test = X[mask]
    y_test = y[mask]

    return (CIFAR10Data(X_train, y_train),
            CIFAR10Data(X_val, y_val),
            CIFAR10Data(X_test, y_test),
            mean_image)


def scoring_function(x, lin_exp_boundary, doubling_rate):
    assert np.all([x >= 0, x <= 1])
    score = np.zeros(x.shape)
    lin_exp_boundary = lin_exp_boundary
    linear_region = np.logical_and(x > 0.1, x <= lin_exp_boundary)
    exp_region = np.logical_and(x > lin_exp_boundary, x <= 1)
    score[linear_region] = 100.0 * x[linear_region]
    c = doubling_rate
    a = 100.0 * lin_exp_boundary / np.exp(lin_exp_boundary * np.log(2) / c)
    b = np.log(2.0) / c
    score[exp_region] = a * np.exp(b * x[exp_region])
    return score
