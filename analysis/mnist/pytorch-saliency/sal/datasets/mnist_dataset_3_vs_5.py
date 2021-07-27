import torch
from torchvision.transforms import *
from torch.utils.data import dataloader
import time, random
from ..utils.pytorch_fixes import *

import tensorflow as tf
from tensorflow.keras.datasets import mnist

import numpy as np

SUGGESTED_BS = 64
SUGGESTED_EPOCHS_PER_STEP = 10
SUGGESTED_BASE = 32
NUM_CLASSES = 10
CLASSES = '''0 0
1 1
2 2
3 3
4 4
5 5
6 6
7 7
8 8
9 9'''.splitlines()

def get_data_x_y(verbose=False) :
    #Load MNIST data

    dataset_name = "mnist_3_vs_5"

    img_rows, img_cols = 28, 28

    num_classes = 10
    batch_size = 32

    included_classes = { 3, 5 }

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    keep_index_train = []
    for i in range(y_train.shape[0]) :
        if y_train[i] in included_classes :
            keep_index_train.append(i)

    keep_index_test = []
    for i in range(y_test.shape[0]) :
        if y_test[i] in included_classes :
            keep_index_test.append(i)

    x_train = x_train[keep_index_train]
    x_test = x_test[keep_index_test]
    y_train = y_train[keep_index_train]
    y_test = y_test[keep_index_test]

    n_train = int((x_train.shape[0] // batch_size) * batch_size)
    n_test = int((x_test.shape[0] // batch_size) * batch_size)
    x_train = x_train[:n_train]
    x_test = x_test[:n_test]
    y_train = y_train[:n_train]
    y_test = y_test[:n_test]


    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if verbose :
        print("x_train.shape = " + str(x_train.shape))

        print("n train samples = " + str(x_train.shape[0]))
        print("n test samples = " + str(x_test.shape[0]))

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    #Binarize images

    def _binarize_images(x, val_thresh=0.5) :

        x_bin = np.zeros(x.shape)
        x_bin[x >= val_thresh] = 1.

        return x_bin

    x_train = _binarize_images(x_train, val_thresh=0.5)
    x_test = _binarize_images(x_test, val_thresh=0.5)
    
    #Add padding to images
    padding = 2
    
    x_train = np.concatenate([
        np.zeros((x_train.shape[0], x_train.shape[1] + 4, 2, 1)),
        np.concatenate([
            np.zeros((x_train.shape[0], 2, x_train.shape[2], 1)),
            x_train,
            np.zeros((x_train.shape[0], 2, x_train.shape[2], 1))
        ], axis=1),
        np.zeros((x_train.shape[0], x_train.shape[1] + 4, 2, 1))
    ], axis=2)
    
    x_test = np.concatenate([
        np.zeros((x_test.shape[0], x_test.shape[1] + 4, 2, 1)),
        np.concatenate([
            np.zeros((x_test.shape[0], 2, x_test.shape[2], 1)),
            x_test,
            np.zeros((x_test.shape[0], 2, x_test.shape[2], 1))
        ], axis=1),
        np.zeros((x_test.shape[0], x_test.shape[1] + 4, 2, 1))
    ], axis=2)
    
    if verbose :
        print("padded x_train.shape = " + str(x_train.shape))
        print("padded x_test.shape = " + str(x_test.shape))
    
    return x_train, y_train, x_test, y_test

class SimpleSet(torch.utils.data.Dataset) :
    
    def __init__(self, x_numpy, y_numpy) :
        self.x = torch.tensor(np.transpose(x_numpy, (0, 3, 1, 2))).float()
        #self.y = torch.tensor(np.array(np.argmax(y_numpy, axis=1), dtype=np.int)).long()
        self.y = np.array(np.argmax(y_numpy, axis=1), dtype=np.int)
        self.length = x_numpy.shape[0]
    
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return self.length

def get_train_dataset():
    
    x_train, y_train, _, _ = get_data_x_y(verbose=True)
    
    return SimpleSet(x_train, y_train)

def get_val_dataset():
    
    _, _, x_test, y_test = get_data_x_y()
    
    return SimpleSet(x_test, y_test)

def get_loader(dataset, batch_size=64, pin_memory=True):
    return dataloader.DataLoader(dataset=dataset, batch_size=batch_size,
                                 shuffle=True, drop_last=True, num_workers=8, pin_memory=True)


def test():
    BS = 64
    SAMP = 20
    dts = get_train_dataset()
    loader = get_loader(dts, batch_size=BS)
    i = 0
    t = time.time()
    for ims, labs in loader:
        i+=1
        if not i%20:
            print "Images per second:", SAMP*BS/(time.time()-t)
            pycat.show(ims[0].numpy())
            t = time.time()
        if i==100:
            break