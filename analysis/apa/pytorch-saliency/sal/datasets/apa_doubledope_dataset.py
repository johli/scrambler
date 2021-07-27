import torch
from torchvision.transforms import *
from torch.utils.data import dataloader
import time, random
from ..utils.pytorch_fixes import *

import numpy as np

SUGGESTED_BS = 64
SUGGESTED_EPOCHS_PER_STEP = 10
SUGGESTED_BASE = 32
NUM_CLASSES = 2
CLASSES = '''Distal
Proximal'''.splitlines()

def get_data_x_y(verbose=False) :
    
    dataset_name = "apa_doubledope"
    
    npzfile = np.load("../../scrambler_pytorch_aparent/" + dataset_name + ".npz")
    
    x_train = npzfile['x_train']
    y_train = npzfile['y_train']
    x_test = npzfile['x_test']
    y_test = npzfile['y_test']
    
    #Add padding to images
    padding = 51
    
    x_train = np.concatenate([
        x_train,
        np.zeros((x_train.shape[0], x_train.shape[1], padding, x_train.shape[3]))
    ], axis=2)
    
    x_test = np.concatenate([
        x_test,
        np.zeros((x_test.shape[0], x_test.shape[1], padding, x_test.shape[3]))
    ], axis=2)
    
    if verbose :
        print("padded x_train.shape = " + str(x_train.shape))
        print("padded x_test.shape = " + str(x_test.shape))
    
    #Make binary labels

    digit_train = np.copy(y_train[:, 0])
    digit_test = np.copy(y_test[:, 0])

    y_train = np.zeros((digit_train.shape[0], 2))
    y_train[digit_train < 0.5, 0] = 1
    y_train[digit_train >= 0.5, 1] = 1

    y_test = np.zeros((digit_test.shape[0], 2))
    y_test[digit_test < 0.5, 0] = 1
    y_test[digit_test >= 0.5, 1] = 1
    
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