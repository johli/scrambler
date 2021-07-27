import torch
from torchvision.transforms import *
from torch.utils.data import dataloader
import time, random
from ..utils.pytorch_fixes import *

import numpy as np

from sklearn import preprocessing
import pandas as pd

SUGGESTED_BS = 64
SUGGESTED_EPOCHS_PER_STEP = 10
SUGGESTED_BASE = 32
NUM_CLASSES = 2
CLASSES = '''Repression
Activation'''.splitlines()

def one_hot_encode(df, col='utr', seq_len=50):
    # Dictionary returning one-hot encoding of nucleotides. 
    nuc_d = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}

    # Creat empty matrix.
    vectors=np.empty([len(df),seq_len,4])

    # Iterate through UTRs and one-hot encode
    for i,seq in enumerate(df[col].str[:seq_len]): 
        seq = seq.lower()
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors

def get_data_x_y(verbose=False) :
    
    dataset_name = "optimus5_synth"
    
    #Train data
    e_train = pd.read_csv("../../scrambler_pytorch_optimus5/bottom5KIFuAUGTop5KIFuAUG.csv")
    e_train.loc[:,'scaled_rl'] = preprocessing.StandardScaler().fit_transform(e_train.loc[:,'rl'].values.reshape(-1,1))

    seq_e_train = one_hot_encode(e_train,seq_len=50)
    x_train = seq_e_train
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
    y_train = np.array(e_train['scaled_rl'].values)
    y_train = np.reshape(y_train, (y_train.shape[0],1))

    y_train = (y_train >= 0.)
    y_train = np.concatenate([1. - y_train, y_train], axis=1)

    #Test data
    allFiles = ["../../scrambler_pytorch_optimus5/optimus5_synthetic_random_insert_if_uorf_1_start_1_stop_variable_loc_512.csv",
                "../../scrambler_pytorch_optimus5/optimus5_synthetic_random_insert_if_uorf_1_start_2_stop_variable_loc_512.csv",
                "../../scrambler_pytorch_optimus5/optimus5_synthetic_random_insert_if_uorf_2_start_1_stop_variable_loc_512.csv",
                "../../scrambler_pytorch_optimus5/optimus5_synthetic_random_insert_if_uorf_2_start_2_stop_variable_loc_512.csv"]

    x_tests = []

    for csv_to_open in allFiles :

        #Load dataset for benchmarking 
        dataset_name = csv_to_open.replace(".csv", "")
        benchmarkSet = pd.read_csv(csv_to_open)

        seq_e_test = one_hot_encode(benchmarkSet, seq_len=50)
        x_test = seq_e_test[:, None, ...]

        print(x_test.shape)

        x_tests.append(x_test)

    x_test = np.concatenate(x_tests, axis=0)
    y_test = -1. * np.ones((x_test.shape[0], 1))

    y_test = (y_test >= 0.)
    y_test = np.concatenate([1. - y_test, y_test], axis=1)
    
    #Add padding to images
    padding = 14
    
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