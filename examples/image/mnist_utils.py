import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Lambda

import os
import pickle
import numpy as np
import pandas as pd

from keras.datasets import mnist

import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

from matplotlib.animation import FuncAnimation
from matplotlib import colors

def load_mnist_data() :
    
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

    print("n train samples = " + str(x_train.shape[0]))
    print("n test samples = " + str(x_test.shape[0]))

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    #Binarize images

    def _binarize_images(x, val_thresh=0.5) :

        x_bin = np.zeros(x.shape)
        x_bin[x >= val_thresh] = 1.

        return x_bin

    x_train = _binarize_images(x_train, val_thresh=0.5)
    x_test = _binarize_images(x_test, val_thresh=0.5)

    #Make categorical channels
    x_train = np.concatenate([1. - x_train, x_train], axis=-1)
    x_test = np.concatenate([1. - x_test, x_test], axis=-1)

    print("x_train.shape = " + str(x_train.shape))
    
    return x_train, y_train, x_test, y_test

def load_mnist_predictor(predictor_path) :
    
    #Load Predictor
    predictor_temp = load_model(predictor_path)

    input_multi_channel = Input(shape=(28, 28, 2), name="image_multi_channel")

    predictor = Model(
        input_multi_channel,
        predictor_temp(Lambda(lambda x: x[..., 1:2])(input_multi_channel))
    )

    predictor.trainable = False
    predictor.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss='mean_squared_error')

    return predictor

#Make GIF animation
def animate_mnist_examples(train_history, x_test, examples_indices, model_name) :

    batch_history = train_history['monitor_batches']
    scores_history = train_history['monitor_importance_scores']

    nll_loss_history = train_history['monitor_nll_losses']

    sel_scores_history = [
        temp_scores[0][examples_indices, ...]
        for temp_scores in scores_history
    ]

    sel_nll_loss_history = [
        temp_nll_loss[examples_indices, ...]
        for temp_nll_loss in nll_loss_history
    ]

    min_nll_loss = np.min(np.array([np.min(sel_nll_loss_history[i]) for i in range(len(sel_nll_loss_history))]))
    max_nll_loss = np.max(np.array([np.max(sel_nll_loss_history[i]) for i in range(len(sel_nll_loss_history))]))

    def _rolling_average(x, window=50) :
        x_avg = []

        for j in range(x.shape[0]) :
            j_min = max(j - window + 1, 0)
            x_avg.append(np.mean(x[j_min:j+1]))

        return np.array(x_avg)

    sel_nll_loss_history = np.concatenate([sel_nll_loss_history[t].reshape(1, -1) for t in range(len(sel_nll_loss_history))], axis=0)

    for example_ix in range(sel_nll_loss_history.shape[1]) :
        sel_nll_loss_history[:, example_ix] = _rolling_average(sel_nll_loss_history[:, example_ix])

    sel_nll_loss_history = [
        sel_nll_loss_history[t, :] for t in range(sel_nll_loss_history.shape[0])
    ]
    
    #Concatenate time slices of all chosen example images
    big_images = []
    big_scores = []

    for t in range(len(sel_scores_history)) :

        big_image = np.concatenate([
            np.concatenate([x_test[0, :, :, 1], x_test[1, :, :, 1], x_test[2, :, :, 1]], axis=0),
            np.concatenate([x_test[3, :, :, 1], x_test[4, :, :, 1], x_test[5, :, :, 1]], axis=0),
            np.concatenate([x_test[6, :, :, 1], x_test[7, :, :, 1], x_test[8, :, :, 1]], axis=0),
        ], axis=1)

        big_score = np.concatenate([
            np.concatenate([sel_scores_history[t][0, :, :, 0], sel_scores_history[t][1, :, :, 0], sel_scores_history[t][2, :, :, 0]], axis=0),
            np.concatenate([sel_scores_history[t][3, :, :, 0], sel_scores_history[t][4, :, :, 0], sel_scores_history[t][5, :, :, 0]], axis=0),
            np.concatenate([sel_scores_history[t][6, :, :, 0], sel_scores_history[t][7, :, :, 0], sel_scores_history[t][8, :, :, 0]], axis=0),
        ], axis=1)

        big_images.append(big_image)    
        big_scores.append(big_score)

    #Animation 1: NLL Loss and Example Images
    n_examples = sel_nll_loss_history[0].shape[0]
    n_frames = len(big_images) - 1

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 9), gridspec_kw={'width_ratios': [1], 'height_ratios': [2, 3.5]})

    #Plot Images
    #ax2.axis('off')
    #ax2.get_yaxis().set_visible(False)
    plt.sca(ax2)
    plt.xticks([], [])
    plt.yticks([], [])

    ax2.imshow(big_images[-1], cmap="Greys", vmin=0.0, vmax=1.0, aspect='equal')
    ax2.imshow(big_scores[-1], alpha=0.75, cmap="hot", vmin=0.0, vmax=np.max(big_scores[-1]), aspect='equal')

    loss_lines = []
    for i in range(n_examples) :
        line, = ax1.plot([], [], linewidth=2)
        loss_lines.append(line)

    plt.sca(ax1)
    plt.ylabel("Reconstruction Loss", fontsize=14)

    plt.xticks([0, batch_history[n_frames-1]], [0, batch_history[n_frames-1]], fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlim(0, batch_history[n_frames-1])
    plt.ylim(0., max_nll_loss + 0.02 * max_nll_loss * np.sign(max_nll_loss))

    plt.title("Weight Update 0\n1x Speedup >", fontsize=14)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.15)

    loss_data_x = [[0] for i in range(n_examples)]
    loss_data_y = [[sel_nll_loss_history[0][i]] for i in range(n_examples)]

    def init() :
        for i in range(n_examples) :
            loss_lines[i].set_data([], [])

        return []

    def animate(t) :
        if t % 10 == 0 :
            print("Grabbing frame " + str(t) + "...")

        if t > 0 :
            for i in range(n_examples) :
                loss_data_x[i].append(batch_history[t])
                loss_data_y[i].append(sel_nll_loss_history[t][i])

                loss_lines[i].set_data(loss_data_x[i], loss_data_y[i])

        curr_speed = 1
        speed_sign = ">"
        if t > 0 :
            curr_speed = int(batch_history[t] - batch_history[t-1])
            if curr_speed <= 1 :
                speed_sign = ">"
            elif curr_speed > 1 and curr_speed <= 5 :
                speed_sign = ">>"
            elif curr_speed > 5 :
                speed_sign = ">>>"

        ax1.set_title("Weight Update " + str(batch_history[t]) + "\n" + str(curr_speed) + "x Speedup " + speed_sign, fontsize=14)

        ax2.clear()
        plt.sca(ax2)
        plt.xticks([], [])
        plt.yticks([], [])

        ax2.imshow(big_images[t], cmap="Greys", vmin=0.0, vmax=1.0, aspect='equal')
        ax2.imshow(big_scores[t], alpha=0.75, cmap="hot", vmin=0.0, vmax=np.max(big_scores[t]), aspect='equal')

        return []


    anim = FuncAnimation(f, animate, init_func=init, frames=n_frames+1, interval=50, blit=True)

    anim.save(model_name + '.gif', writer='imagemagick')
