import keras

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Concatenate, Reshape, Softmax

from keras import backend as K

import keras.losses

import os

import pickle
import numpy as np
from sklearn import preprocessing
import pandas as pd

import matplotlib.pyplot as plt

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

def _one_hot_encode(df, col='utr', seq_len=50):
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

def load_optimus5_data(train_data_path, test_data_path) :
    
    #Train data
    e_train = pd.read_csv(train_data_path)
    e_train.loc[:,'scaled_rl'] = preprocessing.StandardScaler().fit_transform(e_train.loc[:,'rl'].values.reshape(-1,1))

    seq_e_train = _one_hot_encode(e_train,seq_len=50)
    x_train = seq_e_train
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
    y_train = np.array(e_train['scaled_rl'].values)
    y_train = np.reshape(y_train, (y_train.shape[0],1))

    #Test data

    e_test = pd.read_csv(test_data_path)
    e_test.loc[:,'scaled_rl'] = preprocessing.StandardScaler().fit_transform(e_test.loc[:,'rl'].values.reshape(-1,1))

    seq_e_test = _one_hot_encode(e_test, seq_len=50)
    x_test = seq_e_test
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]))
    y_test = np.array(e_test['scaled_rl'].values)
    y_test = np.reshape(y_test, (y_test.shape[0],1))


    e_extra = pd.DataFrame({
        'utr' : [
            "CCGGCTTATCAATGGGAAGCGTCGATTGCGACAAGGGTCGTGCTCGCTAG", #synthetic example
            "CCGGCTTATCAATGGGAAGCGTCGATTGCGACAAGGGTCGTTAGCGCTAG", #synthetic example
            "CCGGCTTATCAATGGGAATGGTCGATTGCGACAAGGGTCGTTAGCGCTAG", #synthetic example
            "CGCGCCTCGGGATGCCCAGCTGATCAAGGAGCTGGGGCTGCGGCTGCTCT", #rs138958351, gain_of_one_oof_uaug, ref
            "CGCGCCTCGGGATGCCCAGATGATCAAGGAGCTGGGGCTGCGGCTGCTCT" #rs138958351, gain_of_one_oof_uaug, var
        ]
    })
    seq_e_extra = _one_hot_encode(e_extra, seq_len=50)
    x_extra = seq_e_extra
    x_extra = np.reshape(x_extra, (x_extra.shape[0], 1, x_extra.shape[1], x_extra.shape[2]))

    y_extra = np.zeros((x_extra.shape[0],1))
    
    #Prepend extra examples to test data
    x_test = np.concatenate([
        x_extra,
        x_test[:-5]
    ], axis=0)
    
    y_test = np.concatenate([
        y_extra,
        y_test[:-5]
    ], axis=0)

    print("x_train.shape = " + str(x_train.shape))
    print("x_test.shape = " + str(x_test.shape))

    print("y_train.shape = " + str(y_train.shape))
    print("y_test.shape = " + str(y_test.shape))
    
    return x_train, y_train, x_test, y_test

def load_optimus5_predictor(predictor_path) :
    
    #Load predictor model

    #Load Predictor
    predictor_temp = load_model(predictor_path)

    input_padded = Input(shape=(1, 50, 4), name="utr_input_padded")

    predictor = Model(
        input_padded,
        predictor_temp(Lambda(lambda x: x[:, 0, ...])(input_padded))
    )

    predictor.trainable = False
    predictor.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss='mean_squared_error')

    return predictor


def _letter_at_dna(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0) :
    
    fp = FontProperties(family="DejaVu Sans", weight="bold")

    globscale = 1.35
    LETTERS = {	"T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
                "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
                "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
                "C" : TextPath((-0.366, 0), "C", size=1, prop=fp),
                "UP" : TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
                "DN" : TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
                "(" : TextPath((-0.25, 0), "(", size=1, prop=fp),
                "." : TextPath((-0.125, 0), "-", size=1, prop=fp),
                ")" : TextPath((-0.1, 0), ")", size=1, prop=fp)}
    COLOR_SCHEME = {'G': 'orange', 
                    'A': 'red', 
                    'C': 'blue', 
                    'T': 'darkgreen',
                    'UP': 'green', 
                    'DN': 'red',
                    '(': 'black',
                    '.': 'black', 
                    ')': 'black'}
    
    text = LETTERS[letter]

    chosen_color = COLOR_SCHEME[letter]
    if color is not None :
        chosen_color = color

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

def _plot_dna_logos_on_axis(ax_logo, pwms, sequence_templates=None, show_template=True, logo_height=1.0, plot_start=0, plot_end=164) :
    
    #Slice according to seq trim index
    pwms = pwms[:, 0, plot_start: plot_end, :]
    sequence_templates = [sequence_template[plot_start: plot_end] for sequence_template in sequence_templates]

    pwms += 0.0001
    for j in range(0, pwms.shape[1]) :
        pwms[:, j, :] /= np.sum(pwms[:, j, :], axis=1).reshape(-1, 1)

    entropies = np.zeros(pwms.shape)
    entropies[pwms > 0] = pwms[pwms > 0] * -np.log2(pwms[pwms > 0])
    entropies = np.sum(entropies, axis=2)
    conservations = 2 - entropies
    
    for k in range(pwms.shape[0]) :
        pwm = pwms[k, :, :]
        sequence_template = sequence_templates[k]
        conservation = conservations[k]

        height_base = (1.0 - logo_height) / 2. + 5 * k + 0.5

        for j in range(0, pwm.shape[0]) :
            sort_index = np.argsort(pwm[j, :])

            for ii in range(0, 4) :
                i = sort_index[ii]

                nt_prob = pwm[j, i] * conservation[j]

                nt = ''
                if i == 0 :
                    nt = 'A'
                elif i == 1 :
                    nt = 'C'
                elif i == 2 :
                    nt = 'G'
                elif i == 3 :
                    nt = 'T'

                color = None
                plot_nt = True
                if sequence_template[j] != '$' :
                    color = 'black'
                    if not show_template :
                        plot_nt = False

                if plot_nt :
                    if ii == 0 :
                        _letter_at_dna(nt, j + 0.5, height_base, nt_prob * logo_height, ax_logo, color=color)
                    else :
                        prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
                        _letter_at_dna(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, ax_logo, color=color)

        plt.sca(ax_logo)

        plt.xlim((0, plot_end - plot_start))
        plt.ylim((-0.1, 5 * 4 + 4))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.axis('off')

        ax_logo.plot([0, plot_end - plot_start], [0.01 + height_base, 0.01 + height_base], color='black', linestyle='-', linewidth=2)

#Make GIF animation
def animate_optimus5_examples(train_history, example_indices, model_name, sequence_template) :
    
    batch_history = train_history['monitor_batches']
    pwm_history = train_history['monitor_pwms']
    scores_history = train_history['monitor_importance_scores']

    nll_loss_history = train_history['monitor_nll_losses']
    entropy_loss_history = train_history['monitor_entropy_losses']

    sel_pwm_history = [
        temp_pwms[0][example_indices, ...]
        for temp_pwms in pwm_history
    ]

    sel_scores_history = [
        temp_scores[0][example_indices, ...]
        for temp_scores in scores_history
    ]

    sel_nll_loss_history = [
        temp_nll_loss[example_indices, ...]
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
    
    #Animation 1: NLL Loss and Example Sequence PWMs
    n_examples = sel_nll_loss_history[0].shape[0]
    n_frames = len(pwm_history) - 1

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), gridspec_kw={'width_ratios': [1], 'height_ratios': [2, 3.5]})

    #Plot PWMs
    ax2.axis('off')
    ax2.get_yaxis().set_visible(False)

    _plot_dna_logos_on_axis(ax2, sel_pwm_history[-1], sequence_templates=[sequence_template] * n_examples, show_template=True, logo_height=0.75, plot_start=0, plot_end=50)

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
        ax2.axis('off')

        _plot_dna_logos_on_axis(ax2, sel_pwm_history[t], sequence_templates=[sequence_template] * n_examples, show_template=True, logo_height=0.75, plot_start=0, plot_end=50)

        return []


    anim = FuncAnimation(f, animate, init_func=init, frames=n_frames+1, interval=50, blit=True)

    anim.save(model_name + '.gif', writer='imagemagick')
