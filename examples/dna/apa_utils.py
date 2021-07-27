import keras

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Concatenate, Reshape, Softmax

from keras import backend as K

import keras.losses

import os

import pickle
import numpy as np
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

def load_apa_data(data_path, encoder) :
    
    #Define dataset/experiment name
    dataset_name = "apa_doubledope"

    #Load cached dataframe
    data_df = pd.read_csv(data_path, sep='\t')

    print("len(data_df) = " + str(len(data_df)) + " (loaded)")
    
    #Load data matrices
    test_set_size = 0.05

    batch_size = 32

    #Generate training and test set indexes
    data_index = np.arange(len(data_df), dtype=np.int)

    train_index = data_index[:-int(len(data_df) * test_set_size)]
    test_index = data_index[train_index.shape[0]:]

    data_df['seq'] = data_df['padded_seq'].str.slice(180, 180 + 205)

    n_train = train_index.shape[0] // batch_size * batch_size
    n_test = test_index.shape[0] // batch_size * batch_size

    train_index = train_index[:n_train]
    test_index = test_index[:n_test]

    train_df = data_df.iloc[train_index].copy().reset_index(drop=True)
    test_df = data_df.iloc[test_index].copy().reset_index(drop=True)

    #Load data matrices

    x_train = np.concatenate([encoder(row['seq'])[None, None, ...] for _, row in train_df.iterrows()], axis=0)
    x_test = np.concatenate([encoder(row['seq'])[None, None, ...] for _, row in test_df.iterrows()], axis=0)

    y_train = np.array(train_df['proximal_usage'].values).reshape(-1, 1)
    y_test = np.array(test_df['proximal_usage'].values).reshape(-1, 1)

    print("x_train.shape = " + str(x_train.shape))
    print("x_test.shape = " + str(x_test.shape))

    print("y_train.shape = " + str(y_train.shape))
    print("y_test.shape = " + str(y_test.shape))
    
    return x_train, y_train, x_test, y_test

def load_apa_predictor(predictor_path) :
    
    #Load predictor model

    #Shared model definition
    layer_1 = Conv2D(96, (8, 4), padding='valid', activation='relu')
    layer_1_pool = MaxPooling2D(pool_size=(2, 1))
    layer_2 = Conv2D(128, (6, 1), padding='valid', activation='relu')
    layer_dense = Dense(256, activation='relu')
    layer_drop = Dropout(0.2)

    def shared_model(seq_input, distal_pas_input) :
        return layer_drop(
            layer_dense(
                Concatenate()([
                    Flatten()(
                        layer_2(
                            layer_1_pool(
                                layer_1(
                                    seq_input
                                )
                            )
                        )
                    ),
                    distal_pas_input
                ])
            ),
            training=False
        )

    #Inputs
    seq_input = Input(name="seq_input", shape=(1, 205, 4))

    permute_layer = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)))

    lib_input = Lambda(lambda x: K.tile(K.expand_dims(K.constant(np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])), axis=0), (K.shape(x)[0], 1)))(seq_input)
    distal_pas_input = Lambda(lambda x: K.tile(K.expand_dims(K.constant(np.array([1.])), axis=0), (K.shape(x)[0], 1)))(seq_input)

    plasmid_out_shared = Concatenate()([shared_model(permute_layer(seq_input), distal_pas_input), lib_input])

    plasmid_out_cut = Dense(206, activation='softmax', kernel_initializer='zeros')(plasmid_out_shared)
    plasmid_out_iso = Dense(1, activation='sigmoid', kernel_initializer='zeros', name="apa_logodds")(plasmid_out_shared)

    predictor_temp = Model(
        inputs=[
            seq_input
        ],
        outputs=[
            plasmid_out_iso,
            plasmid_out_cut
        ]
    )

    predictor_temp.load_weights(predictor_path)

    predictor = Model(
        inputs=predictor_temp.inputs,
        outputs=[
            predictor_temp.outputs[0]
        ]
    )

    predictor.trainable = False

    predictor.compile(
        optimizer=keras.optimizers.SGD(lr=0.1),
        loss='mean_squared_error'
    )

    return predictor

def load_apa_predictor_cleavage_logodds(predictor_path) :
    
    #Load predictor model

    #Shared model definition
    layer_1 = Conv2D(96, (8, 4), padding='valid', activation='relu')
    layer_1_pool = MaxPooling2D(pool_size=(2, 1))
    layer_2 = Conv2D(128, (6, 1), padding='valid', activation='relu')
    layer_dense = Dense(256, activation='relu')
    layer_drop = Dropout(0.2)

    def shared_model(seq_input, distal_pas_input) :
        return layer_drop(
            layer_dense(
                Concatenate()([
                    Flatten()(
                        layer_2(
                            layer_1_pool(
                                layer_1(
                                    seq_input
                                )
                            )
                        )
                    ),
                    distal_pas_input
                ])
            ),
            training=False
        )

    #Inputs
    seq_input = Input(name="seq_input", shape=(205, 4))

    permute_layer = Lambda(lambda x: x[..., None])

    lib_input = Input(name="lib_input", shape=(13,))
    distal_pas_input = Input(name="distal_pas_input", shape=(1,))

    plasmid_out_shared = Concatenate()([shared_model(permute_layer(seq_input), distal_pas_input), lib_input])

    plasmid_out_cut = Dense(206, activation='softmax', kernel_initializer='zeros')(plasmid_out_shared)
    plasmid_out_iso = Dense(1, activation='sigmoid', kernel_initializer='zeros', name="apa_logodds")(plasmid_out_shared)

    predictor_temp = Model(
        inputs=[
            seq_input,
            lib_input,
            distal_pas_input
        ],
        outputs=[
            plasmid_out_iso,
            plasmid_out_cut
        ]
    )

    predictor_temp.load_weights(predictor_path)
    
    rel_cut_layer = Lambda(lambda x: x[:, 76:76+40] / K.sum(x[:, 76:76+40], axis=1)[:, None], name='rel_cut')
    rel_cut_summarize_layer = Lambda(lambda x: K.concatenate([K.sum(x[:, 5:5+3], axis=1)[:, None], K.sum(x[:, 15:15+3], axis=1)[:, None], K.sum(x[:, 25:25+3], axis=1)[:, None], K.sum(x[:, 35:35+3], axis=1)[:, None]], axis=1), name='rel_cut_summarize')
    rel_cut_logodds_layer = Lambda(lambda x: K.log(K.clip(x, K.epsilon(), 1.)), name='rel_cut_logodds')

    predictor = Model(
        inputs=predictor_temp.inputs,
        outputs=[rel_cut_logodds_layer(rel_cut_summarize_layer(rel_cut_layer(predictor_temp.outputs[1])))]
    )

    predictor.trainable = False

    predictor.compile(
        optimizer=keras.optimizers.SGD(lr=0.1),
        loss='mean_squared_error'
    )

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
def animate_apa_examples(train_history, example_indices, model_name, sequence_template) :
    
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

    _plot_dna_logos_on_axis(ax2, sel_pwm_history[-1], sequence_templates=[sequence_template] * n_examples, show_template=True, logo_height=0.75, plot_start=70-25, plot_end=76+40)

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
            elif curr_speed > 10 :
                speed_sign = ">>>>"

        ax1.set_title("Weight Update " + str(batch_history[t]) + "\n" + str(curr_speed) + "x Speedup " + speed_sign, fontsize=14)

        ax2.clear()
        ax2.axis('off')

        _plot_dna_logos_on_axis(ax2, sel_pwm_history[t], sequence_templates=[sequence_template] * n_examples, show_template=True, logo_height=0.75, plot_start=70-25, plot_end=76+40)

        return []


    anim = FuncAnimation(f, animate, init_func=init, frames=n_frames+1, interval=50, blit=True)

    anim.save(model_name + '.gif', writer='imagemagick')
