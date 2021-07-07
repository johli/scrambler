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


def get_amino_seq(row, index, flips, homodimers) :
    
    is_flip = True if flips[index] == 1 else False
    is_homodimer = True if homodimers[index] == 1 else False
    
    amino_seq_1, amino_seq_2 = row['amino_seq_1'], row['amino_seq_2']
    if is_flip :
        amino_seq_1, amino_seq_2 = row['amino_seq_2'], row['amino_seq_1']
    if is_homodimer and row['interacts'] < 0.5 :
        amino_seq_2 = amino_seq_1
    
    return amino_seq_1, amino_seq_2

def load_ppi_data(train_data_path, test_data_path, encoder) :
    
    #Define dataset/experiment name
    dataset_name = "coiled_coil_binders"
    
    #Load data matrices
    train_df = pd.read_csv(train_data_path, sep='\t')
    test_df = pd.read_csv(test_data_path, sep='\t')
    
    n_train = len(train_df)
    n_test = len(test_df)
    
    print('Training set size = ' + str(n_train))
    print('Test set size = ' + str(n_test))
    
    #Append special binder pair to test set for monitoring / animation
    test_df = pd.concat([
        pd.DataFrame({
            'amino_seq_1' : ["TAEELLEVHKKSDRVTKEHLRVSEEILKVVEVLTRGEVSSEVLKRVLRKLEELTDKLRRVTEEQRRVVEKLN"],
            'amino_seq_2' : ["DLEDLLRRLRRLVDEQRRLVEELERVSRRLEKAVRDNEDERELARLSREHSDIQDKHDKLAREILEVLKRLLERTE"],
            'interacts' : 1
        }),
        test_df.iloc[:-1]
    ]).copy().reset_index(drop=True)
    
    #Calculate sequence lengths
    train_df['amino_seq_1_len'] = train_df['amino_seq_1'].str.len()
    train_df['amino_seq_2_len'] = train_df['amino_seq_2'].str.len()
    
    test_df['amino_seq_1_len'] = test_df['amino_seq_1'].str.len()
    test_df['amino_seq_2_len'] = test_df['amino_seq_2'].str.len()

    #Load data matrices
    flips = np.random.choice([0, 1], size=len(train_df), replace=True, p=[0.5, 0.5])
    homodimers = np.random.choice([0, 1], size=len(train_df), replace=True, p=[0.95, 0.05])

    x_1_train = np.concatenate([encoder(get_amino_seq(row, index, flips, homodimers)[0])[None, None, ...] for index, row in train_df.iterrows()], axis=0)
    x_2_train = np.concatenate([encoder(get_amino_seq(row, index, flips, homodimers)[1])[None, None, ...] for index, row in train_df.iterrows()], axis=0)
    l_1_train = np.array([len(get_amino_seq(row, index, flips, homodimers)[0]) for index, row in train_df.iterrows()]).reshape(-1, 1)
    l_2_train = np.array([len(get_amino_seq(row, index, flips, homodimers)[1]) for index, row in train_df.iterrows()]).reshape(-1, 1)
    
    y_train = np.array(train_df['interacts'].values).reshape(-1, 1)
    
    flips = np.zeros(len(test_df))
    homodimers = np.zeros(len(test_df))
    
    x_1_test = np.concatenate([encoder(get_amino_seq(row, index, flips, homodimers)[0])[None, None, ...] for index, row in test_df.iterrows()], axis=0)
    x_2_test = np.concatenate([encoder(get_amino_seq(row, index, flips, homodimers)[1])[None, None, ...] for index, row in test_df.iterrows()], axis=0)
    l_1_test = np.array([len(get_amino_seq(row, index, flips, homodimers)[0]) for index, row in test_df.iterrows()]).reshape(-1, 1)
    l_2_test = np.array([len(get_amino_seq(row, index, flips, homodimers)[1]) for index, row in test_df.iterrows()]).reshape(-1, 1)
    
    y_test = np.array(test_df['interacts'].values).reshape(-1, 1)

    print("x_1_train.shape = " + str(x_1_train.shape))
    print("x_2_train.shape = " + str(x_2_train.shape))
    print("l_1_train.shape = " + str(l_1_train.shape))
    print("l_2_train.shape = " + str(l_2_train.shape))
    print("y_train.shape = " + str(y_train.shape))
    
    print("x_1_test.shape = " + str(x_1_test.shape))
    print("x_2_test.shape = " + str(x_2_test.shape))
    print("l_1_test.shape = " + str(l_1_test.shape))
    print("l_2_test.shape = " + str(l_2_test.shape))
    print("y_test.shape = " + str(y_test.shape))
    
    return x_1_train, x_2_train, l_1_train, l_2_train, y_train, x_1_test, x_2_test, l_1_test, l_2_test, y_test

def load_ppi_predictor(predictor_path) :
    
    #Load predictor model

    #Load Predictor
    predictor_temp = load_model(predictor_path, custom_objects={ 'sigmoid_nll' : lambda y_true, y_pred: y_pred })

    input_1_padded = Input(shape=(1, 81, 20), name="protein_input_1_padded")
    input_2_padded = Input(shape=(1, 81, 20), name="protein_input_2_padded")
    
    collapse_1 = Lambda(lambda x: x[:, 0, ...], name="collapse_protein_input_1")
    collapse_2 = Lambda(lambda x: x[:, 0, ...], name="collapse_protein_input_2")
    
    predictor = Model(
        [input_1_padded, input_2_padded],
        predictor_temp([collapse_1(input_1_padded), collapse_2(input_2_padded)])
    )

    predictor.trainable = False
    predictor.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss='mean_squared_error')
    
    return predictor


def _protein_letter_at(letter, x, y, yscale=1, ax=None, color='black', alpha=1.0):

    #fp = FontProperties(family="Arial", weight="bold")
    #fp = FontProperties(family="Ubuntu", weight="bold")
    fp = FontProperties(family="DejaVu Sans", weight="bold")

    globscale = 1.35
    LETTERS = {"T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
               "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
               "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
               "C" : TextPath((-0.366, 0), "C", size=1, prop=fp),

               "L" : TextPath((-0.35, 0), "L", size=1, prop=fp),
               "M" : TextPath((-0.35, 0), "M", size=1, prop=fp),
               "F" : TextPath((-0.35, 0), "F", size=1, prop=fp),
               "W" : TextPath((-0.35, 0), "W", size=1, prop=fp),
               "K" : TextPath((-0.35, 0), "K", size=1, prop=fp),
               "Q" : TextPath((-0.35, 0), "Q", size=1, prop=fp),
               "E" : TextPath((-0.35, 0), "E", size=1, prop=fp),
               "S" : TextPath((-0.35, 0), "S", size=1, prop=fp),
               "P" : TextPath((-0.35, 0), "P", size=1, prop=fp),
               "V" : TextPath((-0.35, 0), "V", size=1, prop=fp),
               "I" : TextPath((-0.35, 0), "I", size=1, prop=fp),
               "Y" : TextPath((-0.35, 0), "Y", size=1, prop=fp),
               "H" : TextPath((-0.35, 0), "H", size=1, prop=fp),
               "R" : TextPath((-0.35, 0), "R", size=1, prop=fp),
               "N" : TextPath((-0.35, 0), "N", size=1, prop=fp),
               "D" : TextPath((-0.35, 0), "D", size=1, prop=fp),
               "U" : TextPath((-0.35, 0), "U", size=1, prop=fp),
               "!" : TextPath((-0.35, 0), "!", size=1, prop=fp),

               "UP" : TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
               "DN" : TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
               "(" : TextPath((-0.25, 0), "(", size=1, prop=fp),
               "." : TextPath((-0.125, 0), "-", size=1, prop=fp),
               ")" : TextPath((-0.1, 0), ")", size=1, prop=fp)}


    if letter in LETTERS :
        text = LETTERS[letter]
    else :
        text = TextPath((-0.35, 0), letter, size=1, prop=fp)

    chosen_color = color

    if chosen_color is None :
        chosen_color = 'black'
        if letter in ['A', 'I', 'L', 'M', 'F', 'W', 'V'] : #Hydrophobic
            chosen_color = 'blue'
        elif letter in ['K' ,'R'] : #Positive charge
            chosen_color = 'red'
        elif letter in ['E', 'D'] : #Negative charge
            chosen_color = 'magenta'
        elif letter in ['N', 'Q', 'S', 'T'] : #Polar
            chosen_color = 'green'
        elif letter in ['C'] : #Cysteines
            chosen_color = 'pink'
        elif letter in ['G'] : #Glycines
            chosen_color = 'orange'
        elif letter in ['P'] : #Prolines
            chosen_color = 'yellow'
        elif letter in ['H', 'Y'] : #Aromatic
            chosen_color = 'cyan'

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

def _plot_protein_logo_on_axis(ax, y_label, residue_map, pwm, sequence_template=None, logo_height=1.0, plot_start=0, plot_end=164, color_reference=None, sequence_colors=None) :

    inv_residue_map = {
        i : sp for sp, i in residue_map.items()
    }

    #Slice according to seq trim index
    pwm = pwm[plot_start: plot_end, :]
    sequence_template = sequence_template[plot_start: plot_end]

    entropy = np.zeros(pwm.shape)
    entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(np.clip(pwm[pwm > 0], 1e-6, 1. - 1e-6))
    entropy = np.sum(entropy, axis=1)
    conservation = np.log2(len(residue_map)) - entropy#2 - entropy

    height_base = (1.0 - logo_height) / 2.

    for j in range(0, pwm.shape[0]) :
        sort_index = np.argsort(pwm[j, :])

        for ii in range(0, len(residue_map)) :
            i = sort_index[ii]

            nt_prob = pwm[j, i] * conservation[j]

            nt = inv_residue_map[i]

            color = None

            if color_reference is not None :
                if sequence_colors[j] != -1 and sequence_colors[j] >= 0 and sequence_colors[j] < len(color_reference) :
                    color = color_reference[sequence_colors[j]]
                else :
                    color = 'black'

            if sequence_template[j] != '$' :
                color = 'black'

            if nt_prob > (1. / 20.) :
                if ii == 0 :
                    _protein_letter_at(nt, j + 0.5, height_base, nt_prob * logo_height, ax, color=color)
                else :
                    prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
                    _protein_letter_at(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, ax, color=color)

    plt.sca(ax)
    plt.xlim((0, plot_end - plot_start))
    plt.ylim((0, np.log2(len(residue_map))))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    plt.axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax.text(-0.02, 0.5, y_label, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, color='black', fontsize=12, weight="bold")
    
    return

def _plot_protein_importance_scores_on_axis(ax, y_label, importance_scores, ref_seq, score_clip=None, score_max=None, single_color=None, sequence_template='', fixed_sequence_template_scores=True, plot_start=0, plot_end=96) :

    end_pos = ref_seq.find("#")

    if score_clip is not None :
        importance_scores = np.clip(np.copy(importance_scores), -score_clip, score_clip)

    max_score = np.max(np.sum(importance_scores[:, :], axis=0)) + 0.01
    if score_max is not None :
        max_score = score_max

    for i in range(0, len(ref_seq)) :
        mutability_score = np.sum(importance_scores[:, i])
        color = None if single_color is None else single_color
        if sequence_template is not None and sequence_template != '' and sequence_template[i] != '$' :
            color = 'black'
            if fixed_sequence_template_scores :
                mutability_score = max_score
        _protein_letter_at(ref_seq[i], i + 0.5, 0, mutability_score, ax, color=color)

    plt.sca(ax)
    plt.xlim((0, plot_end - plot_start))
    plt.ylim((0, max_score))
    plt.axis('off')
    plt.yticks([0.0, max_score], [0.0, max_score], fontsize=16)
    plt.axhline(y=0.01, color='black', linestyle='-', linewidth=2)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax.text(-0.02, 0.5, y_label, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, color='black', fontsize=12, weight="bold")
    
    return

#Make GIF animation
def animate_ppi_example(train_history, model_name, encoder, residue_map, x_1_test, x_2_test, l_1_test, l_2_test, test_ix, sequence_masks, sequence_template_1, sequence_template_2, normalize_scores=False, flip_loss=True, is_occlusion=True) :
    
    score_quantile = 0.95
    
    batch_history = train_history['monitor_batches']
    pwm_history = train_history['monitor_pwms']
    scores_history = train_history['monitor_importance_scores']
    nll_loss_history = train_history['monitor_nll_losses']

    sel_pwm_history_1 = [
        temp_pwms[0][:1, ...] * sequence_masks[l_1_test[test_ix, 0]][None, None, :, None]
        for temp_pwms in pwm_history
    ]
    sel_pwm_history_2 = [
        temp_pwms[1][:1, ...] * sequence_masks[l_2_test[test_ix, 0]][None, None, :, None]
        for temp_pwms in pwm_history
    ]
    
    sel_scores_history_1 = [
        temp_scores[0][:1, ...] * sequence_masks[l_1_test[test_ix, 0]][None, None, :, None]
        for temp_scores in scores_history
    ]
    sel_scores_history_2 = [
        temp_scores[1][:1, ...] * sequence_masks[l_2_test[test_ix, 0]][None, None, :, None]
        for temp_scores in scores_history
    ]

    sel_nll_loss_history = [
        temp_nll_loss[:1, ...] * (-1. if is_occlusion else 1.)
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

    f, (ax1, ax_bg_1, ax2, ax3, ax4, ax_bg_2, ax5, ax6, ax7) = plt.subplots(1 + 6 + 2, 1, figsize=(9, 9), gridspec_kw={'width_ratios': [1], 'height_ratios': [5, 0.1, 1, 1, 1, 0.1, 1, 1, 1]})

    #Plot PWMs
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax5.axis('off')
    ax6.axis('off')
    ax7.axis('off')
    
    ax2.get_xaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax4.get_xaxis().set_visible(False)
    ax5.get_xaxis().set_visible(False)
    ax6.get_xaxis().set_visible(False)
    ax7.get_xaxis().set_visible(False)
    
    ax2.get_yaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)
    ax7.get_yaxis().set_visible(False)
    
    ax_bg_1.axis('off')
    ax_bg_1.get_xaxis().set_visible(False)
    ax_bg_1.get_yaxis().set_visible(False)
    
    ax_bg_2.axis('off')
    ax_bg_2.get_xaxis().set_visible(False)
    ax_bg_2.get_yaxis().set_visible(False)
    
    q_1 = np.quantile(sel_scores_history_1[-1][0, 0, :, :], q=score_quantile)
    q_2 = np.quantile(sel_scores_history_2[-1][0, 0, :, :], q=score_quantile)
    
    seq_1 = encoder.decode(x_1_test[test_ix, 0, :, :])[:l_1_test[test_ix, 0]]
    seq_2 = encoder.decode(x_2_test[test_ix, 0, :, :])[:l_2_test[test_ix, 0]]

    _plot_protein_logo_on_axis(ax2, "Binder 1", residue_map, x_1_test[test_ix, 0, :, :], sequence_template=sequence_template_1.replace('#', '@'), color_reference=['red'], sequence_colors=np.zeros(81, dtype=np.int).tolist(), plot_start=0, plot_end=76)
    _plot_protein_logo_on_axis(ax3, "PSSM 1", residue_map, sel_pwm_history_1[-1][0, 0, :, :], sequence_template=sequence_template_1.replace('#', '@'), color_reference=['red'], sequence_colors=np.zeros(81, dtype=np.int).tolist(), plot_start=0, plot_end=76)
    _plot_protein_importance_scores_on_axis(ax4, "Scores 1", sel_scores_history_1[-1][0, 0, :, :].T, seq_1, score_clip=q_1, score_max=q_1 if not normalize_scores else None, sequence_template=sequence_template_1, single_color='red', fixed_sequence_template_scores=False, plot_start=0, plot_end=76)

    _plot_protein_logo_on_axis(ax5, "Binder 2", residue_map, x_2_test[test_ix, 0, :, :], sequence_template=sequence_template_2.replace('#', '@'), color_reference=['red'], sequence_colors=np.zeros(81, dtype=np.int).tolist(), plot_start=0, plot_end=76)
    _plot_protein_logo_on_axis(ax6, "PSSM 2", residue_map, sel_pwm_history_2[-1][0, 0, :, :], sequence_template=sequence_template_2.replace('#', '@'), color_reference=['red'], sequence_colors=np.zeros(81, dtype=np.int).tolist(), plot_start=0, plot_end=76)
    _plot_protein_importance_scores_on_axis(ax7, "Scores 2", sel_scores_history_2[-1][0, 0, :, :].T, seq_2, score_clip=q_2, score_max=q_2 if not normalize_scores else None, sequence_template=sequence_template_2, single_color='red', fixed_sequence_template_scores=False, plot_start=0, plot_end=76)
    
    loss_lines = []
    for i in range(n_examples) :
        line, = ax1.plot([], [], linewidth=3)
        loss_lines.append(line)

    plt.sca(ax1)
    plt.ylabel("Reconstruction Loss", fontsize=14)

    plt.xticks([0, batch_history[n_frames-1]], [0, batch_history[n_frames-1]], fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlim(0, batch_history[n_frames-1])
    
    if flip_loss :
        plt.ylim(0., max_nll_loss + 0.02 * max_nll_loss * np.sign(max_nll_loss))
    else :
        plt.ylim(min_nll_loss - 0.02 * min_nll_loss * np.sign(min_nll_loss), max_nll_loss + 0.02 * max_nll_loss * np.sign(max_nll_loss))

    plt.title("Weight Update 0\n1x Speedup >", fontsize=14)

    #plt.tight_layout()
    #plt.subplots_adjust(wspace=0.15)
    
    plt.subplots_adjust(wspace=0.15, hspace=0.5)
    
    #plt.show()
    
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
        ax3.clear()
        ax4.clear()
        ax5.clear()
        ax6.clear()
        ax7.clear()
        
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')
        ax5.axis('off')
        ax6.axis('off')
        ax7.axis('off')
        
        if normalize_scores :
            q_1 = np.quantile(sel_scores_history_1[t][0, 0, :, :], q=score_quantile)
            q_2 = np.quantile(sel_scores_history_2[t][0, 0, :, :], q=score_quantile)

        _plot_protein_logo_on_axis(ax2, "Binder 1", residue_map, x_1_test[test_ix, 0, :, :], sequence_template=sequence_template_1.replace('#', '@'), color_reference=['red'], sequence_colors=np.zeros(81, dtype=np.int).tolist(), plot_start=0, plot_end=76)
        _plot_protein_logo_on_axis(ax3, "PSSM 1", residue_map, sel_pwm_history_1[t][0, 0, :, :], sequence_template=sequence_template_1.replace('#', '@'), color_reference=['red'], sequence_colors=np.zeros(81, dtype=np.int).tolist(), plot_start=0, plot_end=76)
        _plot_protein_importance_scores_on_axis(ax4, "Scores 1", sel_scores_history_1[t][0, 0, :, :].T, seq_1, score_clip=q_1, score_max=q_1 if not normalize_scores else None, sequence_template=sequence_template_1, single_color='red', fixed_sequence_template_scores=False, plot_start=0, plot_end=76)

        _plot_protein_logo_on_axis(ax5, "Binder 2", residue_map, x_2_test[test_ix, 0, :, :], sequence_template=sequence_template_2.replace('#', '@'), color_reference=['red'], sequence_colors=np.zeros(81, dtype=np.int).tolist(), plot_start=0, plot_end=76)
        _plot_protein_logo_on_axis(ax6, "PSSM 2", residue_map, sel_pwm_history_2[t][0, 0, :, :], sequence_template=sequence_template_2.replace('#', '@'), color_reference=['red'], sequence_colors=np.zeros(81, dtype=np.int).tolist(), plot_start=0, plot_end=76)
        _plot_protein_importance_scores_on_axis(ax7, "Scores 2", sel_scores_history_2[t][0, 0, :, :].T, seq_2, score_clip=q_2, score_max=q_2 if not normalize_scores else None, sequence_template=sequence_template_2, single_color='red', fixed_sequence_template_scores=False, plot_start=0, plot_end=76)
        
        return []


    anim = FuncAnimation(f, animate, init_func=init, frames=n_frames+1, interval=50, blit=True)

    anim.save(model_name + ('_normalized_scores' if normalize_scores else '') + '.gif', writer='imagemagick')
    
    return
