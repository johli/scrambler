import keras
from keras.models import Sequential, Model, load_model

from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D, GRU, CuDNNLSTM, CuDNNGRU, BatchNormalization, LocallyConnected2D, Permute, TimeDistributed, Bidirectional
from keras.layers import Concatenate, Reshape, Softmax, Conv2DTranspose, Embedding, Multiply
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras import regularizers
from keras import backend as K
from keras.utils.generic_utils import Progbar
from keras.layers.merge import _Merge
import keras.losses

from functools import partial

from collections import defaultdict

import tensorflow as tf
from tensorflow.python.framework import ops

import isolearn.keras as iso

import numpy as np

import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import os
import pickle
import numpy as np
import pandas as pd

import scipy.sparse as sp
import scipy.io as spio

import matplotlib.pyplot as plt

from keras.backend.tensorflow_backend import set_session

from scipy.signal import gaussian

def contain_tf_gpu_mem_usage() :
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

contain_tf_gpu_mem_usage()

class EpochVariableCallback(Callback) :
    
    def __init__(self, my_variable, my_func) :
        self.my_variable = my_variable       
        self.my_func = my_func
        
    def on_epoch_begin(self, epoch, logs={}) :
        K.set_value(self.my_variable, self.my_func(K.get_value(self.my_variable), epoch))

from tensorflow.python.framework import ops


from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints

class InstanceNormalization(Layer):
    def __init__(self, axes=(1, 2), trainable=True, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axes = axes
        self.trainable = trainable
    def build(self, input_shape):
        self.beta  = self.add_weight(name='beta',shape=(input_shape[-1],),
                                     initializer='zeros',trainable=self.trainable)
        self.gamma = self.add_weight(name='gamma',shape=(input_shape[-1],),
                                     initializer='ones',trainable=self.trainable)
    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, self.axes, keep_dims=True)
        return tf.nn.batch_normalization(inputs, mean, variance, self.beta, self.gamma, 1e-6)

#Stochastic Binarized Neuron helper functions (Tensorflow)
#ST Estimator code adopted from https://r2rt.com/beyond-binary-ternary-and-one-hot-neurons.html
#See Github https://github.com/spitis/

def st_sampled_softmax(logits):
    with ops.name_scope("STSampledSoftmax") as namescope :
        nt_probs = tf.nn.softmax(logits)
        onehot_dim = logits.get_shape().as_list()[1]
        sampled_onehot = tf.one_hot(tf.squeeze(tf.multinomial(logits, 1), 1), onehot_dim, 1.0, 0.0)
        with tf.get_default_graph().gradient_override_map({'Ceil': 'Identity', 'Mul': 'STMul'}):
            return tf.ceil(sampled_onehot * nt_probs)

def st_hardmax_softmax(logits):
    with ops.name_scope("STHardmaxSoftmax") as namescope :
        nt_probs = tf.nn.softmax(logits)
        onehot_dim = logits.get_shape().as_list()[1]
        sampled_onehot = tf.one_hot(tf.argmax(nt_probs, 1), onehot_dim, 1.0, 0.0)
        with tf.get_default_graph().gradient_override_map({'Ceil': 'Identity', 'Mul': 'STMul'}):
            return tf.ceil(sampled_onehot * nt_probs)

@ops.RegisterGradient("STMul")
def st_mul(op, grad):
    return [grad, grad]

#Gumbel Distribution Sampler
def gumbel_softmax(logits, temperature=0.5) :
    gumbel_dist = tf.contrib.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
    batch_dim = logits.get_shape().as_list()[0]
    onehot_dim = logits.get_shape().as_list()[1]
    return gumbel_dist.sample()

#PWM Masking and Sampling helper functions

def mask_pwm(inputs) :
    pwm, onehot_template, onehot_mask = inputs

    return pwm * onehot_mask + onehot_template

def sample_pwm_st(pwm_logits, n_channels=4, temperature=None) :
    n_examples = K.shape(pwm_logits)[0]
    input_size_x = K.shape(pwm_logits)[1]
    input_size_y = K.shape(pwm_logits)[2]

    flat_pwm = K.reshape(pwm_logits, (n_examples * input_size_x * input_size_y, n_channels))
    sampled_pwm = st_sampled_softmax(flat_pwm)

    return K.reshape(sampled_pwm, (n_examples, input_size_x, input_size_y, n_channels))

def sample_pwm_gumbel(pwm_logits, n_channels=4, temperature=0.5) :
    n_examples = K.shape(pwm_logits)[0]
    input_size_x = K.shape(pwm_logits)[1]
    input_size_y = K.shape(pwm_logits)[2]

    flat_pwm = K.reshape(pwm_logits, (n_examples * input_size_x * input_size_y, n_channels))
    sampled_pwm = gumbel_softmax(flat_pwm, temperature=temperature)

    return K.reshape(sampled_pwm, (n_examples, input_size_x, input_size_y, n_channels))

#Generator helper functions
def initialize_templates(model, template_matrices, background_matrices, model_prefix='') :

    n_channels = template_matrices[0].shape[-1]
    
    embedding_templates = []
    embedding_masks = []
    embedding_backgrounds = []

    for k in range(len(template_matrices)) :
        onehot_template = template_matrices[k]
        onehot_template_log = np.zeros(onehot_template.shape)

        for i in range(onehot_template.shape[0]) :
            for j in range(onehot_template.shape[1]) :
                if np.sum(onehot_template[i, j, :]) >= 1. :
                    channel_ix = np.argmax(onehot_template[i, j, :])
                    onehot_template_log[i, j, :] = -4.0
                    onehot_template_log[i, j, channel_ix] = 10.0

        onehot_mask = np.zeros(onehot_template.shape)
        for i in range(onehot_template.shape[0]) :
            for j in range(onehot_template.shape[1]) :
                if np.sum(onehot_template[i, j, :]) <= 0. :
                    onehot_mask[i, j, :] = 1.0

        embedding_templates.append(onehot_template_log.reshape(1, -1))
        embedding_masks.append(onehot_mask.reshape(1, -1))
        embedding_backgrounds.append(background_matrices[k].reshape(1, -1))

    embedding_templates = np.concatenate(embedding_templates, axis=0)
    embedding_masks = np.concatenate(embedding_masks, axis=0)
    embedding_backgrounds = np.concatenate(embedding_backgrounds, axis=0)

    model.get_layer(model_prefix + 'template_dense').set_weights([embedding_templates])
    model.get_layer(model_prefix + 'template_dense').trainable = False

    model.get_layer(model_prefix + 'mask_dense').set_weights([embedding_masks])
    model.get_layer(model_prefix + 'mask_dense').trainable = False
    
    model.get_layer(model_prefix + 'background_dense').set_weights([embedding_backgrounds])
    model.get_layer(model_prefix + 'background_dense').trainable = False

#Generator construction function
def build_sampler(batch_size, input_size_x, input_size_y, n_classes=1, n_samples=1, sample_mode='st', n_channels=4, gumbel_temp=0.5, model_prefix='') :

    #Initialize Reshape layer
    reshape_layer = Reshape((input_size_x, input_size_y, n_channels))
    
    #Initialize background matrix
    onehot_background_dense = Embedding(n_classes, input_size_x * input_size_y * n_channels, embeddings_initializer='zeros', name=model_prefix + 'background_dense')

    #Initialize template and mask matrices
    onehot_template_dense = Embedding(n_classes, input_size_x * input_size_y * n_channels, embeddings_initializer='zeros', name=model_prefix + 'template_dense')
    onehot_mask_dense = Embedding(n_classes, input_size_x * input_size_y * n_channels, embeddings_initializer='ones', name=model_prefix + 'mask_dense')

    #Initialize Templating and Masking Lambda layer
    masking_layer = Lambda(mask_pwm, output_shape = (input_size_x, input_size_y, n_channels), name=model_prefix + 'masking_layer')
    background_layer = Lambda(lambda x: x[0] + x[1], name=model_prefix + 'background_layer')
    
    #Initialize PWM normalization layer
    pwm_layer = Softmax(axis=-1, name=model_prefix + 'pwm')
    
    #Initialize sampling layers
    sample_func = None
    if sample_mode == 'st' :
        sample_func = sample_pwm_st
    elif sample_mode == 'gumbel' :
        sample_func = sample_pwm_gumbel
    
    upsampling_layer = Lambda(lambda x: K.tile(x, [n_samples, 1, 1, 1]), name=model_prefix + 'upsampling_layer')
    sampling_layer = Lambda(lambda x: sample_func(x, n_channels=n_channels, temperature=gumbel_temp), name=model_prefix + 'pwm_sampler')
    permute_layer = Lambda(lambda x: K.permute_dimensions(K.reshape(x, (n_samples, batch_size, input_size_x, input_size_y, n_channels)), (1, 0, 2, 3, 4)), name=model_prefix + 'permute_layer')
    
    def _sampler_func(class_input, raw_logits) :
        
        #Get Template and Mask
        onehot_background = reshape_layer(onehot_background_dense(class_input))
        onehot_template = reshape_layer(onehot_template_dense(class_input))
        onehot_mask = reshape_layer(onehot_mask_dense(class_input))
        
        #Add Template and Multiply Mask
        pwm_logits = masking_layer([background_layer([raw_logits, onehot_background]), onehot_template, onehot_mask])
        
        #Compute PWM (Nucleotide-wise Softmax)
        pwm = pwm_layer(pwm_logits)
        
        #Tile each PWM to sample from and create sample axis
        pwm_logits_upsampled = upsampling_layer(pwm_logits)
        sampled_pwm = sampling_layer(pwm_logits_upsampled)
        sampled_pwm = permute_layer(sampled_pwm)

        sampled_mask = permute_layer(upsampling_layer(onehot_mask))
        
        return pwm_logits, pwm, sampled_pwm, onehot_mask, sampled_mask
    
    return _sampler_func

#Scrambler network definition

def make_resblock(n_channels=64, window_size=8, dilation_rate=1, group_ix=0, layer_ix=0, drop_rate=0.0, norm_mode='instance') :

    #Initialize res block layers
    batch_norm_0 = lambda x: x
    if norm_mode == 'instance' :
        batch_norm_0 = InstanceNormalization(name='scrambler_resblock_' + str(group_ix) + '_' + str(layer_ix) + '_norm_0')
    elif norm_mode == 'batch' :
        batch_norm_0 = BatchNormalization(name='scrambler_resblock_' + str(group_ix) + '_' + str(layer_ix) + '_norm_0')

    relu_0 = Lambda(lambda x: K.relu(x, alpha=0.0))

    conv_0 = Conv2D(n_channels, window_size, dilation_rate=dilation_rate, strides=(1, 1), padding='same', activation='linear', kernel_initializer='glorot_normal', name='scrambler_resblock_' + str(group_ix) + '_' + str(layer_ix) + '_conv_0')

    batch_norm_1 = lambda x: x
    if norm_mode == 'instance' :
        batch_norm_1 = InstanceNormalization(name='scrambler_resblock_' + str(group_ix) + '_' + str(layer_ix) + '_norm_1')
    elif norm_mode == 'batch' :
        batch_norm_1 = BatchNormalization(name='scrambler_resblock_' + str(group_ix) + '_' + str(layer_ix) + '_norm_1')

    relu_1 = Lambda(lambda x: K.relu(x, alpha=0.0))

    conv_1 = Conv2D(n_channels, window_size, dilation_rate=dilation_rate, strides=(1, 1), padding='same', activation='linear', kernel_initializer='glorot_normal', name='scrambler_resblock_' + str(group_ix) + '_' + str(layer_ix) + '_conv_1')

    skip_1 = Lambda(lambda x: x[0] + x[1], name='scrambler_resblock_' + str(group_ix) + '_' + str(layer_ix) + '_skip_1')

    drop_1 = None
    if drop_rate > 0.0 :
        drop_1 = Dropout(drop_rate)
    
    #Execute res block
    def _resblock_func(input_tensor) :
        batch_norm_0_out = batch_norm_0(input_tensor)
        relu_0_out = relu_0(batch_norm_0_out)
        conv_0_out = conv_0(relu_0_out)

        batch_norm_1_out = batch_norm_1(conv_0_out)
        relu_1_out = relu_1(batch_norm_1_out)
        
        if drop_rate > 0.0 :
            conv_1_out = drop_1(conv_1(relu_1_out))
        else :
            conv_1_out = conv_1(relu_1_out)

        skip_1_out = skip_1([conv_1_out, input_tensor])
        
        return skip_1_out

    return _resblock_func

def mask_dropout_multi_scale(mask, n_spatial_dims=1, drop_scales=[1, 2, 4, 7], min_drop_rate=0.0, max_drop_rate=0.5) :
    
    rates = K.random_uniform(shape=(K.shape(mask)[0], 1, 1, 1), minval=min_drop_rate, maxval=max_drop_rate)
    
    scale_logits = K.random_uniform(shape=(K.shape(mask)[0], len(drop_scales), 1, 1, 1), minval=-5., maxval=5.)
    scale_probs = K.softmax(scale_logits, axis=1)
    
    ret_mask = mask
    for drop_scale_ix, drop_scale in enumerate(drop_scales) :
        ret_mask = mask_dropout(ret_mask, rates * scale_probs[:, drop_scale_ix, ...], drop_scale=drop_scale, n_spatial_dims=n_spatial_dims)
    
    return K.switch(K.learning_phase(), ret_mask, mask)

def mask_dropout(mask, drop_rates, drop_scale=1, n_spatial_dims=1) :
    
    random_tensor_downsampled = K.random_uniform(shape=(
        K.shape(mask)[0],
        1 if n_spatial_dims == 1 else K.cast(K.shape(mask)[1] / drop_scale, dtype=tf.int32),
        K.cast(K.shape(mask)[2] / drop_scale, dtype=tf.int32),
        K.shape(mask)[3]
    ), minval=0.0, maxval=1.0)
    
    keep_mask_downsampled = random_tensor_downsampled >= drop_rates
    
    keep_mask = K.repeat_elements(keep_mask_downsampled, rep=drop_scale, axis=2)
    if n_spatial_dims > 1 :
        keep_mask = K.repeat_elements(keep_mask, rep=drop_scale, axis=1)
    
    ret_mask = mask * K.cast(keep_mask, dtype=tf.float32)
    
    return ret_mask

def mask_dropout_single_scale(mask, n_spatial_dims=1, drop_scale=1, min_drop_rate=0.0, max_drop_rate=0.5) :
    
    rates = K.random_uniform(shape=(K.shape(mask)[0], 1, 1, 1), minval=min_drop_rate, maxval=max_drop_rate)
    
    random_tensor_downsampled = K.random_uniform(shape=(
        K.shape(mask)[0],
        1 if n_spatial_dims == 1 else K.cast(K.shape(mask)[1] / drop_scale, dtype=tf.int32),
        K.cast(K.shape(mask)[2] / drop_scale, dtype=tf.int32),
        K.shape(mask)[3]
    ), minval=0.0, maxval=1.0)
    
    keep_mask_downsampled = random_tensor_downsampled >= rates
    
    keep_mask = K.repeat_elements(keep_mask_downsampled, rep=drop_scale, axis=2)
    if n_spatial_dims > 1 :
        keep_mask = K.repeat_elements(keep_mask, rep=drop_scale, axis=1)
    
    ret_mask = mask * K.cast(keep_mask, dtype=tf.float32)
    
    return K.switch(K.learning_phase(), ret_mask, mask)

def load_scrambler_network(input_size_x, input_size_y, scrambler_mode='inclusion', n_out_channels=4, n_spatial_dims=1, n_groups=1, n_resblocks_per_group=4, n_channels=32, window_size=8, mask_smoothing=False, smooth_window_size=None, dilation_rates=[1], drop_rate=0.0, norm_mode='instance', mask_dropout=False, mask_drop_scales=[1, 5], mask_min_drop_rate=0.0, mask_max_drop_rate=0.5, use_label_input=False) :

    conv_0 = Conv2D(n_channels, (1, 1), strides=(1, 1), padding='same', activation='linear', kernel_initializer='glorot_normal', name='scrambler_conv_0')
    
    label_concat = None
    if use_label_input :
        label_concat = Lambda(lambda x: K.concatenate([x[0], K.tile(K.expand_dims(K.expand_dims(x[1], axis=-1), axis=-1), (1, K.shape(x[0])[1], K.shape(x[0])[2], 1))], axis=-1))
    
    mask_drop = None
    mask_concat = None
    mask_multiply = None
    if mask_dropout :
        if len(mask_drop_scales) <= 1 :
            mask_drop = Lambda(lambda x: mask_dropout_single_scale(x, drop_scale=mask_drop_scales[0], min_drop_rate=mask_min_drop_rate, max_drop_rate=mask_max_drop_rate), output_shape=(1, input_size_y, 1) if n_spatial_dims == 1 else (input_size_x, input_size_y, 1), name='scrambler_mask_drop')
        else :
            mask_drop = Lambda(lambda x: mask_dropout_multi_scale(x, drop_scales=mask_drop_scales, min_drop_rate=mask_min_drop_rate, max_drop_rate=mask_max_drop_rate), output_shape=(1, input_size_y, 1) if n_spatial_dims == 1 else (input_size_x, input_size_y, 1), name='scrambler_mask_drop')
        
        mask_concat = Concatenate(axis=-1)
        mask_multiply = Lambda(lambda x: x[0] * x[1])
    
    skip_convs = []
    resblock_groups = []
    for group_ix in range(n_groups) :
        
        skip_convs.append(Conv2D(n_channels, (1, 1), strides=(1, 1), padding='same', activation='linear', kernel_initializer='glorot_normal', name='scrambler_skip_conv_' + str(group_ix)))
        
        resblocks = []
        for layer_ix in range(n_resblocks_per_group) :
            resblocks.append(make_resblock(n_channels=n_channels, window_size=(1, window_size) if n_spatial_dims == 1 else (window_size, window_size), dilation_rate=dilation_rates[group_ix], group_ix=group_ix, layer_ix=layer_ix, drop_rate=drop_rate, norm_mode=norm_mode))
        
        resblock_groups.append(resblocks)

    last_block_conv = Conv2D(n_channels, (1, 1), strides=(1, 1), padding='same', activation='linear', kernel_initializer='glorot_normal', name='scrambler_last_block_conv')
    
    skip_add = Lambda(lambda x: x[0] + x[1], name='scrambler_skip_add')
    
    final_conv = Conv2D(1, (1, 1), strides=(1, 1), padding='same', activation='softplus', kernel_initializer='glorot_normal', name='scrambler_final_conv')
    
    smooth_conv = None
    if mask_smoothing :
        smooth_conv = Conv2D(1, (1, smooth_window_size) if n_spatial_dims == 1 else (smooth_window_size, smooth_window_size), strides=(1, 1), use_bias=False, padding='same', activation='linear', kernel_initializer='ones', name='scrambler_smooth_conv')
    
    onehot_to_logits = Lambda(lambda x: 2. * x - 1., name='scrambler_onehot_to_logits')
    
    scale_logits = Lambda(lambda x: x[1] * K.tile(x[0], (1, 1, 1, n_out_channels)), name='scrambler_logit_scale')
    if scrambler_mode == 'occlusion' :
        scale_logits = Lambda(lambda x: x[1] / K.maximum(K.tile(x[0], (1, 1, 1, n_out_channels)), K.epsilon()), name='scrambler_logit_scale')
    
    def _scrambler_func(example_input, mask_input=None, label_input=None) :
        
        total_input = example_input
        if use_label_input :
            total_input = label_concat([total_input, label_input])
        if mask_dropout :
            mask_dropped = mask_drop(mask_input)
            total_input = mask_concat([total_input, mask_dropped])
        
        conv_0_out = conv_0(total_input)

        #Connect group of res blocks
        output_tensor = conv_0_out

        #Res block group execution
        skip_conv_outs = []
        for group_ix in range(n_groups) :
            skip_conv_out = skip_convs[group_ix](output_tensor)
            skip_conv_outs.append(skip_conv_out)

            for layer_ix in range(n_resblocks_per_group) :
                output_tensor = resblock_groups[group_ix][layer_ix](output_tensor)
        
        #Last res block extr conv
        last_block_conv_out = last_block_conv(output_tensor)

        skip_add_out = last_block_conv_out
        for group_ix in range(n_groups) :
            skip_add_out = skip_add([skip_add_out, skip_conv_outs[group_ix]])

        #Final conv out
        final_conv_out = final_conv(skip_add_out)
        
        if mask_dropout :
            final_conv_out = mask_multiply([final_conv_out, mask_dropped])
        
        if mask_smoothing :
            final_conv_out = smooth_conv(final_conv_out)
        
        #Scale logits by importance scores
        scaled_logits = scale_logits([final_conv_out, onehot_to_logits(example_input)])
        
        return scaled_logits, final_conv_out

    return _scrambler_func

def load_finetuning_model(batch_size, input_size_x, input_size_y, scrambler_mode='inclusion', n_out_channels=4, n_spatial_dims=1, mask_smoothing=False, smooth_window_size=None, norm_mode='instance', max_score_clip=4.) :

    #seed_input = Lambda(lambda x: K.zeros((K.shape(x)[0], 1), dtype=tf.int32))
    seed_input = Lambda(lambda x: K.constant(np.arange(batch_size), dtype=tf.int32))
    
    mask_dense = Embedding(batch_size, input_size_x * input_size_y, embeddings_initializer='glorot_normal', name='ft_scrambler_mask_dense')
    
    mask_reshape = Reshape((input_size_x, input_size_y, 1))
    
    mask_conv, mask_norm = None, None
    if norm_mode is not None and norm_mode == 'conv' :
        mask_conv = Conv2D(1, (1, 1), strides=(1, 1), padding='same', activation='linear', kernel_initializer='glorot_normal', name='ft_scrambler_mask_conv')
    elif norm_mode is not None and norm_mode == 'instance' :
        mask_norm = InstanceNormalization(name='ft_scrambler_mask_norm')
    
    mask_act = Activation('softplus')
    
    smooth_conv = None
    if mask_smoothing :
        smooth_conv = Conv2D(1, (1, smooth_window_size) if n_spatial_dims == 1 else (smooth_window_size, smooth_window_size), strides=(1, 1), use_bias=False, padding='same', activation='linear', kernel_initializer='ones', name='ft_scrambler_smooth_conv')
    
    onehot_to_logits = Lambda(lambda x: 2. * x - 1., name='ft_scrambler_onehot_to_logits')
    
    scale_logits = Lambda(lambda x: x[1] * K.tile(x[0], (1, 1, 1, n_out_channels)), name='ft_scrambler_logit_scale')
    if scrambler_mode == 'occlusion' :
        scale_logits = Lambda(lambda x: x[1] / K.maximum(K.tile(x[0], (1, 1, 1, n_out_channels)), K.epsilon()), name='ft_scrambler_logit_scale')
    
    clip_scores = Lambda(lambda x, max_score_clip=max_score_clip: K.relu(K.clip(x[1], 0., max_score_clip) - x[0]), name='ft_scrambler_clip_scores')
    
    drop_multiply = Lambda(lambda x: x[0] * x[1], name='ft_scrambler_drop_multiply')
    
    def _scrambler_func(sequence_input, drop_input, pretrained_scores) :

        mask_in = mask_reshape(mask_dense(seed_input(sequence_input)))
        
        #Final conv out
        if norm_mode is not None and norm_mode == 'conv' : 
            mask_out = mask_conv(mask_in)
        elif norm_mode is not None and norm_mode == 'instance' :
            mask_out = mask_norm(mask_in)
            #mask_out = mask_norm(mask_in, training=True)
        else :
            mask_out = mask_in
        
        mask_act_out = mask_act(mask_out)
        
        scores_out = drop_multiply([clip_scores([mask_act_out, pretrained_scores]), drop_input])
        
        if mask_smoothing :
            scores_out = smooth_conv(scores_out)
        
        #Scale inputs by importance scores
        scaled_inputs = scale_logits([scores_out, onehot_to_logits(sequence_input)])
        
        return scaled_inputs, scores_out

    return _scrambler_func

def load_optimization_model(batch_size, input_size_x, input_size_y, scrambler_mode='inclusion', n_out_channels=4, n_spatial_dims=1, mask_smoothing=False, smooth_window_size=None, norm_mode='instance') :

    #seed_input = Lambda(lambda x: K.zeros((K.shape(x)[0], 1), dtype=tf.int32))
    seed_input = Lambda(lambda x: K.constant(np.arange(batch_size), dtype=tf.int32))
    
    mask_dense = Embedding(batch_size, input_size_x * input_size_y, embeddings_initializer='glorot_normal', name='ot_scrambler_mask_dense')
    
    mask_reshape = Reshape((input_size_x, input_size_y, 1))
    
    mask_conv, mask_norm = None, None
    if norm_mode is not None and norm_mode == 'conv' :
        mask_conv = Conv2D(1, (1, 1), strides=(1, 1), padding='same', activation='linear', kernel_initializer='glorot_normal', name='ot_scrambler_mask_conv')
    elif norm_mode is not None and norm_mode == 'instance' :
        mask_norm = InstanceNormalization(name='ot_scrambler_mask_norm')
    
    mask_act = Activation('softplus')
    
    smooth_conv = None
    if mask_smoothing :
        smooth_conv = Conv2D(1, (1, smooth_window_size) if n_spatial_dims == 1 else (smooth_window_size, smooth_window_size), strides=(1, 1), use_bias=False, padding='same', activation='linear', kernel_initializer='ones', name='ot_scrambler_smooth_conv')
    
    onehot_to_logits = Lambda(lambda x: 2. * x - 1., name='ot_scrambler_onehot_to_logits')
    
    scale_logits = Lambda(lambda x: x[1] * K.tile(x[0], (1, 1, 1, n_out_channels)), name='ot_scrambler_logit_scale')
    if scrambler_mode == 'occlusion' :
        scale_logits = Lambda(lambda x: x[1] / K.maximum(K.tile(x[0], (1, 1, 1, n_out_channels)), K.epsilon()), name='ot_scrambler_logit_scale')
    
    drop_multiply = Lambda(lambda x: x[0] * x[1], name='ot_scrambler_drop_multiply')
    
    def _scrambler_func(sequence_input, drop_input) :

        mask_in = mask_reshape(mask_dense(seed_input(sequence_input)))
        
        #Final conv out
        if norm_mode is not None and norm_mode == 'conv' : 
            mask_out = mask_conv(mask_in)
        elif norm_mode is not None and norm_mode == 'instance' :
            mask_out = mask_norm(mask_in)
            #mask_out = mask_norm(mask_in, training=True)
        else :
            mask_out = mask_in
        
        mask_act_out = mask_act(mask_out)
        
        scores_out = drop_multiply([mask_act_out, drop_input])
        
        if mask_smoothing :
            scores_out = smooth_conv(scores_out)
        
        #Scale inputs by importance scores
        scaled_inputs = scale_logits([scores_out, onehot_to_logits(sequence_input)])
        
        return scaled_inputs, scores_out

    return _scrambler_func

#Keras loss functions

def get_mse() :
    
    def _mse(y_true, y_pred) :
        return K.mean((y_true[..., 0] - y_pred[..., 0])**2, axis=-1)
    
    return _mse

def get_linear_max_nll() :
    
    def _mse(y_true, y_pred) :
        return -K.mean(y_pred[..., 0], axis=-1)
    
    return _mse

def get_linear_min_nll() :
    
    def _mse(y_true, y_pred) :
        return K.mean(y_pred[..., 0], axis=-1)
    
    return _mse

def get_softmax_kl_divergence() :

    def _softmax_kl_divergence(y_true, y_pred) :

        y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        return K.mean(K.sum(y_true * K.log(y_true / y_pred), axis=-1), axis=-1)
    
    return _softmax_kl_divergence

def get_sigmoid_kl_divergence() :

    def _kl_divergence(y_true, y_pred) :

        y_true = K.clip(y_true[..., 0], K.epsilon(), 1.0 - K.epsilon())
        y_pred = K.clip(y_pred[..., 0], K.epsilon(), 1.0 - K.epsilon())

        return K.mean(y_true * K.log(y_true / y_pred) + (1.0 - y_true) * K.log((1.0 - y_true) / (1.0 - y_pred)), axis=-1)
    
    return _kl_divergence

def get_symmetric_sigmoid_kl_divergence() :

    def _kl_divergence(y_true, y_pred) :

        y_pred = K.clip(y_pred[..., 0], K.epsilon(), 1.0 - K.epsilon())
        y_true = K.clip(y_true[..., 0], K.epsilon(), 1.0 - K.epsilon())
        
        left_mean_kl = K.mean(y_true * K.log(y_true / y_pred) + (1.0 - y_true) * K.log((1.0 - y_true) / (1.0 - y_pred)), axis=-1)
        right_mean_kl = K.mean(y_pred * K.log(y_pred / y_true) + (1.0 - y_pred) * K.log((1.0 - y_pred) / (1.0 - y_true)), axis=-1)

        return left_mean_kl + right_mean_kl
    
    return _kl_divergence

def get_sigmoid_max_nll() :

    def _max_nll(y_pred) :

        y_pred = K.clip(y_pred[..., 0], K.epsilon(), 1.0)
        
        return K.mean(-K.log(y_pred), axis=-1)
    
    return _max_nll

def get_sigmoid_min_nll() :

    def _min_nll(y_pred) :

        y_pred = K.clip(y_pred[..., 0], 0.0, 1.0 - K.epsilon())
        
        return K.mean(-K.log(1.0 - y_pred), axis=-1)
    
    return _min_nll

def get_margin_entropy_ame_masked(x_start, x_end, y_start, y_end, max_bits=1.0) :
    
    def _margin_entropy_ame_masked(pwm, pwm_mask, pwm_background) :
        conservation = pwm[:, x_start:x_end, y_start:y_end, :] * K.log(K.clip(pwm[:, x_start:x_end, y_start:y_end, :], K.epsilon(), 1. - K.epsilon()) / pwm_background[:, x_start:x_end, y_start:y_end, :]) / K.log(2.0)
        conservation = K.sum(conservation, axis=-1)
        
        mask = K.max(pwm_mask[:, x_start:x_end, y_start:y_end, :], axis=-1)
        n_unmasked = K.sum(mask, axis=(1, 2))
        
        mean_conservation = K.sum(conservation * mask, axis=(1, 2)) / n_unmasked

        margin_conservation = K.switch(mean_conservation > K.constant(max_bits, shape=(1,)), mean_conservation - K.constant(max_bits, shape=(1,)), K.zeros_like(mean_conservation))
    
        return margin_conservation
    
    return _margin_entropy_ame_masked

def get_target_entropy_sme_masked(x_start, x_end, y_start, y_end, target_bits=1.0) :
    
    def _target_entropy_sme_masked(pwm, pwm_mask, pwm_background) :
        conservation = pwm[:, x_start:x_end, y_start:y_end, :] * K.log(K.clip(pwm[:, x_start:x_end, y_start:y_end, :], K.epsilon(), 1. - K.epsilon()) / pwm_background[:, x_start:x_end, y_start:y_end, :]) / K.log(2.0)
        conservation = K.sum(conservation, axis=-1)
        
        mask = K.max(pwm_mask[:, x_start:x_end, y_start:y_end, :], axis=-1)
        n_unmasked = K.sum(mask, axis=(1, 2))
        
        mean_conservation = K.sum(conservation * mask, axis=(1, 2)) / n_unmasked

        return (mean_conservation - target_bits)**2
    
    return _target_entropy_sme_masked

def get_weighted_loss(loss_coeff=1.) :
    
    def _min_pred(y_true, y_pred) :
        return loss_coeff * y_pred
    
    return _min_pred

class ScramblerMonitor(Callback):
    def __init__(self, scrambler_model, loss_model, loss_tensors, input_tensors, n_inputs=1, track_mode='batch', batch_freq_dict=None, batch_size=32) :
        
        self.scrambler_model = scrambler_model
        self.loss_model = loss_model
        self.track_mode = track_mode
        self.batch_freq_dict = batch_freq_dict
        self.batch_size = batch_size
        
        self.loss_tensors = loss_tensors
        self.input_tensors = input_tensors
        self.n_inputs = n_inputs
        
        self.batch_history = []
        self.epoch_history = []
        self.nll_loss_history = []
        self.entropy_loss_history = []
        
        self.scores_history = []
        self.pwm_history = []

        self.n_epochs = 0
        self.n_batches = 0
        
        self.batch_freq = 10
        if self.batch_freq_dict is not None and 0 in self.batch_freq_dict :
            self.batch_freq = self.batch_freq_dict[0]

        nll_loss, entropy_loss, scores, pwms = self._predict_vals()

        #Track metrics
        self.batch_history.append(self.n_batches)
        self.epoch_history.append(self.n_epochs)
        self.scores_history.append(scores)
        self.pwm_history.append(pwms)
        self.nll_loss_history.append(nll_loss)
        self.entropy_loss_history.append(entropy_loss)
    
    def _predict_vals(self) :
        
        nll_loss, entropy_loss = self.loss_model.predict(x=self.loss_tensors, batch_size=self.batch_size)
        pred_bundle = self.scrambler_model.predict(x=self.input_tensors, batch_size=self.batch_size)
        
        pwms = []
        scores = []
        for input_ix in range(self.n_inputs) :
            pwm = pred_bundle[self.n_inputs + input_ix]
            score = pred_bundle[3 * self.n_inputs + input_ix]
            
            pwms.append(pwm)
            scores.append(score)
        
        return nll_loss, entropy_loss, scores, pwms
    
    def on_batch_end(self, batch, logs={}) :
        self.n_batches += 1
        
        #if batch == 0 and self.batch_freq_dict is not None and self.n_epochs in self.batch_freq_dict :
        #    self.batch_freq = self.batch_freq_dict[self.n_epochs]
        if self.batch_freq_dict is not None and self.n_batches in self.batch_freq_dict :
            self.batch_freq = self.batch_freq_dict[self.n_batches]
        
        if self.track_mode == 'batch' and batch % self.batch_freq == 0 :
            nll_loss, entropy_loss, scores, pwms = self._predict_vals()

            #Track metrics
            self.batch_history.append(self.n_batches)
            self.epoch_history.append(self.n_epochs)
            self.scores_history.append(scores)
            self.pwm_history.append(pwms)
            self.nll_loss_history.append(nll_loss)
            self.entropy_loss_history.append(entropy_loss)

    def on_epoch_end(self, epoch, logs={}) :
        self.n_epochs += 1

        if self.track_mode == 'epoch' :
            nll_loss, entropy_loss, scores, pwms = self._predict_vals()

            #Track metrics
            self.epoch_history.append(self.n_epochs)
            self.scores_history.append(scores)
            self.pwm_history.append(pwms)
            self.nll_loss_history.append(nll_loss)
            self.entropy_loss_history.append(entropy_loss)

class LossHistory(keras.callbacks.Callback) :
    
    def __init__(self, loss_names=['nll', 'entropy']) :
        self.loss_names = loss_names
        self.loss_dict = {
            loss_name : []
            for loss_name in loss_names
        }

    def on_batch_end(self, batch, logs={}) :
        for loss_name in self.loss_names :
            self.loss_dict[loss_name].append(logs.get(loss_name + '_loss'))

def initialize_backgrounds(model, background_matrices, model_prefix='') :

    flat_background_matrices = []

    for k in range(len(background_matrices)) :
        flat_background_matrices.append(background_matrices[k].reshape(1, -1))

    flat_background_matrices = np.concatenate(flat_background_matrices, axis=0)

    model.get_layer(model_prefix + 'x_mean_dense').set_weights([flat_background_matrices])
    model.get_layer(model_prefix + 'x_mean_dense').trainable = False

class Scrambler :
    
    def __init__(self, n_inputs=1, n_classes=1, multi_input_mode='siamese', scrambler_mode='inclusion', input_size_x=1, input_size_y=100, n_out_channels=4, input_templates=None, input_backgrounds=None, batch_size=32, n_samples=32, sample_mode='st', zeropad_input=False, mask_dropout=False, network_config={'n_groups' : 1, 'n_resblocks_per_group' : 4, 'n_channels' : 32, 'window_size' : 8, 'dilation_rates' : [1], 'drop_rate' : 0.25, 'norm_mode' : 'instance', 'mask_smoothing' : True, 'mask_smoothing_window_size' : 7, 'mask_smoothing_std' : 1.5, 'mask_drop_scales' : [1, 5], 'mask_min_drop_rate' : 0.0, 'mask_max_drop_rate' : 0.5, 'label_input' : False}) :
        
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.multi_input_mode = multi_input_mode
        self.scrambler_mode = scrambler_mode
        self.input_size_x = input_size_x if input_size_x is not None else 1
        self.input_size_y = input_size_y
        self.n_out_channels = n_out_channels
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.sample_mode = sample_mode
        self.zeropad_input = zeropad_input
        
        self.n_groups = network_config['n_groups']
        self.n_resblocks_per_group = network_config['n_resblocks_per_group']
        self.n_channels = network_config['n_channels']
        self.window_size = network_config['window_size']
        self.dilation_rates = network_config['dilation_rates']
        self.drop_rate = network_config['drop_rate']
        self.norm_mode = network_config['norm_mode']
        
        self.mask_smoothing = network_config['mask_smoothing']
        self.mask_smoothing_window_size = network_config['mask_smoothing_window_size']
        self.mask_smoothing_std = network_config['mask_smoothing_std']
        
        self.mask_dropout = mask_dropout
        self.mask_drop_scales = network_config['mask_drop_scales']
        self.mask_min_drop_rate = network_config['mask_min_drop_rate']
        self.mask_max_drop_rate = network_config['mask_max_drop_rate']
        
        self.label_input = network_config['label_input']
        
        self.input_templates = input_templates
        if self.input_templates is None :
            self.input_templates = [np.zeros((input_size_x, input_size_y, n_channels))]
        
        self.input_backgrounds = input_backgrounds
        if self.input_backgrounds is None :
            self.input_backgrounds = [np.ones((input_size_x, input_size_y, n_channels)) * (1. / float(n_channels))]
        
        self.input_backgrounds_log = [np.log(input_background) for input_background in input_backgrounds]
        
        mask_smoothing_conv_weight = gaussian(self.mask_smoothing_window_size, self.mask_smoothing_std)
        
        #Load scrambler
        scrambler = load_scrambler_network(
            self.input_size_x,
            self.input_size_y,
            scrambler_mode=self.scrambler_mode,
            n_out_channels=self.n_out_channels,
            n_spatial_dims=1 if self.input_size_x == 1 else 2,
            n_groups=self.n_groups,
            n_resblocks_per_group=self.n_resblocks_per_group,
            n_channels=self.n_channels,
            window_size=self.window_size,
            mask_smoothing=self.mask_smoothing,
            smooth_window_size=self.mask_smoothing_window_size,
            dilation_rates=self.dilation_rates,
            drop_rate=self.drop_rate,
            norm_mode=self.norm_mode,
            mask_dropout=self.mask_dropout,
            mask_drop_scales=self.mask_drop_scales,
            mask_min_drop_rate=self.mask_min_drop_rate,
            mask_max_drop_rate=self.mask_max_drop_rate,
            use_label_input=self.label_input
        )
        
        self.scrambler = scrambler

        #Load sampler
        sampler = build_sampler(self.batch_size, self.input_size_x, self.input_size_y, n_classes=len(self.input_templates), n_samples=self.n_samples, sample_mode=self.sample_mode, n_channels=self.n_out_channels)
        
        self.sampler = sampler
        
        #Build scrambler model
        scrambler_classes = []
        scrambler_inputs = []
        scrambler_drops = []
        for input_ix in range(self.n_inputs) :
            scrambler_classes.append(Input(shape=(1,), name='scrambler_group_' + str(input_ix)))
            scrambler_inputs.append(Input(shape=(self.input_size_x, self.input_size_y, self.n_out_channels), name='scrambler_input_' + str(input_ix)))
            if self.mask_dropout :
                scrambler_drops.append(Input(shape=(self.input_size_x, self.input_size_y, 1), name='scrambler_drop_' + str(input_ix)))
            else :
                scrambler_drops.append(None)
        
        scrambler_label = None
        if self.label_input :
            scrambler_label = Input(shape=(self.n_classes,), name='scrambler_label')
        
        scrambled_logits = []
        importance_scores = []
        pwm_logits = []
        pwms = []
        sampled_pwms = []
        if self.multi_input_mode == 'siamese' or self.n_inputs == 1 :
            
            for input_ix in range(self.n_inputs) :
                
                scrambled_logit, importance_score = scrambler(scrambler_inputs[input_ix], scrambler_drops[input_ix], scrambler_label)

                scrambled_logits.append(scrambled_logit)
                importance_scores.append(importance_score)
        else :
            
            scrambler_input_concat = Lambda(lambda x: K.concatenate(x, axis=2))
            scrambler_drop_concat = Lambda(lambda x: K.concatenate(x, axis=2))
            
            scrambler_input = scrambler_input_concat(scrambler_inputs)
            scrambler_drop = scrambler_input_concat(scrambler_drops) if self.mask_dropout else None

            scrambled_logit, importance_score = scrambler(scrambler_input, scrambler_drop, scrambler_label)

            scrambler_logit_split = Lambda(lambda x: [x[:, :, k*input_size_y:(k+1)*input_size_y, :] for k in range(self.n_inputs)])
            scrambler_score_split = Lambda(lambda x: [x[:, :, k*input_size_y:(k+1)*input_size_y, :] for k in range(self.n_inputs)])
            
            scrambled_logits = scrambler_logit_split(scrambled_logit)
            importance_scores = scrambler_score_split(importance_score)

        for input_ix in range(self.n_inputs) :
                
            pwm_logit, pwm, sampled_pwm, _, sampled_mask = sampler(scrambler_classes[input_ix], scrambled_logits[input_ix])

            if zeropad_input :
                zeropad_layer = Lambda(lambda x: x[0] * x[1], name='zeropad_' + str(input_ix))
                sampled_pwm = zeropad_layer([sampled_pwm, sampled_mask])

            pwm_logits.append(pwm_logit)
            pwms.append(pwm)
            sampled_pwms.append(sampled_pwm)
        
        scrambler_model = Model(
            scrambler_classes + scrambler_inputs + (scrambler_drops if scrambler_drops[0] is not None else []) + ([scrambler_label] if scrambler_label is not None else []),
            pwm_logits + pwms + sampled_pwms + importance_scores
        )

        #Initialize Templates and Masks
        initialize_templates(scrambler_model, self.input_templates, self.input_backgrounds_log)

        #Freeze gaussian smoothing kernel
        if self.mask_smoothing :
            scrambler_model.get_layer("scrambler_smooth_conv").set_weights([
                np.reshape(np.array(mask_smoothing_conv_weight), (1, self.mask_smoothing_window_size, 1, 1) if self.input_size_x == 1 else (self.mask_smoothing_window_size, self.mask_smoothing_window_size, 1, 1))
            ])
            scrambler_model.get_layer("scrambler_smooth_conv").trainable = False

        scrambler_model.compile(
            optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
            loss='mean_squared_error'
        )
        
        self.scrambler_model = scrambler_model
    
    def load_model(self, model_path) :
        #Load model
        
        self.scrambler_model.load_weights(model_path, by_name=True)
        print('Loaded scrambler model from %s ' % (model_path))
    
    def save_model(self, model_path) :
        #Save model and weights
        
        self.scrambler_model.save(model_path)
        print('Saved scrambler model at %s ' % (model_path))

    def get_model(self) :
        return self.scrambler_model

    def interpret(self, x, y=None, drop=None, group=None) :
        
        if not isinstance(x, list) :
            x = [x]
        
        signal_is_1d = False
        if len(x[0].shape) == 3 :
            signal_is_1d = True
            
            for i in range(len(x)) :
                x[i] = x[i][:, None, ...]
        
        if group is None :
            group = [np.zeros((x[0].shape[0], 1)) for input_ix in range(self.n_inputs)]
        
        if not isinstance(group, list) :
            group = [group]
        
        if drop is None :
            drop = [
                np.ones((x[0].shape[0], x[0].shape[1], x[0].shape[2], 1)) for input_ix in range(self.n_inputs)
            ] if self.mask_dropout else []
        
        if not isinstance(drop, list) :
            drop = [drop]
        
        label = []
        if y is not None :
            label = [y] if self.label_input else []
        
        input_tensors = group + x + drop + label
        
        #Pad data
        n_pad = self.batch_size - x[0].shape[0] % self.batch_size
        if n_pad == self.batch_size :
            n_pad = 0
        
        if n_pad > 0 :
            input_tensors = [
                np.concatenate([input_tensor, np.zeros(tuple([n_pad] + list(input_tensor.shape)[1:]))], axis=0)
                for input_tensor in input_tensors
            ]

        pred_bundle = self.scrambler_model.predict(x=input_tensors, batch_size=self.batch_size, verbose=True)
        
        if n_pad > 0 :
            pred_bundle = [
                pred_bundle_member[:-n_pad, ...] for pred_bundle_member in pred_bundle
            ]
        
        pwms = []
        samples = []
        scores = []
        for input_ix in range(self.n_inputs) :
            pwm = pred_bundle[self.n_inputs + input_ix]
            sample = pred_bundle[2 * self.n_inputs + input_ix]
            score = pred_bundle[3 * self.n_inputs + input_ix]
            
            if signal_is_1d :
                pwms.append(pwm[:, 0, ...])
                samples.append(sample[:, :, 0, ...])
                scores.append(score[:, 0, ...])
            else:
                pwms.append(pwm)
                samples.append(sample)
                scores.append(score)
        
        if len(pwms) <= 1 :
            return pwms[0], samples[0], scores[0]
        
        return pwms, samples, scores
    
    def train(self, predictor, x_train, y_train, x_test, y_test, n_epochs, extra_input_train=None, extra_input_test=None, group_train=None, group_test=None, monitor_test_indices=None, monitor_batch_freq_dict={0 : 1, 1 : 5, 5 : 10}, adam_lr=0.0001, adam_beta_1=0.5, adam_beta_2=0.9, nll_mode='reconstruction', predictor_task='classification', custom_loss_func=None, reference='predictor', entropy_mode='target', entropy_bits=0., entropy_weight=1.) :
        
        if not isinstance(x_train, list) :
            x_train = [x_train]
            x_test = [x_test]
        
        if group_train is None :
            group_train = [np.zeros((x_train[0].shape[0], 1)) for input_ix in range(self.n_inputs)]
            group_test = [np.zeros((x_test[0].shape[0], 1)) for input_ix in range(self.n_inputs)]
        
        if not isinstance(group_train, list) :
            group_train = [group_train]
            group_test = [group_test]
        
        if extra_input_train is None :
            extra_input_train = []
            extra_input_test = []
        
        if not isinstance(extra_input_train, list) :
            extra_input_train = [extra_input_train]
            extra_input_test = [extra_input_test]
        
        n_trim_train = x_train[0].shape[0] % self.batch_size
        n_trim_test = x_test[0].shape[0] % self.batch_size
        
        if n_trim_train > 0 :
            print("(Trimming size of training data to " + str(int(x_train[0].shape[0] - n_trim_train)) + " examples).")
            
            for i in range(len(x_train)) :
                x_train[i] = x_train[i][:-n_trim_train]
            
            for i in range(len(group_train)) :
                group_train[i] = group_train[i][:-n_trim_train]
            
            for i in range(len(extra_input_train)) :
                extra_input_train[i] = extra_input_train[i][:-n_trim_train]
            
            y_train = y_train[:-n_trim_train]
        
        if n_trim_test > 0 :
            print("(Trimming size of test data to " + str(int(x_test[0].shape[0] - n_trim_test)) + " examples).")
            
            for i in range(len(x_test)) :
                x_test[i] = x_test[i][:-n_trim_test]
            
            for i in range(len(group_test)) :
                group_test[i] = group_test[i][:-n_trim_test]
            
            for i in range(len(extra_input_test)) :
                extra_input_test[i] = extra_input_test[i][:-n_trim_test]
            
            y_test = y_test[:-n_trim_test]
        
        if monitor_test_indices is not None and len(monitor_test_indices) % self.batch_size > 0 :
            monitor_test_indices = monitor_test_indices[:-len(monitor_test_indices) % self.batch_size]
            if len(monitor_test_indices) <= 0 :
                monitor_test_indices = None
        
        signal_is_1d = False
        if len(x_train[0].shape) == 3 :
            signal_is_1d = True
            
            for i in range(len(x_train)) :
                x_train[i] = x_train[i][:, None, ...]
                x_test[i] = x_test[i][:, None, ...]
        
        #Freeze predictor
        predictor.trainable = False
        predictor.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss='mean_squared_error')
        
        #Build loss model pipeline

        #Define model inputs
        scrambler_classes = []
        scrambler_inputs = []
        scrambler_drops = []
        scrambler_extra_inputs = []
        for input_ix in range(self.n_inputs) :
            scrambler_classes.append(Input(shape=(1,), name='t_scrambler_group_' + str(input_ix)))
            scrambler_inputs.append(Input(shape=(self.input_size_x, self.input_size_y, self.n_out_channels), name='t_scrambler_input_' + str(input_ix)))
            if self.mask_dropout :
                scrambler_drops.append(Input(shape=(self.input_size_x, self.input_size_y, 1), name='t_scrambler_drop_' + str(input_ix)))
            else :
                scrambler_drops.append(None)
        
        scrambler_label = None
        if self.label_input or reference == 'label' :
            scrambler_label = Input(shape=(self.n_classes,), name='t_scrambler_label')
        
        for extra_in_ix, extra_in in enumerate(extra_input_train) :
            scrambler_extra_inputs.append(Input(shape=tuple(list(extra_in.shape)[1:]), name='t_scrambler_extra_' + str(extra_in_ix)))
        
        scrambled_logits = []
        importance_scores = []
        pwm_logits = []
        pwms = []
        sampled_pwms = []
        if self.multi_input_mode == 'siamese' or self.n_inputs == 1 :
            
            for input_ix in range(self.n_inputs) :
                
                scrambled_logit, importance_score = self.scrambler(scrambler_inputs[input_ix], scrambler_drops[input_ix], scrambler_label)

                scrambled_logits.append(scrambled_logit)
                importance_scores.append(importance_score)
        else :
            
            scrambler_input_concat = Lambda(lambda x: K.concatenate(x, axis=2))
            scrambler_drop_concat = Lambda(lambda x: K.concatenate(x, axis=2))
            
            scrambler_input = scrambler_input_concat(scrambler_inputs)
            scrambler_drop = scrambler_input_concat(scrambler_drops) if self.mask_dropout else None

            scrambled_logit, importance_score = self.scrambler(scrambler_input, scrambler_drop, scrambler_label)

            scrambler_logit_split = Lambda(lambda x: [x[:, :, k*self.input_size_y:(k+1)*self.input_size_y, :] for k in range(self.n_inputs)])
            scrambler_score_split = Lambda(lambda x: [x[:, :, k*self.input_size_y:(k+1)*self.input_size_y, :] for k in range(self.n_inputs)])
            
            scrambled_logits = scrambler_logit_split(scrambled_logit)
            importance_scores = scrambler_score_split(importance_score)

        deflated_sampled_pwms = []
        pwm_masks = []
        sampled_masks = []
        for input_ix in range(self.n_inputs) :
                
            pwm_logit, pwm, sampled_pwm, pwm_mask, sampled_mask = self.sampler(scrambler_classes[input_ix], scrambled_logits[input_ix])

            if self.zeropad_input :
                zeropad_layer = Lambda(lambda x: x[0] * x[1], name='t_zeropad_' + str(input_ix))
                sampled_pwm = zeropad_layer([sampled_pwm, sampled_mask])

            pwm_logits.append(pwm_logit)
            pwms.append(pwm)
            sampled_pwms.append(sampled_pwm)
            pwm_masks.append(pwm_mask)
            sampled_masks.append(sampled_mask)

            #Define layer to deflate sample axis
            deflate_scrambled_sample = Lambda(lambda x: K.reshape(x, (self.batch_size * self.n_samples, self.input_size_x, self.input_size_y, self.n_out_channels)), name='t_deflate_scrambled_sample_' + str(input_ix))

            #Deflate sample axis
            deflated_sampled_pwm = deflate_scrambled_sample(sampled_pwm)
            deflated_sampled_pwms.append(deflated_sampled_pwm)

        #Make reference prediction on non-scrambled input sequence
        switch_to_1d_non_scrambled = Lambda(lambda x, signal_is_1d=signal_is_1d: x[:, 0, ...] if signal_is_1d else x, name='t_switch_to_1d_non_scrambled')
        scrambler_inputs_to_pred = [switch_to_1d_non_scrambled(scrambler_input) for scrambler_input in scrambler_inputs]
        y_pred_non_scrambled_deflated = predictor(scrambler_inputs_to_pred + scrambler_extra_inputs) if reference == 'predictor' else scrambler_label

        #Make prediction on scrambled sequence samples
        switch_to_1d_scrambled = Lambda(lambda x, signal_is_1d=signal_is_1d: x[:, 0, ...] if signal_is_1d else x, name='t_switch_to_1d_scrambled')
        deflated_sampled_pwms_to_pred = [switch_to_1d_scrambled(deflated_sampled_pwm) for deflated_sampled_pwm in deflated_sampled_pwms]
        scrambler_extra_inputs_repeated = []
        for extra_in_ix, scrambler_extra_inp in enumerate(scrambler_extra_inputs) :
            repeat_scrambler_extra_input = Lambda(lambda x: K.repeat_elements(x, self.n_samples, axis=0), name='repeat_scrambler_extra_input_' + str(extra_in_ix))
            scrambler_extra_inputs_repeated.append(repeat_scrambler_extra_input(scrambler_extra_inp))
        y_pred_scrambled_deflated = predictor(deflated_sampled_pwms_to_pred + scrambler_extra_inputs_repeated)

        #Define layer to inflate sample axis
        inflate_non_scrambled_prediction = Lambda(lambda x: K.tile(K.expand_dims(x, axis=1), (1, self.n_samples, 1)), name='t_inflate_non_scrambled_prediction')
        inflate_scrambled_prediction = Lambda(lambda x: K.reshape(x, (self.batch_size, self.n_samples, self.n_classes)), name='t_inflate_scrambled_prediction')

        #Inflate sample axis
        y_pred_non_scrambled = inflate_non_scrambled_prediction(y_pred_non_scrambled_deflated)
        y_pred_scrambled = inflate_scrambled_prediction(y_pred_scrambled_deflated)
        
        #Define background matrix embeddings
        seq_reshape_layer = Reshape((self.input_size_x, self.input_size_y, self.n_out_channels))

        x_mean_dense = Embedding(len(self.input_templates), self.input_size_x * self.input_size_y * self.n_out_channels, embeddings_initializer='zeros', name='x_mean_dense')
        
        x_means = [seq_reshape_layer(x_mean_dense(scrambler_class)) for scrambler_class in scrambler_classes]

        scrambler_mode_coeff = 1.
        if self.scrambler_mode == 'occlusion' and reference == 'predictor' :
            scrambler_mode_coeff = -1.
        elif self.scrambler_mode == 'occlusion' and reference == 'label' :
            if self.n_classes <= 1 :
                y_pred_non_scrambled = Lambda(lambda x: 1. - x, name="t_invert_binary_label")(y_pred_non_scrambled)
        
        #NLL cost
        nll_loss_func = None
        
        if custom_loss_func is not None :
            nll_loss_func = custom_loss_func
            scrambler_mode_coeff = 1.
        else :
            if nll_mode == 'reconstruction' and predictor_task == 'classification' :
                if self.n_classes > 1 :
                    nll_loss_func = get_softmax_kl_divergence()
                else :
                    nll_loss_func = get_sigmoid_kl_divergence()
            elif nll_mode == 'reconstruction' and predictor_task == 'classification_sym' :
                nll_loss_func = get_symmetric_sigmoid_kl_divergence()
            elif nll_mode == 'maximization' and predictor_task == 'classification' :
                nll_loss_func = get_sigmoid_max_nll()
            elif nll_mode == 'minimization' and predictor_task == 'classification' :
                nll_loss_func = get_sigmoid_min_nll()
            elif nll_mode == 'reconstruction' and predictor_task == 'regression' :
                nll_loss_func = get_mse()
            elif nll_mode == 'maximization' and predictor_task == 'regression' :
                nll_loss_func = get_linear_max_nll()
            elif nll_mode == 'minimization' and predictor_task == 'regression' :
                nll_loss_func = get_linear_min_nll()

        #Entropy cost
        entropy_loss_func = None
        if entropy_mode == 'target' :
            entropy_loss_func = get_target_entropy_sme_masked(x_start=0, x_end=self.input_size_x, y_start=0, y_end=self.input_size_y, target_bits=entropy_bits)
        elif entropy_mode == 'maximization' :
            entropy_loss_func = get_margin_entropy_ame_masked(x_start=0, x_end=self.input_size_x, y_start=0, y_end=self.input_size_y, max_bits=entropy_bits)

        #Execute NLL cost
        nll_loss = Lambda(lambda x: scrambler_mode_coeff * nll_loss_func(x[0], x[1]), name='nll')([
            y_pred_non_scrambled,
            y_pred_scrambled
        ])

        #Execute entropy cost
        entropy_losses = []
        for input_ix in range(self.n_inputs) :
            entropy_loss = Lambda(lambda x: K.expand_dims(entropy_loss_func(x[0], x[1], x[2]), axis=-1), name='entropy_' + str(input_ix))([
                pwms[input_ix],
                pwm_masks[input_ix],
                x_means[input_ix]
            ])
            entropy_losses.append(entropy_loss)
        
        entropy_loss = None
        if len(entropy_losses) > 1 :
            entropy_loss = Lambda(lambda x: K.mean(x, axis=-1), name='entropy')(Concatenate(axis=-1)(entropy_losses))
        else :
            entropy_loss = Lambda(lambda x: K.mean(x[0], axis=-1), name='entropy')(entropy_losses)
        
        loss_model = Model(
            scrambler_classes + scrambler_inputs + (scrambler_drops if scrambler_drops[0] is not None else []) + ([scrambler_label] if scrambler_label is not None else []) + scrambler_extra_inputs,
            [nll_loss, entropy_loss]
        )

        #Initialize Templates and Masks
        initialize_templates(loss_model, self.input_templates, self.input_backgrounds_log)

        #Initialize Sequence Length Parameters
        initialize_backgrounds(loss_model, self.input_backgrounds)

        loss_model.compile(
            optimizer=keras.optimizers.Adam(lr=adam_lr, beta_1=adam_beta_1, beta_2=adam_beta_2),
            loss={
                'nll' : get_weighted_loss(loss_coeff=1.0),
                'entropy' : get_weighted_loss(loss_coeff=entropy_weight)
            }
        )
        
        self.loss_model = loss_model

        #Execute training procedure
        callbacks = []
            
        dummy_train = np.zeros((x_train[0].shape[0], 1))
        dummy_test = np.zeros((x_test[0].shape[0], 1))
        
        drop_train = [
            np.ones((x_train[0].shape[0], x_train[0].shape[1], x_train[0].shape[2], 1)) for input_ix in range(self.n_inputs)
        ] if self.mask_dropout else []
        drop_test = [
            np.ones((x_test[0].shape[0], x_test[0].shape[1], x_test[0].shape[2], 1)) for input_ix in range(self.n_inputs)
        ] if self.mask_dropout else []
        
        label_train = [y_train] if (self.label_input or reference == 'label') else []
        label_test = [y_test] if (self.label_input or reference == 'label') else []
        
        monitor = None
        if monitor_test_indices is not None :
            
            group_track = [g[monitor_test_indices] for g in group_test]
            x_track = [x[monitor_test_indices] for x in x_test]
            drop_track = [d[monitor_test_indices] for d in drop_test]
            label_track = [l[monitor_test_indices] for l in label_test]
            extra_input_track = [e[monitor_test_indices] for e in extra_input_test]
            
            monitor_loss_tensors = group_track + x_track + drop_track + label_track + extra_input_track
            monitor_tensors = group_track + x_track + drop_track + (label_track if self.label_input else [])
            
            monitor = ScramblerMonitor(self.scrambler_model, self.loss_model, monitor_loss_tensors, monitor_tensors, n_inputs=self.n_inputs, track_mode='batch', batch_freq_dict=monitor_batch_freq_dict, batch_size=self.batch_size)
            callbacks.append(monitor)

        train_history = loss_model.fit(
            group_train + x_train + drop_train + label_train + extra_input_train,
            [dummy_train, dummy_train],
            shuffle=True,
            epochs=n_epochs,
            batch_size=self.batch_size,
            validation_data=(
                group_test + x_test + drop_test + label_test + extra_input_test,
                [dummy_test, dummy_test]
            ),
            callbacks=callbacks
        )
        
        train_history = train_history.history
        
        if monitor is not None :
            train_history['monitor_batches'] = monitor.batch_history
            train_history['monitor_epochs'] = monitor.epoch_history
            train_history['monitor_importance_scores'] = monitor.scores_history
            train_history['monitor_pwms'] = monitor.pwm_history
            train_history['monitor_nll_losses'] = monitor.nll_loss_history
            train_history['monitor_entropy_losses'] = monitor.entropy_loss_history
        
        return train_history
    
    def finetune(self, predictor, x, y, batch_size, n_iters, drop=None, extra_input=None, group=None, norm_mode='instance', max_score_clip=4., adam_lr=0.01, adam_beta_1=0.5, adam_beta_2=0.9, nll_mode='reconstruction', predictor_task='classification', custom_loss_func=None, reference='predictor', entropy_mode='target', entropy_bits=0., entropy_weight=1.) :
        
        #Collect pre-trained importance scores
        print("Generating pre-trained scores...")
        _, _, pretrained_scores = self.interpret(x, y=y, drop=drop, group=group)
        
        #Load finetuner
        finetuner = load_finetuning_model(
            batch_size,
            self.input_size_x,
            self.input_size_y,
            scrambler_mode=self.scrambler_mode,
            n_out_channels=self.n_out_channels,
            n_spatial_dims=1 if self.input_size_x == 1 else 2,
            mask_smoothing=self.mask_smoothing,
            smooth_window_size=self.mask_smoothing_window_size,
            norm_mode=norm_mode,
            max_score_clip=max_score_clip
        )
        
        return self._optimize(predictor, finetuner, [pretrained_scores], x, y, batch_size, n_iters, 'finetune', drop=drop, extra_input=extra_input, group=group, adam_lr=adam_lr, adam_beta_1=adam_beta_1, adam_beta_2=adam_beta_2, nll_mode=nll_mode, predictor_task=predictor_task, custom_loss_func=custom_loss_func, reference=reference, entropy_mode=entropy_mode, entropy_bits=entropy_bits, entropy_weight=entropy_weight)
    
    def optimize(self, predictor, x, y, batch_size, n_iters, drop=None, extra_input=None, group=None, norm_mode='instance', adam_lr=0.01, adam_beta_1=0.5, adam_beta_2=0.9, nll_mode='reconstruction', predictor_task='classification', custom_loss_func=None, reference='predictor', entropy_mode='target', entropy_bits=0., entropy_weight=1.) :
        
        #Load optimizer
        optimizer = load_optimization_model(
            batch_size,
            self.input_size_x,
            self.input_size_y,
            scrambler_mode=self.scrambler_mode,
            n_out_channels=self.n_out_channels,
            n_spatial_dims=1 if self.input_size_x == 1 else 2,
            mask_smoothing=self.mask_smoothing,
            smooth_window_size=self.mask_smoothing_window_size,
            norm_mode=norm_mode
        )
        
        return self._optimize(predictor, optimizer, [], x, y, batch_size, n_iters, 'optimize', drop=drop, extra_input=extra_input, group=group, adam_lr=adam_lr, adam_beta_1=adam_beta_1, adam_beta_2=adam_beta_2, nll_mode=nll_mode, predictor_task=predictor_task, custom_loss_func=custom_loss_func, reference=reference, entropy_mode=entropy_mode, entropy_bits=entropy_bits, entropy_weight=entropy_weight)
    
    def _optimize(self, predictor, finetuner, pretrained_scores, x, y, batch_size, n_iters, opt_mode, drop=None, extra_input=None, group=None, adam_lr=0.01, adam_beta_1=0.5, adam_beta_2=0.9, nll_mode='reconstruction', predictor_task='classification', custom_loss_func=None, reference='predictor', entropy_mode='target', entropy_bits=0., entropy_weight=1.) :
        
        if not isinstance(x, list) :
            x = [x]
        
        signal_is_1d = False
        if len(x[0].shape) == 3 :
            signal_is_1d = True
            
            for i in range(len(x)) :
                x[i] = x[i][:, None, ...]
        
        if group is None :
            group = [np.zeros((x[0].shape[0], 1)) for input_ix in range(self.n_inputs)]
        
        if not isinstance(group, list) :
            group = [group]
        
        if extra_input is None :
            extra_input = []
        
        if not isinstance(extra_input, list) :
            extra_input = [extra_input]
        
        if drop is None :
            drop = [
                np.ones((x[0].shape[0], x[0].shape[1], x[0].shape[2], 1)) for input_ix in range(self.n_inputs)
            ]
        
        if not isinstance(drop, list) :
            drop = [drop]
        
        #Freeze predictor
        predictor.trainable = False
        predictor.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss='mean_squared_error')
        
        #Build finetuning model
        mask_smoothing_conv_weight = gaussian(self.mask_smoothing_window_size, self.mask_smoothing_std)
        
        #self.finetuner = finetuner

        #Load sampler
        finetuning_sampler = build_sampler(batch_size, self.input_size_x, self.input_size_y, n_classes=len(self.input_templates), n_samples=self.n_samples, sample_mode=self.sample_mode, n_channels=self.n_out_channels, model_prefix='ft_')
        
        #self.finetuning_sampler = finetuning_sampler
        
        #Build scrambler model
        scrambler_classes = []
        scrambler_inputs = []
        scrambler_drops = []
        scrambler_pretrained_scores = []
        for input_ix in range(self.n_inputs) :
            scrambler_classes.append(Input(batch_shape=(batch_size, 1), name='f_scrambler_group_' + str(input_ix)))
            scrambler_inputs.append(Input(batch_shape=(batch_size, self.input_size_x, self.input_size_y, self.n_out_channels), name='f_scrambler_input_' + str(input_ix)))
            scrambler_drops.append(Input(batch_shape=(batch_size, self.input_size_x, self.input_size_y, 1), name='f_scrambler_drop_' + str(input_ix)))
            if opt_mode == 'finetune' :
                scrambler_pretrained_scores.append(Input(batch_shape=(batch_size, self.input_size_x, self.input_size_y, 1), name='f_scrambler_pretrained_scores_' + str(input_ix)))
        
        scrambled_logits = []
        importance_scores = []
        pwm_logits = []
        pwms = []
        sampled_pwms = []
        if self.multi_input_mode == 'siamese' or self.n_inputs == 1 :
            
            for input_ix in range(self.n_inputs) :
                
                scrambled_logit, importance_score = None, None
                if opt_mode == 'finetune' :
                    scrambled_logit, importance_score = finetuner(scrambler_inputs[input_ix], scrambler_drops[input_ix], scrambler_pretrained_scores[input_ix])
                else :
                    scrambled_logit, importance_score = finetuner(scrambler_inputs[input_ix], scrambler_drops[input_ix])
                
                scrambled_logits.append(scrambled_logit)
                importance_scores.append(importance_score)
        else :
            
            scrambler_input_concat = Lambda(lambda x: K.concatenate(x, axis=2))
            scrambler_drop_concat = Lambda(lambda x: K.concatenate(x, axis=2))
            scrambler_pretrained_score_concat = Lambda(lambda x: K.concatenate(x, axis=2))
            
            scrambler_input = scrambler_input_concat(scrambler_inputs)
            scrambler_drop = scrambler_input_concat(scrambler_drops) if self.mask_dropout else None
            scrambler_pretrained_score = scrambler_pretrained_score_concat(scrambler_pretrained_scores) if opt_mode == 'finetune' else None
            
            scrambled_logit, importance_score = None, None
            if opt_mode == 'finetune' :
                scrambled_logit, importance_score = finetuner(scrambler_input, scrambler_drop, scrambler_pretrained_score)
            else :
                scrambled_logit, importance_score = finetuner(scrambler_input, scrambler_drop)
            
            scrambler_logit_split = Lambda(lambda x: [x[:, :, k*input_size_y:(k+1)*input_size_y, :] for k in range(self.n_inputs)])
            scrambler_score_split = Lambda(lambda x: [x[:, :, k*input_size_y:(k+1)*input_size_y, :] for k in range(self.n_inputs)])
            
            scrambled_logits = scrambler_logit_split(scrambled_logit)
            importance_scores = scrambler_score_split(importance_score)

        for input_ix in range(self.n_inputs) :
                
            pwm_logit, pwm, sampled_pwm, _, sampled_mask = finetuning_sampler(scrambler_classes[input_ix], scrambled_logits[input_ix])

            if self.zeropad_input :
                zeropad_layer = Lambda(lambda x: x[0] * x[1], name='f_zeropad_' + str(input_ix))
                sampled_pwm = zeropad_layer([sampled_pwm, sampled_mask])

            pwm_logits.append(pwm_logit)
            pwms.append(pwm)
            sampled_pwms.append(sampled_pwm)
        
        finetuning_model = Model(
            scrambler_classes + scrambler_inputs + scrambler_drops + scrambler_pretrained_scores,
            pwm_logits + pwms + sampled_pwms + importance_scores
        )

        #Initialize Templates and Masks
        initialize_templates(finetuning_model, self.input_templates, self.input_backgrounds_log, model_prefix='ft_')

        #Freeze gaussian smoothing kernel
        if self.mask_smoothing :
            finetuning_model.get_layer("ft_scrambler_smooth_conv").set_weights([
                np.reshape(np.array(mask_smoothing_conv_weight), (1, self.mask_smoothing_window_size, 1, 1) if input_size_x == 1 else (self.mask_smoothing_window_size, self.mask_smoothing_window_size, 1, 1))
            ])
            finetuning_model.get_layer("ft_scrambler_smooth_conv").trainable = False

        finetuning_model.compile(
            optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
            loss='mean_squared_error'
        )
        
        #self.finetuning_model = finetuning_model
        
        #Build loss model pipeline
        
        #Define model inputs
        scrambler_classes = []
        scrambler_inputs = []
        scrambler_drops = []
        scrambler_pretrained_scores = []
        scrambler_extra_inputs = []
        for input_ix in range(self.n_inputs) :
            scrambler_classes.append(Input(batch_shape=(batch_size, 1), name='ft_scrambler_group_' + str(input_ix)))
            scrambler_inputs.append(Input(batch_shape=(batch_size, self.input_size_x, self.input_size_y, self.n_out_channels), name='ft_scrambler_input_' + str(input_ix)))
            scrambler_drops.append(Input(batch_shape=(batch_size, self.input_size_x, self.input_size_y, 1), name='ft_scrambler_drop_' + str(input_ix)))
            if opt_mode == 'finetune' :
                scrambler_pretrained_scores.append(Input(batch_shape=(batch_size, self.input_size_x, self.input_size_y, 1), name='ft_scrambler_pretrained_scores_' + str(input_ix)))
        
        scrambler_label = None
        if reference == 'label' :
            scrambler_label = Input(batch_shape=(batch_size, self.n_classes), name='ft_scrambler_label')
        
        for extra_in_ix, extra_in in enumerate(extra_input) :
            scrambler_extra_inputs.append(Input(batch_shape=tuple([batch_size] + list(extra_in.shape)[1:]), name='ft_scrambler_extra_' + str(extra_in_ix)))
        
        scrambled_logits = []
        importance_scores = []
        pwm_logits = []
        pwms = []
        sampled_pwms = []
        if self.multi_input_mode == 'siamese' or self.n_inputs == 1 :
            
            for input_ix in range(self.n_inputs) :
                
                scrambled_logit, importance_score = None, None
                if opt_mode == 'finetune' :
                    scrambled_logit, importance_score = finetuner(scrambler_inputs[input_ix], scrambler_drops[input_ix], scrambler_pretrained_scores[input_ix])
                else :
                    scrambled_logit, importance_score = finetuner(scrambler_inputs[input_ix], scrambler_drops[input_ix])
                
                scrambled_logits.append(scrambled_logit)
                importance_scores.append(importance_score)
        else :
            
            scrambler_input_concat = Lambda(lambda x: K.concatenate(x, axis=2))
            scrambler_drop_concat = Lambda(lambda x: K.concatenate(x, axis=2))
            scrambler_pretrained_score_concat = Lambda(lambda x: K.concatenate(x, axis=2))
            
            scrambler_input = scrambler_input_concat(scrambler_inputs)
            scrambler_drop = scrambler_input_concat(scrambler_drops) if self.mask_dropout else None
            scrambler_pretrained_score = scrambler_pretrained_score_concat(scrambler_pretrained_scores) if opt_mode == 'finetune' else None
            
            scrambled_logit, importance_score = None, None
            if opt_mode == 'finetune' :
                scrambled_logit, importance_score = finetuner(scrambler_input, scrambler_drop, scrambler_pretrained_score)
            else :
                scrambled_logit, importance_score = finetuner(scrambler_input, scrambler_drop)
            
            scrambler_logit_split = Lambda(lambda x: [x[:, :, k*input_size_y:(k+1)*input_size_y, :] for k in range(self.n_inputs)])
            scrambler_score_split = Lambda(lambda x: [x[:, :, k*input_size_y:(k+1)*input_size_y, :] for k in range(self.n_inputs)])
            
            scrambled_logits = scrambler_logit_split(scrambled_logit)
            importance_scores = scrambler_score_split(importance_score)

        deflated_sampled_pwms = []
        pwm_masks = []
        sampled_masks = []
        for input_ix in range(self.n_inputs) :
                
            pwm_logit, pwm, sampled_pwm, pwm_mask, sampled_mask = finetuning_sampler(scrambler_classes[input_ix], scrambled_logits[input_ix])

            if self.zeropad_input :
                zeropad_layer = Lambda(lambda x: x[0] * x[1], name='ft_zeropad_' + str(input_ix))
                sampled_pwm = zeropad_layer([sampled_pwm, sampled_mask])

            pwm_logits.append(pwm_logit)
            pwms.append(pwm)
            sampled_pwms.append(sampled_pwm)
            pwm_masks.append(pwm_mask)
            sampled_masks.append(sampled_mask)

            #Define layer to deflate sample axis
            deflate_scrambled_sample = Lambda(lambda x: K.reshape(x, (self.batch_size * self.n_samples, self.input_size_x, self.input_size_y, self.n_out_channels)), name='t_deflate_scrambled_sample_' + str(input_ix))

            #Deflate sample axis
            deflated_sampled_pwm = deflate_scrambled_sample(sampled_pwm)
            deflated_sampled_pwms.append(deflated_sampled_pwm)

        #Make reference prediction on non-scrambled input sequence
        switch_to_1d_non_scrambled = Lambda(lambda x, signal_is_1d=signal_is_1d: x[:, 0, ...] if signal_is_1d else x, name='t_switch_to_1d_non_scrambled')
        scrambler_inputs_to_pred = [switch_to_1d_non_scrambled(scrambler_input) for scrambler_input in scrambler_inputs]
        y_pred_non_scrambled_deflated = predictor(scrambler_inputs_to_pred + scrambler_extra_inputs) if reference == 'predictor' else scrambler_label

        #Make prediction on scrambled sequence samples
        switch_to_1d_scrambled = Lambda(lambda x, signal_is_1d=signal_is_1d: x[:, 0, ...] if signal_is_1d else x, name='t_switch_to_1d_scrambled')
        deflated_sampled_pwms_to_pred = [switch_to_1d_scrambled(deflated_sampled_pwm) for deflated_sampled_pwm in deflated_sampled_pwms]
        scrambler_extra_inputs_repeated = []
        for extra_in_ix, scrambler_extra_inp in enumerate(scrambler_extra_inputs) :
            repeat_scrambler_extra_input = Lambda(lambda x: K.repeat_elements(x, self.n_samples, axis=0), name='repeat_scrambler_extra_input_' + str(extra_in_ix))
            scrambler_extra_inputs_repeated.append(repeat_scrambler_extra_input(scrambler_extra_inp))
        y_pred_scrambled_deflated = predictor(deflated_sampled_pwms_to_pred + scrambler_extra_inputs_repeated)

        #Define layer to inflate sample axis
        inflate_non_scrambled_prediction = Lambda(lambda x: K.tile(K.expand_dims(x, axis=1), (1, self.n_samples, 1)), name='ft_inflate_non_scrambled_prediction')
        inflate_scrambled_prediction = Lambda(lambda x: K.reshape(x, (self.batch_size, self.n_samples, self.n_classes)), name='ft_inflate_scrambled_prediction')

        #Inflate sample axis
        y_pred_non_scrambled = inflate_non_scrambled_prediction(y_pred_non_scrambled_deflated)
        y_pred_scrambled = inflate_scrambled_prediction(y_pred_scrambled_deflated)
        
        #Define background matrix embeddings
        seq_reshape_layer = Reshape((self.input_size_x, self.input_size_y, self.n_out_channels))

        x_mean_dense = Embedding(len(self.input_templates), self.input_size_x * self.input_size_y * self.n_out_channels, embeddings_initializer='zeros', name='ft_x_mean_dense')
        
        x_means = [seq_reshape_layer(x_mean_dense(scrambler_class)) for scrambler_class in scrambler_classes]

        scrambler_mode_coeff = 1.
        if self.scrambler_mode == 'occlusion' and reference == 'predictor' :
            scrambler_mode_coeff = -1.
        elif self.scrambler_mode == 'occlusion' and reference == 'label' :
            if self.n_classes <= 1 :
                y_pred_non_scrambled = Lambda(lambda x: 1. - x, name="ft_invert_binary_label")(y_pred_non_scrambled)
        
        #NLL cost
        nll_loss_func = None
        
        if custom_loss_func is not None :
            nll_loss_func = custom_loss_func
            scrambler_mode_coeff = 1.
        else :
            if nll_mode == 'reconstruction' and predictor_task == 'classification' :
                if self.n_classes > 1 :
                    nll_loss_func = get_softmax_kl_divergence()
                else :
                    nll_loss_func = get_sigmoid_kl_divergence()
            elif nll_mode == 'reconstruction' and predictor_task == 'classification_sym' :
                nll_loss_func = get_symmetric_sigmoid_kl_divergence()
            elif nll_mode == 'maximization' and predictor_task == 'classification' :
                nll_loss_func = get_sigmoid_max_nll()
            elif nll_mode == 'minimization' and predictor_task == 'classification' :
                nll_loss_func = get_sigmoid_min_nll()
            elif nll_mode == 'reconstruction' and predictor_task == 'regression' :
                nll_loss_func = get_mse()
            elif nll_mode == 'maximization' and predictor_task == 'regression' :
                nll_loss_func = get_linear_max_nll()
            elif nll_mode == 'minimization' and predictor_task == 'regression' :
                nll_loss_func = get_linear_min_nll()

        #Entropy cost
        entropy_loss_func = None
        if entropy_mode == 'target' :
            entropy_loss_func = get_target_entropy_sme_masked(x_start=0, x_end=self.input_size_x, y_start=0, y_end=self.input_size_y, target_bits=entropy_bits)
        elif entropy_mode == 'maximization' :
            entropy_loss_func = get_margin_entropy_ame_masked(x_start=0, x_end=self.input_size_x, y_start=0, y_end=self.input_size_y, max_bits=entropy_bits)

        #Execute NLL cost
        
        nll_loss = Lambda(lambda x: K.reshape(K.sum(scrambler_mode_coeff * nll_loss_func(x[0], x[1]), axis=0), (1,)), name='ft_nll')([
            y_pred_non_scrambled,
            y_pred_scrambled
        ])

        #Execute entropy cost
        entropy_losses = []
        for input_ix in range(self.n_inputs) :
            entropy_loss = Lambda(lambda x: K.expand_dims(entropy_loss_func(x[0], x[1], x[2]), axis=-1), name='ft_entropy_' + str(input_ix))([
                pwms[input_ix],
                pwm_masks[input_ix],
                x_means[input_ix]
            ])
            entropy_losses.append(entropy_loss)
        
        entropy_loss = None
        if len(entropy_losses) > 1 :
            entropy_loss = Lambda(lambda x: K.reshape(K.sum(K.mean(x, axis=-1), axis=0), (1,)), name='ft_entropy')(Concatenate(axis=-1)(entropy_losses))
        else :
            entropy_loss = Lambda(lambda x: K.reshape(K.sum(K.mean(x, axis=-1), axis=0), (1,)), name='ft_entropy')(entropy_losses[0])
        
        finetuning_loss_model = Model(
            scrambler_classes + scrambler_inputs + scrambler_drops + scrambler_pretrained_scores + ([scrambler_label] if scrambler_label is not None else []) + scrambler_extra_inputs,
            [nll_loss, entropy_loss]
        )

        #Initialize Templates and Masks
        initialize_templates(finetuning_loss_model, self.input_templates, self.input_backgrounds_log, model_prefix='ft_')

        #Initialize Sequence Length Parameters
        initialize_backgrounds(finetuning_loss_model, self.input_backgrounds, model_prefix='ft_')
        
        opt = keras.optimizers.Adam(lr=adam_lr, beta_1=adam_beta_1, beta_2=adam_beta_2)
        
        finetuning_loss_model.compile(
            optimizer=opt,
            loss={
                'ft_nll' : get_weighted_loss(loss_coeff=1.0),
                'ft_entropy' : get_weighted_loss(loss_coeff=entropy_weight)
            }
        )
        
        #self.finetuning_loss_model = finetuning_loss_model

        #(Re-)Initialize scrambler mask
        def _reset_generator(scrambler_model, verbose=False) :
            session = K.get_session()
            for layer in scrambler_model.layers :
                if 'scrambler' in layer.name :
                    for v in layer.__dict__:
                        v_arg = getattr(layer, v)
                        if hasattr(v_arg,'initializer'):
                            initializer_method = getattr(v_arg, 'initializer')
                            initializer_method.run(session=session)
                            if verbose :
                                print('reinitializing layer {}.{}'.format(layer.name, v))

        #(Re-)Initialize Optimizer
        def _reset_optimizer(opt, verbose=False) :
            session = K.get_session()
            for v in opt.__dict__:
                v_arg = getattr(opt, v)
                if hasattr(v_arg,'initializer'):
                    initializer_method = getattr(v_arg, 'initializer')
                    initializer_method.run(session=session)
                    if verbose :
                        print('reinitializing optimizer parameter {}'.format(v))

        #Reset mask
        _reset_generator(finetuning_model, verbose=False)
        _reset_generator(finetuning_loss_model, verbose=False)
        _reset_optimizer(opt, verbose=False)
        
        #Execute training procedure
        dummy = np.zeros((batch_size, 1))
        
        label = [y] if reference == 'label' else []
        
        input_tensors = group + x + drop + pretrained_scores + label + extra_input
        
        #Pad data
        n_pad = batch_size - x[0].shape[0] % batch_size
        if n_pad == batch_size :
            n_pad = 0
        
        if n_pad > 0 :
            input_tensors = [
                np.concatenate([input_tensor, np.zeros(tuple([n_pad] + list(input_tensor.shape)[1:]))], axis=0)
                for input_tensor in input_tensors
            ]
        
        n_batches = input_tensors[0].shape[0] // batch_size
        
        train_histories = []
        pwms = [[] for i in range(self.n_inputs)]
        samples = [[] for i in range(self.n_inputs)]
        scores = [[] for i in range(self.n_inputs)]
        for batch_ix in range(n_batches) :

            if batch_ix % 20 == 0 :
                print("Finetuning batch " + str(batch_ix) + "...")
            
            input_tensors_batch = [
                input_tensor[batch_ix*batch_size:(batch_ix+1)*batch_size] for input_tensor in input_tensors
            ]

            train_history = LossHistory(loss_names=['ft_nll', 'ft_entropy'])

            # train the autoscrambler
            _ = finetuning_loss_model.fit(
                input_tensors_batch,
                [dummy, dummy],
                epochs=1,
                steps_per_epoch=n_iters,
                #batch_size=batch_size,
                callbacks=[train_history]
            )

            pred_bundle = finetuning_model.predict(x=input_tensors_batch, batch_size=batch_size, verbose=False)
            
            if n_pad > 0 :
                pred_bundle = [
                    pred_bundle_member[:-n_pad, ...] for pred_bundle_member in pred_bundle
                ]
        
            temp_pwms = []
            temp_samples = []
            temp_scores = []
            for input_ix in range(self.n_inputs) :
                temp_pwm = pred_bundle[self.n_inputs + input_ix]
                temp_sample = pred_bundle[2 * self.n_inputs + input_ix]
                temp_score = pred_bundle[3 * self.n_inputs + input_ix]

                temp_pwms.append(temp_pwm)
                temp_samples.append(temp_sample)
                temp_scores.append(temp_score)
            
            for i in range(self.n_inputs) :
                pwms[i].append(temp_pwms[i])
                samples[i].append(temp_samples[i])
                scores[i].append(temp_scores[i])
            
            train_histories.append(train_history.loss_dict)

            #Reset mask
            _reset_generator(finetuning_model)
            _reset_generator(finetuning_loss_model)
            _reset_optimizer(opt)
        
        for i in range(self.n_inputs) :
            if signal_is_1d :
                pwms[i] = np.concatenate(pwms[i], axis=0)[:, 0, ...]
                samples[i] = np.concatenate(samples[i], axis=0)[:, :, 0, ...]
                scores[i] = np.concatenate(scores[i], axis=0)[:, 0, ...]
            else :
                pwms[i] = np.concatenate(pwms[i], axis=0)
                samples[i] = np.concatenate(samples[i], axis=0)
                scores[i] = np.concatenate(scores[i], axis=0)
        
        if len(pwms) <= 1 :
            return pwms[0], samples[0], scores[0], train_histories
        
        return pwms, samples, scores, train_histories
