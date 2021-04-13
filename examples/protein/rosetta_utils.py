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
import string

import keras.backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer

from scrambler.models import *

def one_hot_encode_msa(msa, ns=21) :

    one_hot = np.zeros((msa.shape[0], msa.shape[1], ns))
    for i in range(msa.shape[0]) :
        for j in range(msa.shape[1]) :
            one_hot[i, j, int(msa[i, j])] = 1.

    return one_hot

def parse_a3m(filename):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in open(filename,"r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    return msa

def make_a3m(seqs) :
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    msa[msa > 20] = 20
    
    return msa

#Code from https://gist.github.com/mayukh18/c576a37a74a9a5160ff32a535c2907b9
class AdamAccumulate(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):
        if accum_iters < 1:
            raise ValueError('accum_iters must be >= 1')
        super(AdamAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        completed_updates = K.cast(K.tf.floordiv(self.iterations, self.accum_iters), K.floatx())

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * completed_updates))

        t = completed_updates + 1

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

        # self.iterations incremented after processing a batch
        # batch:              1 2 3 4 5 6 7 8 9
        # self.iterations:    0 1 2 3 4 5 6 7 8
        # update_switch = 1:        x       x    (if accum_iters=4)  
        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):

            sum_grad = tg + g
            avg_grad = sum_grad / self.accum_iters_float

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))
            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))
            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints

class LegacyInstanceNormalization(Layer):

    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(LegacyInstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#1-hot MSA to PSSM
def msa2pssm(msa1hot, w):
    beff = tf.reduce_sum(w)
    f_i = tf.reduce_sum(w[:,None,None]*msa1hot, axis=0) / beff + 0.005#1e-9
    h_i = tf.reduce_sum( -f_i * tf.math.log(f_i), axis=1)
    return tf.concat([f_i, h_i[:,None]], axis=1)

#Reweight MSA based on cutoff
def reweight(msa1hot, cutoff):
    with tf.name_scope('reweight'):
        id_min = tf.cast(tf.shape(msa1hot)[1], tf.float32) * cutoff
        id_mtx = tf.tensordot(msa1hot, msa1hot, [[1,2], [1,2]])
        id_mask = id_mtx > id_min
        w = 1.0/tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32),-1)
    return w

#Shrunk covariance inversion
def fast_dca(msa1hot, weights, penalty = 4.5):

    nr = tf.shape(msa1hot)[0]
    nc = tf.shape(msa1hot)[1]
    ns = tf.shape(msa1hot)[2]

    with tf.name_scope('covariance'):
        x = tf.reshape(msa1hot, (nr, nc * ns))
        num_points = tf.reduce_sum(weights) - tf.sqrt(tf.reduce_mean(weights))
        mean = tf.reduce_sum(x * weights[:,None], axis=0, keepdims=True) / num_points
        x = (x - mean) * tf.sqrt(weights[:,None])
        cov = tf.matmul(tf.transpose(x), x)/num_points

    with tf.name_scope('inv_convariance'):
        cov_reg = cov + tf.eye(nc * ns) * penalty / tf.sqrt(tf.reduce_sum(weights))
        inv_cov = tf.linalg.inv(cov_reg)

        x1 = tf.reshape(inv_cov,(nc, ns, nc, ns))
        x2 = tf.transpose(x1, [0,2,1,3])
        features = tf.reshape(x2, (nc, nc, ns * ns))

        x3 = tf.sqrt(tf.reduce_sum(tf.square(x1[:,:-1,:,:-1]),(1,3))) * (1-tf.eye(nc))
        apc = tf.reduce_sum(x3,0,keepdims=True) * tf.reduce_sum(x3,1,keepdims=True) / tf.reduce_sum(x3)
        contacts = (x3 - apc) * (1-tf.eye(nc))

    return tf.concat([features, contacts[:,:,None]], axis=2)

#Collect input features (keras code)
def keras_collect_features(inputs, wmin=0.8) :
    f1d_seq_batched, msa1hot_batched = inputs

    f1d_seq = f1d_seq_batched[0, ...]
    msa1hot = msa1hot_batched[0, ...]

    nrow = K.shape(msa1hot)[0]
    ncol = K.shape(msa1hot)[1]

    w = reweight(msa1hot, wmin)

    # 1D features
    f1d_pssm = msa2pssm(msa1hot, w)

    f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
    f1d = tf.expand_dims(f1d, axis=0)
    f1d = tf.reshape(f1d, [1,ncol,42])

    # 2D features
    f2d_dca = tf.cond(nrow>1, lambda: fast_dca(msa1hot, w), lambda: tf.zeros([ncol,ncol,442], tf.float32))
    f2d_dca = tf.expand_dims(f2d_dca, axis=0)

    f2d = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,ncol,1]), 
                    tf.tile(f1d[:,None,:,:], [1,ncol,1,1]),
                    f2d_dca], axis=-1)
    f2d = tf.reshape(f2d, [1,ncol,ncol,442+2*42])

    return f2d

#Collect input features (tf code)
def pssm_func(inputs, diag=0.0):
    x,y = inputs
    _,_,L,A = [tf.shape(y)[k] for k in range(4)]
    with tf.name_scope('1d_features'):
        # sequence
        x_i = x[0,:,:20]
        # pssm
        f_i = y[0,0, :, :]
        # entropy
        h_i = tf.zeros((L,1))
        #h_i = K.sum(-f_i * K.log(f_i + 1e-8), axis=-1, keepdims=True)
        # tile and combined 1D features
        feat_1D = tf.concat([x_i,f_i,h_i], axis=-1)
        feat_1D_tile_A = tf.tile(feat_1D[:,None,:], [1,L,1])
        feat_1D_tile_B = tf.tile(feat_1D[None,:,:], [L,1,1])

    with tf.name_scope('2d_features'):
        ic = diag * tf.eye(L*A)
        ic = tf.reshape(ic,(L,A,L,A))
        ic = tf.transpose(ic,(0,2,1,3))
        ic = tf.reshape(ic,(L,L,A*A))
        i0 = tf.zeros([L,L,1])
        feat_2D = tf.concat([ic,i0], axis=-1)

        feat = tf.concat([feat_1D_tile_A, feat_1D_tile_B, feat_2D],axis=-1)
        return tf.reshape(feat, [1,L,L,442+2*42])

def load_trrosetta_model(model_path) :

    saved_model = load_model(model_path, custom_objects = {
        'InstanceNormalization' : LegacyInstanceNormalization,
        'reweight' : reweight,
        'wmin' : 0.8,
        'msa2pssm' : msa2pssm,
        'tf' : tf,
        'fast_dca' : fast_dca,
        'keras_collect_features' : pssm_func
    })
    
    return saved_model

def _get_kl_divergence_keras(p_dist, p_theta, p_phi, p_omega, t_dist, t_theta, t_phi, t_omega) :
    
    kl_dist = K.mean(K.sum(t_dist * K.log(t_dist / p_dist), axis=-1), axis=(-1, -2))
    kl_theta = K.mean(K.sum(t_theta * K.log(t_theta / p_theta), axis=-1), axis=(-1, -2))
    kl_phi = K.mean(K.sum(t_phi * K.log(t_phi / p_phi), axis=-1), axis=(-1, -2))
    kl_omega = K.mean(K.sum(t_omega * K.log(t_omega / p_omega), axis=-1), axis=(-1, -2))
    
    return K.mean(kl_dist + kl_theta + kl_phi + kl_omega, axis=1)

def optimize_trrosetta_scores(predictor, x, batch_size, n_iters, input_background, drop=None, scrambler_mode='inclusion', norm_mode='instance', adam_accum_iters=2, adam_lr=0.01, adam_beta_1=0.5, adam_beta_2=0.9, n_samples=4, sample_mode='gumbel', entropy_mode='target', entropy_bits=0., entropy_weight=1.) :

    if not isinstance(x, list) :
        x = [x]

    group = [np.zeros((x[0].shape[0], 1))]

    if not isinstance(group, list) :
        group = [group]

    if drop is None :
        drop = [np.ones((x[0].shape[0], x[0].shape[1], x[0].shape[2], 1))]

    if not isinstance(drop, list) :
        drop = [drop]
    
    input_background_log = np.log(input_background)

    #Freeze predictor
    predictor.trainable = False
    predictor.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss='mean_squared_error')
    
    seq_length = x[0].shape[2]

    #Build optimization model
    scrambler = load_optimization_model(
        batch_size,
        1,
        seq_length,
        scrambler_mode=scrambler_mode,
        n_out_channels=20,
        n_spatial_dims=1,
        mask_smoothing=False,
        smooth_window_size=1,
        norm_mode=norm_mode
    )

    #Load sampler
    sampler = build_sampler(batch_size, 1, seq_length, n_classes=1, n_samples=n_samples, sample_mode=sample_mode, n_channels=20, model_prefix='')

    #Build scrambler model
    scrambler_class = Input(shape=(1,), name='scrambler_class')
    scrambler_input = Input(shape=(1, seq_length, 20), name='scrambler_input')
    scrambler_drop = Input(shape=(1, seq_length, 1), name='scrambler_drop')

    scrambled_logit, importance_score = scrambler(scrambler_input, scrambler_drop)

    pwm_logit, pwm, sampled_pwm, _, _ = sampler(scrambler_class, scrambled_logit)

    scrambler_model = Model([scrambler_class, scrambler_input, scrambler_drop], [pwm, sampled_pwm, importance_score])

    #Initialize Templates and Masks
    initialize_templates(scrambler_model, [np.zeros((1, seq_length, 20))], [input_background_log[None, ...]], model_prefix='')

    scrambler_model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
        loss='mean_squared_error'
    )

    #Build loss model pipeline
    
    #Define model inputs
    scrambler_class = Input(batch_shape=(1, 1), name='t_scrambler_class')
    scrambler_input = Input(batch_shape=(1, 1, seq_length, 20), name='t_scrambler_input')
    scrambler_drop = Input(batch_shape=(1, 1, seq_length, 1), name='t_scrambler_drop')

    scrambled_in, importance_scores = scrambler(scrambler_input, scrambler_drop)

    #Run sampler
    _, scrambled_pwm, scrambled_sample, pwm_mask, sampled_mask = sampler(scrambler_class, scrambled_in)

    #Define layer to deflate sample axis
    deflate_scrambled_sample = Lambda(lambda x: K.reshape(x, (batch_size * n_samples, 1, seq_length, 20)), name='deflate_scrambled_sample')

    #Deflate sample axis
    scrambled_sample_deflated = deflate_scrambled_sample(scrambled_sample)

    #Make reference prediction on non-scrambled input sequence
    collapse_input_layer_non_scrambled = Lambda(lambda x: x[:, 0, :, :], output_shape=(seq_length, 20))
    create_msa_layer_non_scrambled = Lambda(lambda x: K.concatenate([x, K.zeros((x.shape[0], x.shape[1], x.shape[2], 1))], axis=-1), output_shape=(1, seq_length, 21))
    collapsed_in_non_scrambled = collapse_input_layer_non_scrambled(scrambler_input)
    collapsed_in_non_scrambled_msa = create_msa_layer_non_scrambled(scrambler_input)

    p_dist_non_scrambled_deflated, p_theta_non_scrambled_deflated, p_phi_non_scrambled_deflated, p_omega_non_scrambled_deflated = predictor([collapsed_in_non_scrambled, collapsed_in_non_scrambled_msa])

    #Make prediction on scrambled sequence samples
    collapse_input_layer = Lambda(lambda x: x[:, 0, :, :], output_shape=(seq_length, 20))
    create_msa_layer = Lambda(lambda x: K.concatenate([x, K.zeros((x.shape[0], x.shape[1], x.shape[2], 1))], axis=-1), output_shape=(1, seq_length, 21))
    collapsed_in = collapse_input_layer(scrambled_sample_deflated)
    collapsed_in_msa = create_msa_layer(scrambled_sample_deflated)

    p_dist_scrambled_deflated, p_theta_scrambled_deflated, p_phi_scrambled_deflated, p_omega_scrambled_deflated = predictor([collapsed_in, collapsed_in_msa])

    #Define layer to inflate sample axis
    inflate_dist_target = Lambda(lambda x: K.expand_dims(x, axis=1), name='inflate_dist_target')
    inflate_theta_target = Lambda(lambda x: K.expand_dims(x, axis=1), name='inflate_theta_target')
    inflate_phi_target = Lambda(lambda x: K.expand_dims(x, axis=1), name='inflate_phi_target')
    inflate_omega_target = Lambda(lambda x: K.expand_dims(x, axis=1), name='inflate_omega_target')

    inflate_dist_prediction = Lambda(lambda x: K.reshape(x, (batch_size, n_samples, seq_length, seq_length, 37)), name='inflate_dist_prediction')
    inflate_theta_prediction = Lambda(lambda x: K.reshape(x, (batch_size, n_samples, seq_length, seq_length, 25)), name='inflate_theta_prediction')
    inflate_phi_prediction = Lambda(lambda x: K.reshape(x, (batch_size, n_samples, seq_length, seq_length, 13)), name='inflate_phi_prediction')
    inflate_omega_prediction = Lambda(lambda x: K.reshape(x, (batch_size, n_samples, seq_length, seq_length, 25)), name='inflate_omega_prediction')

    #Inflate sample axis
    p_dist_non_scrambled = inflate_dist_target(p_dist_non_scrambled_deflated)
    p_theta_non_scrambled = inflate_theta_target(p_theta_non_scrambled_deflated)
    p_phi_non_scrambled = inflate_phi_target(p_phi_non_scrambled_deflated)
    p_omega_non_scrambled = inflate_omega_target(p_omega_non_scrambled_deflated)

    p_dist_scrambled = inflate_dist_prediction(p_dist_scrambled_deflated)
    p_theta_scrambled = inflate_theta_prediction(p_theta_scrambled_deflated)
    p_phi_scrambled = inflate_phi_prediction(p_phi_scrambled_deflated)
    p_omega_scrambled = inflate_omega_prediction(p_omega_scrambled_deflated)
    
    #NLL cost
    nll_loss_func = _get_kl_divergence_keras
    scrambler_mode_coeff = 1.
    if scrambler_mode == 'occlusion' :
        scrambler_mode_coeff = -1.

    #Entropy cost
    entropy_loss_func = None
    if entropy_mode == 'target' :
        entropy_loss_func = get_target_entropy_sme_masked(x_start=0, x_end=1, y_start=0, y_end=seq_length, target_bits=entropy_bits)
    elif entropy_mode == 'maximization' :
        entropy_loss_func = get_margin_entropy_ame_masked(x_start=0, x_end=1, y_start=0, y_end=seq_length, max_bits=entropy_bits)
    
    #Define background matrix embedding
    seq_reshape_layer = Reshape((1, seq_length, 20))

    x_mean_dense = Embedding(1, seq_length * 20, embeddings_initializer='zeros', name='x_mean_dense')

    x_mean = seq_reshape_layer(x_mean_dense(scrambler_class))
    
    #Execute NLL cost
    nll_loss = Lambda(lambda x: K.reshape(K.sum(scrambler_mode_coeff * nll_loss_func(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]), axis=0), (1,)), name='nll')([
        p_dist_non_scrambled,
        p_theta_non_scrambled,
        p_phi_non_scrambled,
        p_omega_non_scrambled,
        p_dist_scrambled,
        p_theta_scrambled,
        p_phi_scrambled,
        p_omega_scrambled
    ])

    #Execute entropy cost
    entropy_loss = Lambda(lambda x: K.reshape(K.sum(entropy_loss_func(x[0], x[1], x[2]), axis=0), (1,)), name='entropy')([
        scrambled_pwm,
        pwm_mask,
        x_mean
    ])
    
    loss_model = Model(
        [scrambler_class, scrambler_input, scrambler_drop],
        [nll_loss, entropy_loss]
    )

    #Initialize Templates and Masks
    initialize_templates(loss_model, [np.zeros((1, seq_length, 20))], [input_background_log[None, ...]], model_prefix='')

    #Initialize Sequence Length Parameters
    initialize_backgrounds(loss_model, [input_background], model_prefix='')

    opt = AdamAccumulate(lr=adam_lr, beta_1=adam_beta_1, beta_2=adam_beta_2, accum_iters=adam_accum_iters)

    loss_model.compile(
        optimizer=opt,
        loss={
            'nll' : get_weighted_loss(loss_coeff=1.0),
            'entropy' : get_weighted_loss(loss_coeff=entropy_weight)
        }
    )

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
    _reset_generator(scrambler_model, verbose=False)
    _reset_generator(loss_model, verbose=False)
    _reset_optimizer(opt, verbose=False)

    #Execute training procedure
    dummy = np.zeros((batch_size, 1))

    input_tensors = group + x + drop

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
    pwms = []
    samples = []
    scores = []
    for batch_ix in range(n_batches) :

        print("Finetuning batch " + str(batch_ix) + "...")

        input_tensors_batch = [
            input_tensor[batch_ix*batch_size:(batch_ix+1)*batch_size] for input_tensor in input_tensors
        ]

        train_history = LossHistory(loss_names=['nll', 'entropy'])

        # train the autoscrambler
        _ = loss_model.fit(
            input_tensors_batch,
            [dummy, dummy],
            epochs=1,
            steps_per_epoch=n_iters,
            #batch_size=batch_size,
            callbacks=[train_history]
        )

        pred_bundle = scrambler_model.predict(x=input_tensors_batch, batch_size=batch_size, verbose=False)

        if n_pad > 0 :
            pred_bundle = [
                pred_bundle_member[:-n_pad, ...] for pred_bundle_member in pred_bundle
            ]

        pwms.append(pred_bundle[0])
        samples.append(pred_bundle[1])
        scores.append(pred_bundle[2])

        train_histories.append(train_history.loss_dict)

        #Reset mask
        _reset_generator(scrambler_model)
        _reset_generator(loss_model)
        _reset_optimizer(opt)

    pwms = np.concatenate(pwms, axis=0)
    samples = np.concatenate(samples, axis=0)
    scores = np.concatenate(scores, axis=0)

    return pwms, samples, scores, train_histories

