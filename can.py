#!/usr/bin/env python3

#  -*- coding: utf-8 -*-

#                              _____
#                             |     |
#                      |----->| EVE |--> P_EVE
#                      |      |_____|
#                      |
#         _______      |       _____
# P ---> |       |     |      |     |
#        | ALICE |-->| C |--->| BOB |--> P_BOB
# K -+-> |_______|         |->|_____|
#    |-------------------->|

import numpy as np



N=16

P = np.random.choice([-1,1],size=(1,N))
P_BOB = np.random.choice([-1,1],size=(1,N))
KK = np.random.choice([-1,1],size=(1,N))

print(P)

from keras import Input
from keras.layers import Dense, Conv1D, Reshape, concatenate
from keras.engine import Model
from keras.models import *
from keras.optimizers import *

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def bob_loss(y_true, y_pred):
    return K.mean(K.sum(K.abs(y_true-y_pred), axis=-1))

def eve_loss(y_true, y_pred):
    return K.mean(K.sum(K.abs(y_true - y_pred), axis=-1))

#(N/2 − Eve L1 error)2/(N/2)2

plaintext = Input(shape=(N,))
cyphertext = Input(shape=(N,))
key = Input(shape=(N,))

alice_input = concatenate([plaintext, key])

alice_ff1 = Dense(2*N)(alice_input)
alice_reshape1 = Reshape((2*N, 1), input_shape=(2*N,))(alice_ff1)
alice_cnn1 = Conv1D(2, (4,), padding='same', activation='sigmoid')(alice_reshape1)
alice_cnn2 = Conv1D(4, (2,), padding='same', strides=2, activation='sigmoid')(alice_cnn1)
alice_cnn3 = Conv1D(4, (4,), padding='same', activation='sigmoid')(alice_cnn2)
alice_cnn4 = Conv1D(1, (1,), padding='same', activation='tanh')(alice_cnn3)
alice_reshape2 = Reshape((N,), input_shape=(N, 1))(alice_cnn4)

alice_model = Model(inputs=[plaintext, key], outputs=[alice_reshape2])
alice_model.compile(loss='categorical_crossentropy', optimizer='adam')

bob_input = concatenate([cyphertext, key])

bob_ff1 = Dense(2*N)(bob_input)
bob_reshape1 = Reshape((2*N, 1), input_shape=(2*N,))(bob_ff1)
bob_cnn1 = Conv1D(2, (4,), padding='same', activation='sigmoid')(bob_reshape1)
bob_cnn2 = Conv1D(4, (2,), padding='same', strides=2, activation='sigmoid')(bob_cnn1)
bob_cnn3 = Conv1D(4, (4,), padding='same', activation='sigmoid')(bob_cnn2)
bob_cnn4 = Conv1D(1, (1,), padding='same', activation='tanh')(bob_cnn3)
bob_reshape2 = Reshape((N,), input_shape=(N, 1))(bob_cnn4)

bob_model = Model(inputs=[cyphertext, key], outputs=[bob_reshape2])
bob_model.compile(loss='categorical_crossentropy', optimizer='adam')

eve_ff1 = Dense(2*N)(cyphertext)
eve_reshape1 = Reshape((2*N, 1), input_shape=(2*N,))(eve_ff1)
eve_cnn1 = Conv1D(2, (4,), padding='same', activation='sigmoid')(eve_reshape1)
eve_cnn2 = Conv1D(4, (2,), padding='same', strides=2, activation='sigmoid')(eve_cnn1)
eve_cnn3 = Conv1D(4, (4,), padding='same', activation='sigmoid')(eve_cnn2)
eve_cnn4 = Conv1D(1, (1,), padding='same', activation='tanh')(eve_cnn3)
eve_reshape2 = Reshape((N,), input_shape=(N, 1))(eve_cnn4)

eve_model = Model(inputs=[cyphertext], outputs=[eve_reshape2])
eve_model.compile(loss='categorical_crossentropy', optimizer='adam')

optim = Adam(lr=0.0008)

ab_model = Model(inputs=[plaintext,key], outputs=bob_model([alice_model([plaintext,key]), key]))
ab_model.compile(optimizer=optim, loss=bob_loss)

ae_model = Model(inputs=[plaintext,key], outputs=eve_model(alice_model([plaintext,key])))
ae_model.compile(optimizer=optim, loss=eve_loss)

max_steps = 10
minibatch_size = 4096
step = 0
while step < max_steps:
    step += 1

    P_batch = np.random.choice([-1, 1], size=(minibatch_size, 16))
    K_batch = np.random.choice([-1, 1], size=(minibatch_size, 16))