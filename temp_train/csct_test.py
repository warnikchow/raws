import numpy as np
import sys

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

import fasttext

model_drama = fasttext.load_model('vectors/model_drama.bin')
drama = read_data('data/drama_total_cor.txt')
drama_raw   = [row[0] for row in drama]

import itertools
drama_allchar = itertools.chain.from_iterable(drama_raw)
drama_idchar = {token: idx for idx, token in enumerate(set(drama_allchar))}

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, LSTM, GRU, SimpleRNN, Dense, Lambda
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import keras.layers as layers

from keras import optimizers
adam_half = optimizers.Adam(lr=0.0005)

from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding

from random import random
from numpy import array
from numpy import cumsum
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

from keras.models import load_model
model_corr100 = load_model('modelcws/rnnconv100_re-09-0.9775.hdf5')

def pred_correction(sent,model,dic,maxlen,wdim):
    conv = np.zeros((1,maxlen,wdim,1))
    charcount = -1
    for j in range(len(sent)):
      if j<maxlen and sent[j]!=' ':
        charcount=charcount+1
        conv[0][charcount,:,0]=dic[sent[j]]
    z = model.predict(conv)[0]
    print(z)
    sent_raw = ''
    count_char=-1
    for j in range(len(sent)):
      if sent[j]!=' ':
        count_char=count_char+1
        sent_raw = sent_raw+sent[j]
        if z[count_char]>0.5:
          sent_raw = sent_raw+' '
    return sent_raw

def pred_correction_rnn(sent,model,dic,maxlen,wdim):
    conv = np.zeros((1,maxlen,wdim,1))
    rnn  = np.zeros((1,maxlen,wdim))
    charcount = -1
    for j in range(len(sent)):
      if j<maxlen and sent[j]!=' ':
        charcount=charcount+1
        conv[0][charcount,:,0]=dic[sent[j]]
        rnn[0][charcount,:]=dic[sent[j]]
    z = model.predict([rnn,conv])[0]
    print(z)
    sent_raw = ''
    count_char=-1
    for j in range(len(sent)):
      if sent[j]!=' ':
        count_char=count_char+1
        sent_raw = sent_raw+sent[j]
        if z[count_char]>0.5:
          sent_raw = sent_raw+' '
    return sent_raw	

def correct(s):
    z = pred_correction_rnn(s,model_corr100,model_drama,100,100)+"\n"
    print('by100:',z)	

correct('아버지친구분당선되셨더라')
correct('뭣이중헌지도모름서')
correct('나너본지두달다돼감')
