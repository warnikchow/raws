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
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, LSTM, GRU, SimpleRNN, Dense, Lambda
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import keras.layers as layers

from keras import optimizers
adam_half = optimizers.Adam(lr=0.0005)
adam_half_2 = optimizers.Adam(lr=0.0002)

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

def featurize_space(sent,maxlen):
    onehot = np.zeros(maxlen)
    countchar = -1
    for i in range(len(sent)-1):
      if sent[i]!=' ' and i<maxlen:
        countchar=countchar+1
        if sent[i+1]==' ':
          onehot[countchar] = 1
    return onehot

def featurize_corpus(corpus,dic,wdim,maxlen):
    onehot = np.zeros((len(corpus),maxlen))
    conv = np.zeros((len(corpus),maxlen,wdim,1))
    rnn  = np.zeros((len(corpus),maxlen,wdim))
    for i in range(len(corpus)):
      if i%1000 == 0:
        print(i)
      onehot[i,:] = featurize_space(corpus[i],maxlen)
      countchar = -1
      for j in range(len(corpus[i])-1):
        if countchar<maxlen-1 and corpus[i][j]!=' ':
          countchar=countchar+1
          conv[i][countchar,:,0]=dic[corpus[i][j]]
          rnn[i][countchar,:]=dic[corpus[i][j]]
    return onehot, conv, rnn

raw_onehot100, raw_conv100, raw_rnn100  = featurize_corpus(drama_raw,model_drama,100,100)

def correction_model(conv,rnn,original,maxlen,wdim,hidden_dim,filename):
    cnn_input = Input(shape=(maxlen,wdim,1), dtype='float32')
    cnn_layer = layers.Conv2D(32,(3,wdim),activation='relu')(cnn_input)
    cnn_layer = layers.MaxPooling2D((2,1))(cnn_layer)
    cnn_layer = layers.Conv2D(32,(3,1),activation='relu')(cnn_layer)
    cnn_layer = layers.Flatten()(cnn_layer)
    cnn_output= Dense(hidden_dim, activation='relu')(cnn_layer)
    cnn_output= Dropout(0.3)(cnn_output)
    rnn_input = Input(shape=(maxlen,wdim), dtype='float32')
    rnn_layer = Bidirectional(LSTM(32,return_sequences=True))(rnn_input)
    attention = Dense(maxlen, activation='relu')(cnn_output)
    attention = layers.Reshape((maxlen,1))(attention)
    rnn_layer = layers.multiply([attention,rnn_layer])
    rnn_hsum  = Lambda(lambda x: K.sum(x, axis=1))(rnn_layer)
    rnn_hsum  = Dropout(0.3)(rnn_hsum)
    output    = Dense(hidden_dim, activation='relu')(rnn_hsum)
    output    = Dropout(0.3)(output)
    output    = Dense(hidden_dim, activation='relu')(output)
    main_output = Dense(maxlen,activation='sigmoid')(output)
    model = Sequential()
    model = Model(inputs=[cnn_input,rnn_input], outputs=[main_output])
    model.compile(optimizer=adam_half, loss="binary_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [checkpoint]
    model.summary()
    model.fit(segmented_conv,original,validation_split=0.01,epochs=20,batch_size=32,callbacks=callbacks_list)

correction_model([raw_conv100,raw_rnn100],raw_onehot100,100,100,64,'modelcws/rnnconvdnn100')

def correction_model_concat(conv,rnn,original,maxlen,wdim,hidden_dim,filename):
    cnn_input = Input(shape=(maxlen,wdim,1), dtype='float32')
    cnn_layer = layers.Conv2D(32,(3,wdim),activation='relu')(cnn_input)
    cnn_layer = layers.MaxPooling2D((2,1))(cnn_layer)
    cnn_layer = layers.Conv2D(32,(3,1),activation='relu')(cnn_layer)
    cnn_layer = layers.Flatten()(cnn_layer)
    rnn_input = Input(shape=(maxlen,wdim), dtype='float32')
    rnn_layer = Bidirectional(LSTM(32,return_sequences=True))(rnn_input)
    rnn_layer = layers.Flatten()(rnn_layer)
    rnn_layer = layers.concatenate([cnn_layer,rnn_layer])
    rnn_layer = Dense(hidden_dim, activation='relu')(rnn_layer)
    rnn_layer = Dropout(0.3)(rnn_layer)
    output    = Dense(hidden_dim, activation='relu')(rnn_layer)
    output    = Dropout(0.3)(output)
    main_output = Dense(maxlen,activation='sigmoid')(output)
    model = Sequential()
    model = Model(inputs=[cnn_input,rnn_input], outputs=[main_output])
    model.compile(optimizer=adam_half, loss="binary_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [checkpoint]
    model.summary()
    model.fit([conv,rnn],original,validation_split=0.1,epochs=30,batch_size=128,callbacks=callbacks_list)

correction_model_concat(raw_conv100,raw_rnn100,raw_onehot100,100,100,128,'modelcwsre/rnnconvdnn100_sigmoid_concat')

from keras.models import load_model
model_corr100 = load_model('modelcws/convdnn100-16-0.9693.hdf5')

def pred_correction(sent,model,dic,maxlen,wdim):
    conv = np.zeros((1,maxlen,wdim,1))
    charcount=-1
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

def correct(s):
    #x = pred_correction(s,model_corr200,model_drama,200,100)
    y = pred_correction(s,model_corr100,model_drama,100,100)
    #z = pred_correction(s,model_corr50,model_drama,50,100)+"\n"
    #print('by200: ',x)
    print('by100: ',y)
    #print('by50 : ',z)	

correct('오늘저녁메뉴는뭐야')
