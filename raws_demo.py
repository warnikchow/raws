import numpy as np
import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.01
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

print('\n\n\n\n\n\n\n\n\n\n\n')

print('###############################################################\n#                                                             #\n# Demonstration : Real-time Automatic Word Segmentation (K/E) #\n#                                                             #\n###############################################################')

import fasttext
import re

print('\nImporting dictionaries...')

dic_kor = fasttext.load_model('vectors/model_kor.bin')
def loadvector(File):
    f = open(File,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    return model
dic_eng = loadvector('vectors/model_eng.txt')

import string
idchar = {}
for i in range(len(string.ascii_lowercase)):
  idchar.update({string.ascii_lowercase[i]:i})

for i in range(10):
  idchar.update({i:i+26})

idchar.update({'#':36})

big  = re.compile(r"[A-Z]")
small= re.compile(r"[a-z]")
num  = re.compile(r"[0-9]")

print('Loading models...')

from keras.models import load_model
model_kor = load_model('model/model_kor.hdf5')
model_eng = load_model('model/model_eng.hdf5')

print('\nEnter "bye" to quit\n')

## Functions_KOR

threshold_kor=0.5
overlap=30

def pred_correction_rnn(sent,model,dic,maxlen,wdim):
    conv = np.zeros((1,maxlen,wdim,1))
    rnn  = np.zeros((1,maxlen,wdim))
    charcount = -1
    for j in range(len(sent)):
      if j<maxlen and sent[j]!=' ':
        charcount=charcount+1
        conv[0][charcount,:,0]=dic[sent[j]]
        rnn[0][charcount,:]=dic[sent[j]]
    z = model.predict([conv,rnn])[0]
    sent_raw = ''
    count_char=-1
    lastpoint=-1
    lastchar=-1
    for j in range(len(sent)):
      if sent[j]!=' ':
        count_char=count_char+1
        sent_raw = sent_raw+sent[j]
        if z[count_char]>threshold_kor:
          sent_raw = sent_raw+' '
          if j<overlap:
            lastpoint=len(sent_raw)
            lastchar=j
    return sent_raw, lastpoint, lastchar

def kor_spacing(s):
    if len(s)<overlap:
      temp,lp,lc = pred_correction_rnn(s,model_kor,dic_kor,100,100)
      z = temp+"\n"
    else:
      z=''
      start=0
      while start<len(s):
        if start+overlap<len(s):
          temp,lp,lc =pred_correction_rnn(s[start:start+2*overlap],model_kor,dic_kor,100,100)
          temp=temp[:lp]
        else:
          temp,lp,lc =pred_correction_rnn(s[start:],model_kor,dic_kor,100,100)
          lc = overlap
        z = z+temp
        start=start+lc+1
      z = z+"\n"
    print('>> Output:',z)
    return z

## Function_ENG

def underscore(hashtag):
    result=''
    for i in range(len(hashtag)):
      if i>0:
        if hashtag[i].isalpha()==True:
          result = result+hashtag[i]
        else:
          result = result+' '
    return result

def split_hashtag(hashtagestring):
    fo = re.compile(r'#[A-Z]{2,}(?![a-z])|[A-Z][a-z]+')
    fi = fo.findall(hashtagestring)
    result = ''
    for var in fi:
        result += var + ' '
    return result

threshold=0.35

def hash_pred(sent,model,dic1,dic2,maxlen,wdim):
    conv = np.zeros((1,maxlen,wdim,1))
    rnn  = np.zeros((1,maxlen,len(dic2)))
    charcount=-1
    lastpoint=-1
    lastchar=-1
    for j in range(len(sent)):
      if charcount<maxlen-1 and sent[j]!=' ':
        charcount=charcount+1
        if sent[j] in dic1:
          conv[0][charcount,:,0]=dic1[sent[j]]
        if sent[j] in dic2:
          rnn[0][charcount,dic2[sent[j]]]=1
    z = model.predict([conv,rnn])[0]
    print(z)
    sent_raw = ''
    count_char=-1
    for j in range(len(sent)):
      if sent[j]!=' ':
        count_char=count_char+1
        sent_raw = sent_raw+sent[j]
        if z[count_char]>threshold:
          sent_raw = sent_raw+' '
          if j<overlap:
            lastpoint=len(sent_raw)
            lastchar=j
    return sent_raw, z[:count_char], count_char, lastpoint, lastchar

def hash_space(tag):
    tag_re = ''
    for i in range(len(tag)):
      if tag[i].isalpha() == True:
        tag_re = tag_re+tag[i].lower()
      else:
        tag_re = tag_re+tag[i]
    sent_raw, z, count_char, lastpoint, lastchar = hash_pred(tag_re,model_eng,dic_eng,idchar,100,100)
    return sent_raw, lastpoint, lastchar

def eng_spacing(s):
    if len(s)<overlap:
      temp,lp,lc = hash_space(s)
      z = temp+"\n"
    else:
      z=''
      start=0
      while start<len(s):
        if start+overlap<len(s):
          temp,lp,lc =hash_space(s[start:start+2*overlap])
          temp=temp[:lp]
        else:
          temp,lp,lc =hash_space(s[start:])
          lc = overlap
        z = z+temp
        start=start+lc+1
      z = z+"\n"
    print('>> Output:',z)
    return z

def eng_hashsegment(hashtag):
    if '_' in hashtag:
      print('>> output:',underscore(hashtag))
      return underscore(hashtag)
    else:
      if re.search(big,hashtag) and  re.search(small,hashtag):
        print('>> output:',split_hashtag(hashtag))
        return split_hashtag(hashtag)
      else:
        return eng_spacing(hashtag[1:])

print('\nEnter "k" for Korean spacing\nEnter "e" for English segmentation\nEnter "bye" to quit\n\n')

## Demonstration

def KOR():
  print('\nAutomatic Korean spacing ...\nEnter "e" to activate English segmentation\nEnter "bye" to quit\n')
  while 1:
    s = input('>> You say: ')
    if s == 'bye':
        sys.exit()
    elif s == 'e':
        ENG()
    else:
        kor_spacing(s)

def ENG():
  print('\nAutomatic English segmentation ...\nEnter "k" to activate identification mode\nEnter "bye" to quit\n')
  while 1:
    s = input('>> You say: ')
    if s == 'bye':
        sys.exit()
    elif s == 'k':
        KOR()
    else:
        eng_spacing(s)

while 1:
  s = input(' Choose: ')
  if s == 'k':
    KOR()
  elif s == 'e':
    ENG()
  elif s == 'bye':
    sys.exit()







