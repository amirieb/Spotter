# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.

Two digits inverted:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs

Three digits inverted:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs

Four digits inverted:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs

Five digits inverted:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs
'''
from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, Dropout, RepeatVector, recurrent
from keras.engine.training import slice_X
import cPickle

class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)
class colors:
    not_spurious = '\033[92m'
    spurious = '\033[91m'
    close = '\033[0m'


# PARAMETERS
DIGITS = 4
INVERT = True
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
MAXLEN = DIGITS + 1 + DIGITS
noise_rate = 0.1  # noise_rate: fraction of noisy instances in data, could be in np.arange(.1, .6, .1)
iterations = 128 # set to 2^i     
acc_thr = .8  # accuracy threshold: if network accuracy against an instance is >= acc_thr, the spotter treats the instance as correctly classified, a smaller value than 1.0 is used for more flexibility.
chars = '0123456789+ '
ctable = CharacterTable(chars, MAXLEN)

 
#LOAD DATA
X_train = cPickle.load(open('data_addition/add_xtrain_' + str(noise_rate), 'rb')) 
y_train = cPickle.load(open('data_addition/add_ytrain_' + str(noise_rate), 'rb'))
X_val = cPickle.load(open('data_addition/add_xval_' + str(noise_rate), 'rb'))
y_val = cPickle.load(open('data_addition/add_yval_' + str(noise_rate), 'rb'))
# X_test = cPickle.load(open('data_addition/add_xtest_' + str(noise_rate), 'rb'))
# y_test = cPickle.load(open('data_addition/add_ytest_' + str(noise_rate), 'rb'))    
print ('shape of training data:  ', X_train.shape, y_train.shape)
print ('shape of validation data:', X_val.shape, y_val.shape)
# print ('shape - test data', X_test.shape, y_test.shape)


# BUILD THE NETWORK
print('Build model..')
RNN = recurrent.LSTM
model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
model.add(Dropout(0.5))
model.add(RepeatVector(DIGITS + 1))
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print (model.summary())


# SIMULTANEOUSLY TRAIN THE NETWORK & IDENTIFY SPURIOUS INSTANCES IN TRAINING DATASET.
# IF YOU WANT TO IDENTIFY ALL SPURIOUS INSTANCES IN YOUR DATASET, USE THE ENTIRE DATA FOR TRAINING
# SET kern='lit' WHEN FITTING THE MODEL AS BELOW: 
history, n_instances, q0_sorted = model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=iterations,
                                           validation_data=(X_val, y_val), kern='lit', acc_thr=acc_thr)


# SHOW A RANKED LIST OF SPURIOUS INSTANCES
max_instances = 50  # show top max_rank spurious instances
print ('\n--------------------------')
print ('ranked list of top', max_instances, 'spurious instances in dataset')
print('is_spurious? | instance | answer_in_dataset')
rank = 1
for item in q0_sorted:
    if rank >= max_instances:
        break
    rank += 1
    rowX, rowy = X_train[np.array([item])], y_train[np.array([item])]
    q = ctable.decode(rowX[0])
    q = str(q[::-1] if INVERT else q)
    ans_gold = ctable.decode(rowy[0])
    # preds = model.predict_classes(rowX, verbose=0)# get network prediction for the instance      
    # ans_pred = ctable.decode(preds[0], calc_argmax=False)
    ab = q.split('+') 
    if int(ab[0]) + int(ab[1]) == int(ans_gold.strip()):
        print(colors.not_spurious + '☒' + colors.close, end=' ')
    else:
        print(colors.spurious + '☑' + colors.close, end=' ')            
    print (q, ans_gold)
    print ('  -------------')

del model

