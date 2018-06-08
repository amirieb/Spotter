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
class generate_data:
    def generate_noisy_data(self):        
        DATA_SIZE = 14000
        TRAINING_SIZE = 10000
        SPLIT_AT = 2000
        for noise_rate in np.arange(.1, 1., .1):
              
            NOISE_SIZE = int (noise_rate * TRAINING_SIZE)
          
            questions = []
            expected = []
            seen = set()
            print('Generating data...')
                 
            count_wrong_inject = 0
                 
            while len(questions) < DATA_SIZE:
                f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
                a, b, c = f(), f(), f()
                while c == b:
                    c = f()
                # Skip any addition questions we've already seen
                # Also skip any such that X+Y == Y+X (hence the sorting)
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                seen.add(key)
                # Pad the data with spaces such that it is always MAXLEN
                q = '{}+{}'.format(a, b)
                query = q + ' ' * (MAXLEN - len(q))
                ans = ''
                if count_wrong_inject < NOISE_SIZE:
                    ans = str(a + c)
                    count_wrong_inject += 1
                else:
                    ans = str(a + b)
                # Answers can be of maximum size DIGITS + 1
                ans += ' ' * (DIGITS + 1 - len(ans))
                if INVERT:
                    query = query[::-1]
                questions.append(query)
                expected.append(ans)
            print('Total addition questions:', len(questions))
            print('Noise_level:', noise_rate, count_wrong_inject)
            print('Vectorization...')
             
            X = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
            y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
            for i, sentence in enumerate(questions):
                X[i] = ctable.encode(sentence, maxlen=MAXLEN)
            for i, sentence in enumerate(expected):
                y[i] = ctable.encode(sentence, maxlen=DIGITS + 1)
                     
            # Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
            # indices = np.arange(len(y))
            # np.random.shuffle(indices)
            # X = X[indices]
            # y = y[indices] 
                 
                 
            (X_train, X_test, X_val) = (slice_X(X, 0, TRAINING_SIZE), slice_X(X, TRAINING_SIZE, TRAINING_SIZE + SPLIT_AT), slice_X(X, TRAINING_SIZE + SPLIT_AT, DATA_SIZE))
            (y_train, y_test, y_val) = (y[0:TRAINING_SIZE], y[TRAINING_SIZE:TRAINING_SIZE + SPLIT_AT], y[TRAINING_SIZE + SPLIT_AT:DATA_SIZE])
                    
                 
                 
            f = open('qrel_' + str(noise_rate), 'w')
            for i in range(TRAINING_SIZE):    
                rowX, rowy = X_train[np.array([i])], y_train[np.array([i])]    
                q = ctable.decode(rowX[0])
                ans = ctable.decode(rowy[0])
                qid = 'Q1'  # + str(q[::-1] if INVERT else q)
                did = 'D' + str(q[::-1] if INVERT else q)
                rel = 0
                if i < count_wrong_inject:
                    rel = 1
                f.write(qid + ' iter ' + did + ' ' + str(rel) + '\n')       
                #print (q[::-1] if INVERT else q), print(' T', ans)
            f.close()
                     
                     
                            
            cPickle.dump(X_train, open('data_add/add_xtrain_' + str(noise_rate), 'wb'))
            cPickle.dump(y_train, open('data_add/add_ytrain_' + str(noise_rate), 'wb'))
            cPickle.dump(X_test, open('data_add/add_xtest_' + str(noise_rate), 'wb'))
            cPickle.dump(y_test, open('data_add/add_ytest_' + str(noise_rate), 'wb'))
            cPickle.dump(X_val, open('data_add/add_xval_' + str(noise_rate), 'wb'))
            cPickle.dump(y_val, open('data_add/add_yval_' + str(noise_rate), 'wb'))
              
              
              
              
            X_train = cPickle.load(open('data_add/add_xtrain_' + str(noise_rate), 'rb'))
            y_train = cPickle.load(open('data_add/add_ytrain_' + str(noise_rate), 'rb'))
            X_test = cPickle.load(open('data_add/add_xtest_' + str(noise_rate), 'rb'))
            y_test = cPickle.load(open('data_add/add_ytest_' + str(noise_rate), 'rb'))    
            X_val = cPickle.load(open('data_add/add_xval_' + str(noise_rate), 'rb'))
            y_val = cPickle.load(open('data_add/add_yval_' + str(noise_rate), 'rb'))
              
            print (X_train.shape)
            print (y_train.shape)
            print (X_test.shape)
            print (y_test.shape)
            print (X_val.shape)
            print (y_val.shape)


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

