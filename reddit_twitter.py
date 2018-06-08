import numpy as np
import string 
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
import keras.preprocessing.text
import re
import random 

printable = set(string.printable)
def remove_hash_tags(text):
    hts = [part[1:] for part in text.split() if part.startswith('#')]
    for ht in hts:
        text = text.replace(ht, '')
    return text
def remove_urls(text):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for u in urls:
        text = text.replace(u, '')
    return text
def remove_users(text):
    users = re.findall("@([a-z0-9_]+)", text, re.I)
    for u in users:
        text = text.replace('@' + u, '')
    return text
def clean_text(text):
    text = remove_urls(text)
    text = remove_users(text.replace('@ ', '@'))
    # text = remove_hash_tags(text)    
    text = re.sub(r'\s+', ' ', text).strip()    
    text = filter(lambda x: x in printable, text)    
    text = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()
def extract_features(text):  
#     ngram_set = set()
#     terms = [t for t in text.split(' ')]
#     for ng in range(1, 3):
#         set_of_ngram = set(zip(*[terms[i:] for i in range(ng)]))
#         ngram_set.update(set_of_ngram)
#     ngrams = ' '.join('='.join(t) for t in ngram_set)
#     return ngrams
    return text
def get_data():
    # READ REDDIT OR TWITTER DATA
    texts = []
    labels = []
    # If Colon, Brain, or Breast cancer is mentioned only in passing without providing details about patient experience or history, the text is considered irrelevant (label = 0).
    for cancer in ['colon']:  # , 'brain', 'breast']:
        f = open('data_reddit/' + cancer + '_cf', 'r')
        documents = f.readlines()
        for i in range(0, len(documents)):
            text = extract_features(clean_text(documents[i].split('\t')[4]))
            lb = documents[i].split('\t')[2].strip().lower()
            texts.append(text)
            if lb == 'relevant':
                labels.append(1)
            else: # 'irrelevant'
                labels.append(0)
        f.close()
    print ('#pos: ', labels.count(1), '#neg:', labels.count(0))
    return texts, labels
def add_to_pool(did, text, label, pool, pool_label):
    if did in pool:
        if text != pool[did]:
            print 'same did, differnt text in pool!!'
            print (text, label)
            print (pool[did], pool_label[did])
    else:
        pool[did] = text
        pool_label[did] = label
    
# PARAMETERS
max_features = 50000
batch_size = 16
embedding_dims = 200
nb_epoch = 32
maxlen = 200
acc_thr = 1.  # accuracy threshold: if network accuracy against an instance is >= acc_thr, the spotter treats the instance as correctly classified, a smaller value than 1.0 can be used for flexibility.

# LOAD DATA        
print('Loading data...')
# READ REDDIT OR TWITTER DATA
texts, labels = get_data()
random.seed('315')
random.shuffle(texts)
random.seed('315')
random.shuffle(labels)
train_size = int(.9 * len(texts))
X_train = texts[:train_size]
y_train = labels[:train_size]
X_test = texts[train_size:]
y_test = labels[train_size:]
tokenizer = keras.preprocessing.text.Tokenizer(nb_words=max_features, filters='') 
tokenizer.fit_on_texts(X_train + X_test)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# BUILD THE NETWORK
print('Build model lit ...')
model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(GlobalAveragePooling1D())
model.add(Dense(256))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print (model.summary())


# SIMULTANEOUSLY TRAIN THE NETWORK & IDENTIFY SPURIOUS INSTANCES IN TRAINING DATASET.
# IF YOU WANT TO IDENTIFY ALL SPURIOUS INSTANCES IN YOUR DATASET, USE THE ENTIRE DATA FOR TRAINING
history, n_instances, q0_sorted = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                                          validation_data=(X_test, y_test), kern='lit', acc_thr=acc_thr)

# SHOW A RANKED LIST OF SPURIOUS INSTANCES
max_rank = 50  # show top max_rank spurious instances
print '\n--------------------------'
print 'ranked list of top', max_rank, 'spurious instances in dataset'
print 'label_in_dataset | instance'
rank = 1
for item in q0_sorted:
    if rank >= max_rank:
        break
    rank += 1
    rowX = X_train[np.array([item])]
    # pred = model.predict_classes(rowX, verbose=0)
    print labels[item], texts[item]
    print '----------------'
     
del model