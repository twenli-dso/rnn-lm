import string
import itertools
import math
import random
import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Bidirectional
#from pickle import load
#from pickle import dump
from keras.models import load_model
from keras.models import model_from_json
from keras.callbacks import Callback
from keras.layers import Dropout
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras import backend as k
import datetime

def load_data(filename):
    total = []
    
    # Open and read file
    file = open(filename, 'r', encoding = 'utf-8', errors = 'ignore')
    data = file.read()
    file.close()
    
    # Split into lines
    data = data.split('\n')
    for i in data:
        i = i.split(':')
        if i[0] == '':
            continue
        total += [i,]
    for j in total:
        j[-1] = j[-1].split(',')
        
    return total

def seq_gen(data, min_len, max_num_seq):
    sequences = []
    num_seq = 0
    for line in data:
        encoded = tokeniser.texts_to_sequences([line])[0]           # Gives each log line in encoded format
        for i in range(min_len-1, len(encoded)):                    # Splits each log line into lines of size 2 to max_len
            sequence = encoded[:i+1]
            sequences.append(sequence)
            num_seq += 1
            if num_seq == max_num_seq:
                break
    #while num_seq < max_num_seq:
        #sequences.append([0]*max_len)
        #num_seq += 1
    return sequences

def end_date_gen(start_date,numdays):
    date = datetime.datetime.strptime(start_date, "%Y%m%d")
    end_date = date + datetime.timedelta(days=numdays)
    return end_date.strftime('%Y%m%d')

def data_gen(start_date,numdays):
    fulldata = []
    i = 1
    day = start_date
    while i <= numdays:
        dayurl = './output/episode_mining/sec_' + day + '.txt'
        fulldata += load_data(dayurl)
        day = end_date_gen(day,1)
        i += 1
    return fulldata

def gen_xy(data, max_num_seq):
    min_len = 10
    sequences = seq_gen(data, min_len, max_num_seq)
    sequences = pad_sequences(sequences, maxlen = max_num_seq, padding = 'pre')
    sequences = array(sequences)
    x, y = sequences[:,:-1], sequences[:,-1]
    return x, y

def train_model(x_train, y_train, num_epochs, batch_size):
    ###################################
    # TensorFlow wizardry
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.25

    # Create a session with the above options specified.
    k.tensorflow_backend.set_session(tf.Session(config=config))
    ###################################

    tf.global_variables_initializer()
    
    early_stopping = EarlyStopping(monitor='loss', patience=2)
    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length = max_len-1))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(vocab_size, activation = 'softmax'))
    #parallel_model = model
    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
    parallel_model.fit(x_train, y_train, epochs = num_epochs, verbose = 1, batch_size = batch_size, callbacks = [early_stopping])
    
    return parallel_model

startdate = '20160408'
num_training_days = 6
testdate = end_date_gen(startdate,num_training_days)

while testdate <= '20160502':
    
    #load training and testing data (users with sequence of event ids)
    train_data = data_gen(startdate,num_training_days) #replace 1 with num_training_days
    test_data = data_gen(testdate,1)
    total_data = train_data + test_data

    print('Number of training users:',len(train_data))
    print('Number of testing users:',len(test_data))
    
    #check if there is testing data
    if len(test_data) == 0:
        print('NO TEST DATA ON',testdate)
        startdate = end_date_gen(startdate,1)
        enddate = end_date_gen(enddate,1)
        testdate = end_date_gen(testdate,1)
        continue

    #split data into sequence of event ids and users
    train_users, train_sequences = list((x[0] for x in train_data)), list((x[1] for x in train_data))
    test_users, test_sequences = list((x[0] for x in test_data)), list((x[1] for x in test_data))
    total_sequences = list((x[1] for x in total_data))

    #tokenize all sequence of event ids
    tokeniser = Tokenizer(filters = '\n')
    tokeniser.fit_on_texts(total_sequences)
    vocab_size = len(tokeniser.word_index)+1
    reverse_word_map = dict(map(reversed, tokeniser.word_index.items()))

    #get avg number of event ids per user (to determine max sequence length when generating sequences)
    total_len = 0
    for sequence in train_sequences:
        total_len += len(sequence)
    avg_len = total_len/len(train_sequences)
    max_len = int(avg_len)
    
    #generate training sequences with sliding window
    x_train, y_train = gen_xy(train_sequences, max_len)
    print(x_train)
    
    #train model
    model = train_model(x_train, y_train, 10, 128)
    
    #generate test sequences
    sliding_test_sequences = []
    for log in test_sequences:
        test_sequence = seq_gen([log,],10,max_len)
        test_sequence = pad_sequences(test_sequence, maxlen = max_len, padding = 'pre')
        sliding_test_sequences += [test_sequence,]
    sliding_test_sequences = array(sliding_test_sequences)
    x_test = []
    y_test = []
    for i in sliding_test_sequences:
        x_test += [i[:,:-1],]
        y_test += [i[:,-1],]
    x_test = array(x_test)
    y_test = array(y_test)
    
    #calculate loss per user and loss threshold
    def CrossEntropy(y):
        return -np.log(y)

    log_ce = []
    loss = 0
    ce = 0

    print ('Total number of logs to process:',len(x_test))

    for log in range(len(x_test)):
        #if log % 1000 == 0:
            #print ('Currently processing log',log)
        ypred = parallel_model.predict(x_test[log])
        correct = y_test[log]
        loss = 0
        ce = 0
        counts = 0
        for i in range(len(correct)-1):
            pos = correct[i]
            y = ypred[i][pos]
            if y == 0:
    ##            print("y=0!")
                counts += 1
                ce += 1
            else:
                counts += 1
                ce += CrossEntropy(y)
        log_ce += [ce,] # ce/counts

    avg_ce = np.mean(log_ce)
    std_dev = np.std(log_ce)
    threshold = avg_ce + (2*std_dev)

    print(avg_ce)
    print(threshold)
    
    #determine anomalies
    outfile = "./results/flagged_lines.txt"
    for count in range(len(log_ce)):
        if log_ce[count] > threshold:
            user = test_users[count]
            xline = []
            for i in x_test[count][-1]:
                try:
                    xline += [str(reverse_word_map[i]),]
                except KeyError:
                    continue

            with open (outfile, 'a') as writefile:
                writefile.write('ANOMALY DETECTED:\n')
                writefile.write('User:%s\n'%(user))
                writefile.write(' '.join(xline) + '\n')
            print('ANOMALY DETECTED:')
            print('User:%s\n'%(user))
            print(' '.join(xline) + '\n')
    
    startdate = end_date_gen(startdate,1)
    testdate = end_date_gen(testdate,1)
    