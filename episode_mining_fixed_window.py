import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

red_hosts = ["T029-787","TLM83-15005823","DE5450-15006304","LX250-15006650","LX250-15006668",'VM-CSL-01', 'VM-CSL-02', 'VM-CSL-03']

def load_data(filename):
    selected_hosts = ['DE5450-15006304', 'VM-CSL-07', 'TLM93P-14014366', 'VM-CSL-10', 'TLM83-15005825', 'VM-CSL-03', 'DE5440-008388', 'VM-CSL-05', 'VM-CSL-11', 'TLM93P-14014370', 'T029-787', 'VM-CSL-01', 'VM-CSL-09', 'T029-449', 'VM-CSL-08', 'TLM83-15005832', 'TLM72-016266', 'VM-CSL-06', 'DE5450-15006348', 'VM-WIN7-03', 'VM-CSL-12', 'VM-CSL-02', 'VM-CSL-04', 'TLM72-016270', 'VM-WIN7-04', 'TLM83-15005823','LX250-15006650','LX250-15006668']
    total = []
    
    # Open and read file
    file = open(filename, 'r', encoding = 'utf-8', errors = 'ignore')
    data = file.read()
    file.close()
    
    # Split into lines
    data = data.split('\n')
    for i in data:
        i = i.split(':')
        host = i[0]
        if host == '':
            continue
        if host in selected_hosts:
            total += [i,]
    for j in total:
        j[-1] = j[-1].split(',')
        
    return total


def seq_gen(data, window_size, max_num_seq):
    sequences = []
    num_seq = 0
    for line in data:
        start_pos = 0
        encoded = tokeniser.texts_to_sequences([line])[0]           # Gives each log line in encoded format
        for i in range(window_size-1, len(encoded)):                    # Splits each log line into lines of size 2 to max_len
            sequence = encoded[start_pos:i+1]
            sequences.append(sequence)
            num_seq += 1
            if num_seq == max_num_seq:
                break
            start_pos += 1
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

def gen_xy(data, window_size, max_num_seq):
    sequences = seq_gen(data, window_size, max_num_seq)
    sequences = pad_sequences(sequences, maxlen = window_size, padding = 'pre')
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
    #config.gpu_options.per_process_gpu_memory_fraction = 0.20

    # Create a session with the above options specified.
    k.tensorflow_backend.set_session(tf.Session(config=config))
    ###################################

    tf.global_variables_initializer()
    
    early_stopping = EarlyStopping(monitor='loss', patience=2)
    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length = window_size-1))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(vocab_size, activation = 'softmax'))
    parallel_model = model
    #parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
    parallel_model.fit(x_train, y_train, epochs = num_epochs, verbose = 1, batch_size = batch_size, callbacks = [early_stopping])
    
    return parallel_model

startdate = '20160408'
num_training_days = 3
testdate = end_date_gen(startdate,num_training_days)

while testdate <= '20160502':
    
    #load training and testing data (users with sequence of event ids)
    train_data = data_gen(startdate,num_training_days)
    test_data = data_gen(testdate,1)
    total_data = train_data + test_data

    print('Number of training user-sequences:',len(train_data))
    print('Number of testing user-sequences:',len(test_data))
    
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
    print("Avg number of event ids per user per day: ", avg_len)
    
    max_len = int(avg_len)
    window_size = 5 #min sequence length for sliding window, not sure how to decide

    #generate training sequences with sliding window
    x_train, y_train = gen_xy(train_sequences, window_size, max_len)
    print("Length of x_train: ", len(x_train))
    
    #train model
    parallel_model = train_model(x_train, y_train, 5, 128)
    
    #generate test sequences
    sliding_test_sequences = []
    for log in test_sequences:
        test_sequence = seq_gen([log,],window_size,max_len)
        test_sequence = pad_sequences(test_sequence, maxlen = window_size, padding = 'pre')
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

    print ('Total number of user-sequences to process:',len(x_test))

    for log in range(len(x_test)):
        print ('Currently processing user-sequence',log)
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

    #print(avg_ce)
    #print(threshold)
    
    #determine anomalies
    flagged_red_hosts = []
    flagged_non_red_hosts = []

    outfile = "./results/window_size_5_results.txt"
    printline = "\n-- STATISTICS FOR " + str(testdate) + " --\n"
    with open (outfile, 'a') as writefile:
        writefile.write(printline + "\n")
    print(printline)

    for count in range(len(log_ce)):
        if log_ce[count] > threshold:
            user = test_users[count]
            if user in red_hosts:
                flagged_red_hosts += [user,]
            else:
                flagged_non_red_hosts += [user,]

            #find token with highest loss
            maxpos = np.argmax(log_ce)
            max_loss_token_index = y_test[count][maxpos]
            max_loss_token = str(reverse_word_map[max_loss_token_index])
            #print("max_loss_token:",max_loss_token)
            #print(x_test[count][maxpos])
                
            xline = []
            for i in x_test[count][maxpos]:
                try:
                    xline += [str(reverse_word_map[i]),]
                except KeyError:
                    continue

            with open (outfile, 'a') as writefile:
                writefile.write('ANOMALY DETECTED:\n')
                writefile.write('User:%s'%(user))
                writefile.write(' '.join(xline))
                writefile.write("Next token: " + max_loss_token + "\n")
            print('ANOMALY DETECTED:')
            print('User:%s'%(user))
            print(' '.join(xline))
            print("Next token:" + max_loss_token + "\n")
    
    #calculate statistics
    appearing_red_hosts = []
    for user in test_users:
        for red_host in red_hosts:
            if user == red_host:
                appearing_red_hosts += [red_host,]

    total_flagged = len(flagged_red_hosts) + len(flagged_non_red_hosts)
    total_logs = len(x_test)

    true_positives = len(flagged_red_hosts)
    false_positives = len(flagged_non_red_hosts)
    false_negatives = len(appearing_red_hosts)-len(flagged_red_hosts)
    true_negatives = total_logs-false_positives

    if true_positives+false_positives == 0:
        precision = 0.0
    else:
        precision = true_positives/(true_positives + false_positives)

    if true_positives+false_negatives == 0:
        recall = 0.0
    else:
        recall = true_positives/(true_positives+false_negatives)

    if precision == 0 or recall == 0:
        f1 = 0.0
    else:
        f1 = 2/((1/precision)+(1/recall))

    true_positive_rate = recall
    if false_positives+true_negatives == 0:
        false_positive_rate = 0
    else:
        false_positive_rate = false_positives / (false_positives+true_negatives)

    missed_red_hosts = [host for host in appearing_red_hosts if host not in flagged_red_hosts]
    with open (outfile, 'a') as writefile:
        writefile.write('No. of Flagged Users: %i \n' % total_flagged)
        writefile.write('Total No. of Users: %i \n\n' % total_logs)
        writefile.write('No. of True Positives: %i \n' % true_positives)
        writefile.write('No. of False Positives:  %i \n' % false_positives)
        writefile.write('No. of False Negatives: %i \n' % false_negatives)
        writefile.write('No. of True Negatives: %i \n\n' % true_negatives)
        writefile.write('Precision: %f \n' % precision)
        writefile.write('Recall: %f \n' % recall)
        writefile.write('F1 Score: %f \n' % f1)
        writefile.write('True Positive Rate:  %f \n' % true_positive_rate)
        writefile.write('False Positive Rate: %f \n\n' % false_positive_rate)
        writefile.write('Flagged red hosts: %s \n' % ','.join(flagged_red_hosts))
        writefile.write('Flagged non-red hosts: %s \n' % ','.join(flagged_non_red_hosts))
        writefile.write('Missed red hosts: %s \n' % ','.join(missed_red_hosts))
    
    print('No. of Flagged Users:',total_flagged)
    print('Total No. of Users:', total_logs)
    print('')
    print('No. of True Positives:',true_positives)
    print('No. of False Positives:',false_positives)
    print('No. of False Negatives:',false_negatives)
    print('No. of True Negatives:',true_negatives)
    print('')
    print('Precision:',precision)
    print('Recall:',recall)
    print('F1 Score:',f1)
    print('True Positive Rate:',true_positive_rate)
    print('False Positive Rate:',false_positive_rate)
    print('')
    print('Flagged red hosts: %s \n' % ','.join(flagged_red_hosts))
    print('Flagged non-red hosts: %s \n' % ','.join(flagged_non_red_hosts))
    print('Missed red hosts: %s \n' % ','.join(missed_red_hosts))

    startdate = end_date_gen(startdate,1)
    testdate = end_date_gen(testdate,1)
    