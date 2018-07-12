
# coding: utf-8

# In[1]:

'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
'''

import string
import datetime
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

#notinlist = ['20160407','20160410','20160427']
outfile = "./output/production_results_20hosts.txt"
# In[16]:


def load_data(filename):
    twenty_hosts = ["T029-787","TLM83-15005823","DE5450-15006304","LX250-15006650","LX250-15006668",'VM-CSL-01', 'VM-CSL-02', 'VM-CSL-03',"TLM83-15005832","LX250-15006645", "DE5450-15006348","LX250-15006652","TLM83-15005825", "DE5450-15006348", "TLM83-15005832","DE5440-008388","LX250-15006643","VM-CSL-04","T029-787","T062-253","DE5450-15006328"]

    total = []
    
    # Open and read file
    file = open(filename, 'r', encoding = 'utf-8', errors = 'ignore')
    data = file.read()
    file.close()
    
    # Split into lines
    data = data.split('\n')
    for i in data:
        i = i.split(',')
        if i[0] == '':
            continue
        if i[1] in twenty_hosts:
            total += [i,]
    return total

def seq_gen(data,min_len):
    sequences = []
    for line in data:
        encoded = tokeniser.texts_to_sequences([line])[0]           # Gives each log line in encoded format
        for i in range(min_len-1, len(encoded)):                    # Splits each log line into lines of size 2 to max_len
            sequence = encoded[:i+1]
            sequences.append(sequence)
    return sequences

def gen_xy(data):
    sequences = seq_gen(data,3)     # Minimum length of 2
    sequences = pad_sequences(sequences, maxlen = max_len, padding = 'pre')
    sequences = array(sequences)
    x, y = sequences[:,:-1], sequences[:,-1]
    return x, y

def end_date_gen(start_date,numdays):
    date = datetime.datetime.strptime(start_date, "%Y%m%d")
    end_date = date + datetime.timedelta(days=numdays)
    return end_date.strftime('%Y%m%d')

def data_gen(start_date,numdays):
    fulldata = []
    i = 1
    day = start_date
    while i <= numdays:
        if day in notinlist:
            day = end_date_gen(day,1)
            continue
        else:
            dayurl = '../extract_features/output/data_security_' + day + '.txt'
            fulldata += load_data(dayurl)
            day = end_date_gen(day,1)
            i += 1
    return fulldata

# In[19]:

red_data = load_data('gtruth.txt')
red = []

startdate = '20160414'
testdate = end_date_gen(startdate,3)

while testdate <= '20160502':

    train_data = data_gen(startdate,4)
    test_data = data_gen(testdate,1)
    total_data = train_data + test_data

    print('Number of training logs:',len(train_data))
    print('Number of testing logs:',len(test_data))

    if len(test_data) == 0:
        print('NO TEST DATA ON',testdate)
        startdate = end_date_gen(startdate,1)
        enddate = end_date_gen(enddate,1)
        testdate = end_date_gen(testdate,1)
        continue

    true_red = []
    for i in red_data:
        hosts = i[4:]
        new = i[0:4] + [hosts,]
        true_red += [new,]

    for log in test_data:
        counter = 0
        for redlog in true_red:
            if (log[0] <= redlog[2] and log[0] >= redlog[3]):
                if log[1] in redlog[-1]:
                    red += ['1',]
                    counter = 1
                    break
        if counter == 0:
            red += ['0',]

    totalanomalies = 0
    for i in red:
        if i == '1':
            totalanomalies += 1
    print('Total True Anomalies:',totalanomalies)

    max_len = 9
    early_stopping = EarlyStopping(monitor='loss', patience=2)
    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

    tokeniser = Tokenizer(filters = '\n')
    tokeniser.fit_on_texts(total_data)
    vocab_size = len(tokeniser.word_index)+1
    reverse_word_map = dict(map(reversed, tokeniser.word_index.items()))
    x_train, y_train = gen_xy(train_data)

    test_sequences = []    
    for log in test_data:
        test_sequence = seq_gen([log,],3)
        test_sequence = pad_sequences(test_sequence, maxlen = max_len, padding = 'pre')
        test_sequences += [test_sequence,]
    test_sequences = array(test_sequences)
    x_test = []
    y_test = []
    for i in test_sequences:
        x_test += [i[:,:-1],]
        y_test += [i[:,-1],]
    x_test = array(x_test)
    y_test = array(y_test)

    #print(x_test[0:20])

    # In[ ]:

    def save_model(model):    
        json_string = model.to_json()
        open('modelProdS.json', 'w').write(json_string)
        model.save_weights('weightsProdS.h5', overwrite=True)


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

    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length = max_len-1))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(vocab_size, activation = 'softmax'))
    #parallel_model = model
    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
    parallel_model.fit(x_train, y_train, epochs = 5, verbose = 1, batch_size = 512, callbacks = [early_stopping])

    save_model(parallel_model)

    printline = "\n-- STATISTICS FOR " + str(testdate) + " --\n"
    with open (outfile, 'a') as writefile:
        writefile.write(printline + "\n")
    print(printline)

    def CrossEntropy(y):
        return -np.log(y)

    log_ce = []
    loss = 0
    ce = 0
    #counter = 0
    true_positives = 0

    print ('Total number of logs to process:',len(x_test))

    for log in range(len(x_test)):
        if log % 1000 == 0:
            print ('Currently processing log',log)
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
    anomalies = 0

    with open (outfile, 'a') as writefile:
        writefile.write('Length of log_ce: %i \n' % len(log_ce))
        writefile.write('Length of red: %i \n' % len(red))
        writefile.write('Length of x_test: %i \n' % len(x_test))
    print('Length of log_ce:',len(log_ce))
    print('Length of red:',len(red))
    print('Length of x_test:',len(x_test))

    for count in range(len(log_ce)):
        if log_ce[count] > threshold:
            if red[count] == '1':
                true_positives += 1
            with open (outfile, 'a') as writefile:
                writefile.write('ANOMALY DETECTED:\n')
            print('ANOMALY DETECTED:')
            xline = []
            for i in x_test[count][-1]:
                try:
                    xline += [str(reverse_word_map[i]),]
                except KeyError:
                    continue

            with open (outfile, 'a') as writefile:
                writefile.write(' '.join(xline) + '\n')
            print(' '.join(xline) + '\n')
            anomalies += 1

    total_flagged = anomalies
    total_logs = len(x_test)

    false_positives = total_flagged-true_positives
    false_negatives = totalanomalies-true_positives
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

    with open (outfile, 'a') as writefile:
        writefile.write('No. of Flagged Logs: %i \n' % total_flagged)
        writefile.write('Total No. of Logs: %i \n\n' % total_logs)
        writefile.write('No. of True Positives: %i \n' % true_positives)
        writefile.write('No. of False Positives:  %i \n' % false_positives)
        writefile.write('No. of False Negatives: %i \n' % false_negatives)
        writefile.write('No. of True Negatives: %i \n\n' % true_negatives)
        writefile.write('Precision: %f \n' % precision)
        writefile.write('Recall: %f \n' % recall)
        writefile.write('F1 Score: %f \n' % f1)
        writefile.write('True Positive Rate:  %f \n' % true_positive_rate)
        writefile.write('False Positive Rate: %f \n' % false_positive_rate)

    print('No. of Flagged Logs:',total_flagged)
    print('Total No. of Logs:', total_logs)
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

    startdate = end_date_gen(startdate,1)
    testdate = end_date_gen(testdate,1)
