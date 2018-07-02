# coding: utf-8

# In[1]:
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


# In[3]:


# Parses data into lists, each of the form
# [time, src user@domain, dst user@domain, src computer, dst computer, auth type, logon type, auth orientation, success/failure]
# '?' for fields that are unknown

def sec_to_day(sec):
    return math.ceil(int(sec)/(24*60*60))

def load_data(filename):
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
        total += [i,]
    return total

def compare_red(data,red):
    counter = 0
    for i in data:
        impt = i[0:2]+i[3:5]
        if impt in red:
            i += ['1',]        # malicious
            counter += 1
        else:
            i += ['0',]        # non malicious
    #print("Total Anomalies for User:",counter)
    return counter

def get_data(start, stop):
    # Load first 12 days of data
    data_files = []
    for i in range(start,stop):
        data_files += ["../data/user_day/redusers_day"+str(i)+".txt",]
    data = []
    for filename in data_files:
        data += load_data(filename)
    return data

def seq_gen(data,min_len):
    sequences = []
    for line in data:
        encoded = tokeniser.texts_to_sequences([line])[0]           # Gives each log line in encoded format
        for i in range(min_len-1, len(encoded)-1):                    # Splits each log line into lines of size 2 to max_len
            sequence = encoded[:i+1]
            sequences.append(sequence)
    return sequences

def gen_xy(data):
    sequences = seq_gen(data,2)     # Minimum length of 2
    sequences = pad_sequences(sequences, maxlen = max_len, padding = 'pre')
    sequences = array(sequences)
    x, y = sequences[:,:-1], sequences[:,-1]
    return x, y


# In[29]:


red = load_data('../data/redteam.txt')

def red_users(data):
    with open(data,'r') as infile:
        total_users = []
        threshold = 12*86400
        while True:
            line = infile.readline()
            line = line.split(',')
            try:
                user = line[1]
                day = sec_to_day(line[0])
                if user not in total_users and day >= 13:
                    total_users += [user,]
            except ValueError:
                break
            except IndexError:
                break
    return total_users

redusers = red_users('../data/redteam.txt')
#redusers = ['U9263@DOM1', 'U655@DOM1','U3005@DOM1']
#print (len(redusers))
filenames = []
for i in redusers:
    filenames += ['../data/online_user/redusers_' + i + '.txt',]
#print(filenames)

curr_stop = 1
curr_test = 13
max_len = 8
outfile = './output/dataresults.txt'

def load_model():
    model = model_from_json(open('onlineplutomodel.json').read())
    model.load_weights('onlineplutoweights.h5')
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model

def save_model(model):    
    json_string = model.to_json()
    open('onlineplutomodel.json', 'w').write(json_string)
    model.save_weights('onlineplutoweights.h5', overwrite=True)

total_true_positive = 0
total_false_positive = 0
total_false_negative = 0
total_true_negative = 0

counter = 1
while counter < len(filenames):

    if curr_test > 58:
        print('Total Predicted Positives for User ' + redusers[counter] + ": " + str(total_true_positive))
        print('Total Actual Positives for User ' + redusers[counter] + ": " + str(user_red))
        print('Percentage of Actual Positives Caught:',total_true_positive/user_red)

        if total_true_positive+total_false_positive == 0:
            total_precision = 0.0
        else:
            total_precision = total_true_positive/(total_true_positive + total_false_positive)

        if total_true_positive+total_false_negative == 0:
            total_recall = 0.0
        else:
            total_recall = total_true_positive/(total_true_positive+total_false_negative)

        if total_precision == 0 or total_recall == 0:
            total_f1 = 0.0
        else:
            total_f1 = 2/((1/total_precision)+(1/total_recall))

        total_true_positive_rate = total_recall
        if total_false_positive+total_true_negative == 0:
            total_false_positive_rate = 0
        else:
            total_false_positive_rate = total_false_positive / (total_false_positive+total_true_negative)

        print('Total False Positives for User ' + redusers[counter] + ": " + str(total_false_positive))
        print('Total False Negatives for User ' + redusers[counter] + ": " + str(total_false_negative))
        print('Total True Negatives for User ' + redusers[counter] + ": " + str(total_true_negative))
        
        print('Total Precision for User ' + redusers[counter] + ": " + str(total_precision))
        print('Total Recall for User' + redusers[counter] + ": " + str(total_recall))
        print('Total F1 Score for User' + redusers[counter] + ": " + str(total_f1))
        print('Total True Positive Rate for User' + redusers[counter] + ": " + str(total_true_positive_rate))
        print('Total False Positive Rate for User' + redusers[counter] + ": " + str(total_false_positive_rate))
        print('')

        with open(outfile,'a') as writefile:
            writefile.write('Total Predicted Positives for User ' + redusers[counter] + ": " + str(total_true_positive) + "\n")
            writefile.write('Total Actual Positives for User ' + redusers[counter] + ": " + str(user_red) + "\n")
            writefile.write('Total False Positives for User ' + redusers[counter] + ": " + str(total_false_positive) + "\n")
            writefile.write('Total False Negatives for User ' + redusers[counter] + ": " + str(total_false_negative) + "\n")
            writefile.write('Total True Negatives for User ' + redusers[counter] + ": " + str(total_true_negative) + "\n")
            writefile.write('Total Precision for User ' + redusers[counter] + ": " + str(total_precision) + "\n")
            writefile.write('Total Recall for User' + redusers[counter] + ": " + str(total_recall) + "\n")
            writefile.write('Total F1 Score for User' + redusers[counter] + ": " + str(total_f1) + "\n")
            writefile.write('Total True Positive Rate for User' + redusers[counter] + ": " + str(total_true_positive_rate) + "\n")
            writefile.write('Total False Positive Rate for User' + redusers[counter] + ": " + str(total_false_positive_rate) + "\n\n")

        total_true_positive = 0
        total_false_positive = 0
        total_false_negative = 0
        total_true_negative = 0
        curr_stop = 1
        curr_test = 13
        counter += 1
        continue
    
    early_stopping = EarlyStopping(monitor='loss', patience=2)
    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

    filelink = filenames[counter]
    total_data = load_data(filelink)
    user_red = compare_red(total_data,red)
    for i in total_data:
        day = math.ceil(int(i[0])/(60*60*24))
        i[0] = str(day)

    # Split train_data by day
    train_data = []
    test_data = []
    for i in total_data:
        if int(i[0]) <= curr_test and int(i[0]) >= curr_stop and i[-1] == '0':    # Avoid training on anomalous data
            train_data += [i,]
        if int(i[0]) == curr_test:
            test_data += [i,]

    if test_data == []:
        printline = "NO DATA FOR USER " + redusers[counter] + " ON DAY " + str(curr_test)
        print(printline)
        curr_test += 1
        with open(outfile,'a') as writefile:
            writefile.write(printline + "\n")
        continue

    for i in train_data:
        i.remove(i[0])      # Remove day

    red_label = []
    red_events = 0
    for i in test_data:
        i.remove(i[0])
        red_label += [i[-1],]
        if i[-1] == '1':
            red_events += 1
        i.remove(i[-1])
    print("No. of Anomalies for User " + redusers[counter] + " ON DAY " +  str(curr_test) + ": " + str(red_events))

    if red_events == 0:
        curr_stop += 1
        curr_test += 1
        continue

    tokeniser = Tokenizer(filters = '\n')
    tokeniser.fit_on_texts(train_data)
    vocab_size = len(tokeniser.word_index)+1
    #reverse_word_map = dict(map(reversed, tokeniser.word_index.items()))
    x_train, y_train = gen_xy(train_data)

    test_sequences = []
    for log in test_data:
        test_sequence = seq_gen([log,],2)
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
    #x_test, y_test = test_sequences[:,:,:-1], test_sequences[:,:,-1]

    ###################################
    # TensorFlow wizardry
    config = tf.ConfigProto(device_count={'GPU': 1})
     
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
     
    # Only allow a total of half the GPU memory to be allocated
    #config.gpu_options.per_process_gpu_memory_fraction = 0.3
     
    # Create a session with the above options specified.
    k.tensorflow_backend.set_session(tf.Session(config=config))
    ###################################

    #tf.global_variables_initializer()

    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length = max_len-1))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(vocab_size, activation = 'softmax'))
    #parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model = model
    parallel_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
    parallel_model.fit(x_train, y_train, epochs = 20, verbose = 1, batch_size = 512, callbacks = [early_stopping])

    save_model(parallel_model)

    printline = "\nSTATISTICS FOR USER " + redusers[counter] + " ON DAY " + str(curr_test) + "\n"
    print(printline)
    with open(outfile,'a') as writefile:
        writefile.write(printline)
    
    def CrossEntropy(y):
        return -np.log(y)

    log_ce = []
    loss = 0
    ce = 0
    #counter = 0
    
    for log in range(len(x_test)):
        if log%10000 == 0:
            print("detecting anomaly for log line ", log)
        ypred = parallel_model.predict(x_test[log])
        correct = y_test[log]
        loss = 0
        ce = 0
        counts = 0
        for i in range(len(correct)-1):
            pos = correct[i]
            y = ypred[i][pos]
            if y == 0:
                print("y=0!")
                counts += 1
                ce += 1
            else:
                counts += 1
                ce += CrossEntropy(y)
        if counts == 0:
            continue
        else:
            log_ce += [ce,]
    avg_ce = np.mean(log_ce)
    #sum_ce = np.sum(log_ce)
    std_dev = np.std(log_ce)
    threshold = avg_ce + (2*std_dev)

    true_positive = 0    # Total correct anomalies detected
    anomalies = 0

    for count in range(len(log_ce)):
        if log_ce[count] > threshold:
            if red_label[count] == '1':
                true_positive += 1
            anomalies += 1

    false_positive = anomalies-true_positive
    false_negative = red_events-true_positive
    true_negative = len(x_test)-true_positive-false_positive-false_negative

    if true_positive+false_positive == 0:
        precision = 0.0
    else:
        precision = true_positive/(true_positive + false_positive)

    if true_positive+false_negative == 0:
        recall = 0.0
    else:
        recall = true_positive/(true_positive+false_negative)

    if precision == 0 or recall == 0:
        f1 = 0.0
    else:
        f1 = 2/((1/precision)+(1/recall))

    true_positive_rate = recall
    if false_positive+true_negative == 0:
        false_positive_rate = 0
    else:
        false_positive_rate = false_positive / (false_positive+true_negative)

    with open(outfile,'a') as writefile:
        writefile.write('True Positive: ' + str(true_positive) + "\n")
        writefile.write('False Positive: ' + str(false_positive) + "\n")
        writefile.write('False Negative: ' + str(false_negative) + "\n")
        writefile.write('True Negative: ' + str(true_negative) + "\n")
        writefile.write('Precision: ' + str(precision) + "\n")
        writefile.write('Recall: ' + str(recall) + "\n")
        writefile.write('F1 Score: ' + str(f1) + "\n")
        writefile.write('True Positive Rate: ' + str(true_positive_rate) + "\n")
        writefile.write('False Positive Rate: ' + str(false_positive_rate) + "\n\n")

    print('True Positive:', true_positive)
    total_true_positive += true_positive
    print('False Positive:', false_positive)
    total_false_positive += false_positive
    print('False Negative:', false_negative)
    total_false_negative += false_negative
    print('True Negative:', true_negative)
    total_true_negative += true_negative
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)
    print('True Positive Rate:',true_positive_rate)
    print('False Positive Rate:',false_positive_rate)
    print('')

    curr_stop += 1
    curr_test += 1