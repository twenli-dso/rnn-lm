import time
import json
import re
import os
import glob

file_list = glob.glob(os.path.join('../extract_features/output/','*'))
#, '*'
#test_list = glob.glob(os.path.join(os.getcwd(),'TestData'))
#print(os.path.join(os.getcwd(),'TrainData', '*'))
#print(file_list)

traincounter = 0
testcounter = 0
allhosts = {}

def end_date_gen(start_date,numdays):
    date = datetime.datetime.strptime(start_date, "%Y%m%d")
    end_date = date + datetime.timedelta(days=numdays)
    return end_date.strftime('%Y%m%d')

#file_list = ['1457598574988-1_2016-03-10']

for url in file_list:
    try:
        with open(url) as data_file:
            while True:
                try:
                    data = data_file.readline().strip()
                    print(data)
                    if data == '':
                        break
                    d = json.loads(data)
                    eventtype = d['Log']['Event']['Name']
                    if eventtype == 'Security':
                        eventid = d['Log']['Event']['EventID']
                        hostname = d['Log']['HostName']
                        timestamp = d['Log']['Event']['Timestamp-Readable-UTC']
                        logtime = timestamp[:10].replace('-','')
                        if hostname in allhosts:
                            if logtime in allhosts[hostname]:
                                allhosts[hostname][logtime] += [eventid,]
                            else:
                                allhosts[hostname][logtime] = [eventid]
                        else:
                            allhosts[hostname] = {logtime:[eventid,]}
                        traincounter += 1

                        if traincounter%10000 == 1:
                            print('Current length:', traincounter)
                            #print(allhosts)
                            
                except KeyError:
                    continue
                except UnicodeDecodeError:
                    continue
                except json.decoder.JSONDecodeError:
                    continue
    except IsADirectoryError:
        continue
print('Logs:',traincounter)
#print(allhosts)

for host in allhosts:
    for date in allhosts[host]:
        #print(allhosts[host][date])
        seq = host + ':' + ','.join(str(i) for i in allhosts[host][date])
        fileurl = './output/episode_mining/sec_' + date + '.txt'
        with open(fileurl,'a') as outfile:
            outfile.write(seq + '\n')
